# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "einops>=0.8.2",
#     "flax>=0.12.6",
#     "jax>=0.9.2",
#     "mnist1d>=0.0.2.post1",
#     "optax>=0.2.8",
#     "simple-parsing>=0.1.8",
# ]
# ///

# A minimal toy implementation of LeJEPA on the mnist1d dataset.

import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange, reduce, repeat
import optax
from optax import softmax_cross_entropy_with_integer_labels as softmax_ce
from functools import partial
from dataclasses import dataclass
import mnist1d
import simple_parsing


def sigreg_loss(embs, n_slices, rngs):
    v, d = embs.shape[-2:]  # last two dims must be views, embedding dim

    A = jax.random.normal(rngs.next(), (d, n_slices))
    A = A / jnp.linalg.norm(A, axis=0, keepdims=True)
    projs = embs @ A

    # theoretical gaussian CF and w(t) weighting
    t = jnp.linspace(-3, 3, 17)
    exp_f = jnp.exp(-0.5 * t**2)

    # Using exp(ix) = cos(x) + i sin(x) and sep error terms to avoid complex numbers
    x_t = rearrange(projs, "b v m -> b v m 1") * t

    # average over batch (MC estimate), keep views
    x_t_cos = reduce(jnp.cos(x_t), "b v m t -> v m t", "mean")
    x_t_sin = reduce(jnp.sin(x_t), "b v m t -> v m t", "mean")

    err = jnp.square(x_t_cos - exp_f) + jnp.square(x_t_sin - 0)

    EP = v * jnp.trapezoid(err * exp_f, t, axis=-1)
    return reduce(EP, "v m -> ", "mean")


def lejepa_loss(encoder, views, n_slices, lamb, rngs):

    embs, loss_embs = encoder(views)

    centers = reduce(loss_embs, "b v d -> b 1 d", "mean")
    pred_loss = jnp.square(loss_embs - centers).mean()
    reg_loss = sigreg_loss(loss_embs, n_slices=n_slices, rngs=rngs)

    loss = lamb * reg_loss + (1 - lamb) * pred_loss
    return loss, (embs, loss_embs, pred_loss, reg_loss)


lejepa_grad = nnx.value_and_grad(lejepa_loss, has_aux=True)


def gen_views(x, n_views, rngs):
    bs, seq_len = x.shape

    # Mirrors mnist1d's generative transforms
    def one_view(key):
        k_shift, k_scale, k_corr, k_iid, k_mask_pos, k_mask_len, k_shear = (
            jax.random.split(key, 7)
        )

        # shift
        shifts = jax.random.randint(k_shift, (bs,), 0, seq_len)
        v = jax.vmap(lambda xi, s: jnp.roll(xi, s))(x, shifts)

        # amplitude scale (per-example)
        scale = 1.0 + 0.3 * jax.random.normal(k_scale, (bs, 1))
        v = v * scale

        # correlated (low-frequency) gaussian-smoothed noise
        raw = jax.random.normal(k_corr, (bs, seq_len))
        kernel = jnp.ones((7,)) / 7
        corr = jax.vmap(lambda r: jnp.convolve(r, kernel, mode="same"))(raw)
        v = v + 0.25 * corr

        # iid noise
        v = v + 0.05 * jax.random.normal(k_iid, (bs, seq_len))

        # masking: zero out a contiguous chunk (length-invariance)
        mask_len = jax.random.randint(k_mask_len, (bs,), 0, 8)  # 0-7 zeros
        mask_pos = jax.random.randint(k_mask_pos, (bs,), 0, seq_len)
        idx = jnp.arange(seq_len)[None, :]  # (1, seq_len)
        mask = (idx >= mask_pos[:, None]) & (idx < (mask_pos + mask_len)[:, None])
        v = jnp.where(mask, 0.0, v)

        # shear: subtract random linear ramp (mnist1d uses scale=0.75)
        coeff = 0.75 * (jax.random.uniform(k_shear, (bs, 1)) - 0.5)
        v = v - coeff * jnp.linspace(-0.5, 0.5, seq_len)

        return v

    keys = jax.random.split(rngs.next(), n_views)
    views = jax.vmap(one_view)(keys)
    return rearrange(views, "v b l -> b v l")


def test_acc(model, loader):
    correct, total = 0, 0
    for x, y in loader:
        correct += (jnp.argmax(model(x), axis=-1) == y).sum()
        total += len(y)
    return correct / total


@jax.jit
def geom(embs):
    d = embs.shape[-1]
    flat = rearrange(embs, "b v d -> (b v) d")
    per_dim_std = jnp.std(flat, axis=0)
    eigs = jnp.maximum(jnp.linalg.eigvalsh(jnp.cov(flat, rowvar=False)), 0.0)
    eff_rank = eigs.sum() ** 2 / (jnp.square(eigs).sum() + 1e-12)
    return per_dim_std.min(), per_dim_std.mean(), eff_rank / d


@dataclass
class Config:
    n_slices: int = 128
    n_views: int = 4
    lamb: float = 0.1
    projector: bool = True

    emb_dim: int = 128
    proj_dim: int = 16
    h_dim: int = 64

    bs: int = 32
    lr: float = 1e-3
    steps: int = 50_000
    seed: int = 0


if __name__ == "__main__":
    cfg = simple_parsing.parse(Config)
    rngs = nnx.Rngs(cfg.seed)
    ds = mnist1d.data.make_dataset()

    def train_loader():
        while True:
            for i in range(len(ds["x"]) // cfg.bs):
                yield (
                    ds["x"][i * cfg.bs : (i + 1) * cfg.bs],
                    ds["y"][i * cfg.bs : (i + 1) * cfg.bs],
                )

    def test_loader():
        for i in range(len(ds["x_test"]) // cfg.bs):
            yield (
                ds["x_test"][i * cfg.bs : (i + 1) * cfg.bs],
                ds["y_test"][i * cfg.bs : (i + 1) * cfg.bs],
            )

    class Encoder(nnx.Module):
        def __init__(self, cfg, rngs):

            conv = partial(nnx.Conv, kernel_size=(5,), padding="SAME", rngs=rngs)
            self.backbone = nnx.Sequential(
                partial(rearrange, pattern="b ... l -> b ... l 1"),
                conv(1, cfg.h_dim),
                nnx.relu,
                conv(cfg.h_dim, cfg.h_dim, strides=(2,)),
                nnx.relu,
                conv(cfg.h_dim, cfg.emb_dim, strides=(2,)),
                partial(reduce, pattern="b ... l d -> b ... d", reduction="mean"),
            )
            self.projector = (
                nnx.Linear(cfg.emb_dim, cfg.proj_dim, rngs=rngs)
                if cfg.projector
                else None
            )

        def __call__(self, views):
            embs = self.backbone(views)
            loss_embs = self.projector(embs) if self.projector else embs
            return embs, loss_embs

    enc = Encoder(cfg, rngs)
    enc_opt = nnx.Optimizer(enc, optax.adamw(cfg.lr), wrt=nnx.Param)

    probe = nnx.Linear(cfg.emb_dim, 10, rngs=rngs)
    probe_grad_fn = nnx.value_and_grad(
        lambda probe, embs, y: softmax_ce(
            logits=rearrange(probe(embs), "b v d -> (b v) d"),
            labels=repeat(y, "b -> (b v)", v=cfg.n_views),
        ).mean()
    )
    probe_opt = nnx.Optimizer(probe, optax.adamw(cfg.lr), wrt=nnx.Param)

    # supervised baseline: cross-entropy classifier without stop gradient
    clf = nnx.Sequential(nnx.clone(enc.backbone), nnx.clone(probe))
    clf_grad_fn = nnx.value_and_grad(lambda clf, x, y: softmax_ce(clf(x), y).mean())
    clf_opt = nnx.Optimizer(clf, optax.adamw(cfg.lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(enc, probe, clf, enc_opt, probe_opt, clf_opt, x, y, rngs):

        views = gen_views(x, n_views=cfg.n_views, rngs=rngs)
        ((_, (embs, loss_embs, pred_loss, reg_loss)), enc_grads) = lejepa_grad(
            enc, views=views, n_slices=cfg.n_slices, lamb=cfg.lamb, rngs=rngs
        )
        enc_opt.update(enc, enc_grads)

        # probe is just a diagnostic, so we don't backprop through the encoder
        embs = jax.lax.stop_gradient(embs)
        probe_loss, probe_grads = probe_grad_fn(probe, embs, y)
        probe_opt.update(probe, probe_grads)

        # classifier baseline
        clf_loss, clf_grads = clf_grad_fn(clf, x, y)
        clf_opt.update(clf, clf_grads)

        return pred_loss, reg_loss, probe_loss, clf_loss, embs, loss_embs

    for step, (x, y) in enumerate(train_loader()):
        pred_loss, reg_loss, probe_loss, clf_loss, embs, loss_embs = train_step(
            enc, probe, clf, enc_opt, probe_opt, clf_opt, x, y, rngs
        )

        if step % 100 == 0:
            std_min, std_mean, iso = geom(embs)
            _, _, pj_iso = geom(loss_embs)
            print(
                f"step={step} pred={pred_loss:.4f} reg={reg_loss:.4f} "
                f"probe={probe_loss:.4f} probe_acc={test_acc(nnx.Sequential(enc.backbone, probe), test_loader()):.4f} "
                f"clf={clf_loss:.4f} clf_acc={test_acc(clf, test_loader()):.4f} "
                f"std_min={std_min:.3f} std_mean={std_mean:.3f} iso={iso:.3f} pj_iso={pj_iso:.3f}"
            )

        if step >= cfg.steps:
            break

    print("done")
