#!/usr/bin/env bash
set -euo pipefail

# Sweep aug_coef × projector on/off to test whether augmentation intensity
# determines how much the projector matters (alignment hypothesis).
# Fixed at default hyperparams; vary aug over a wide range to find the cliff.

mkdir -p logs_aug

extras=()
[[ "${TPU:-0}" == "1" ]] && extras=(--with "jax[tpu]")

lr=1e-3
lamb=0.01
emb_dim=128
n_views=4
steps=50_000

for seed in 0 1 2; do
  for aug in 0.1 0.2 0.3 0.5 0.75 1.0; do
    for projector in true false; do
      if [[ "$projector" == "true" ]]; then
        proj_dims="16 32 64"
      else
        proj_dims="none"
      fi

      for proj_dim in $proj_dims; do
        logfile="logs_aug/aug_${aug}_projector_${projector}_seed_${seed}"
        args=(
          --aug "$aug"
          --lr "$lr"
          --lamb "$lamb"
          --emb_dim "$emb_dim"
          --n_views "$n_views"
          --projector "$projector"
          --steps "$steps"
          --seed "$seed"
        )

        if [[ "$projector" == "true" ]]; then
          logfile="${logfile}_projdim_${proj_dim}"
          args+=(--proj_dim "$proj_dim")
        fi

        if grep -q "^done$" "${logfile}.log" 2>/dev/null; then
          echo "skipping ${logfile}.log (already complete)"
          continue
        fi

        echo "running aug=${aug} projector=${projector} proj_dim=${proj_dim} seed=${seed}"
        uv run "${extras[@]}" main.py "${args[@]}" 2>&1 | tee "${logfile}.log"
      done
    done
  done
done
