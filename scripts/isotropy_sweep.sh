#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs_isotropy

lr=1e-3
lamb=0.1
emb_dim=256
n_views=8
steps=50_000

for seed in 0 1 2; do
  for projector in true false; do
    proj_dims="none"
    if [[ "$projector" == "true" ]]; then
      proj_dims="8 16 32 64"
    fi

    for proj_dim in $proj_dims; do
      logfile="logs_isotropy/seed_${seed}_projector_${projector}"
      args=(
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

      uv run --with jax[tpu] main.py "${args[@]}" 2>&1 | tee "${logfile}.log"
    done
  done
done
