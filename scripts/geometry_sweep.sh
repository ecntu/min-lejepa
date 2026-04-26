#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs_geometry

lr=1e-3
lamb=0.01
n_views=8
steps=50_000

# run with TPU=1 to install jax[tpu]
extras=()
[[ "${TPU:-0}" == "1" ]] && extras=(--with "jax[tpu]")

# 117 runs total
for seed in 0 1 2; do
  for emb_dim in 4 8 16 32 64 128 256; do
    for projector in false true; do
      pd_list=("none")
      if [[ "$projector" == "true" ]]; then
        pd_list=(2 4 8 16 32 64)
      fi

      for proj_dim in "${pd_list[@]}"; do
        if [[ "$projector" == "true" && "$proj_dim" -gt "$emb_dim" ]]; then
          continue
        fi

        logfile="logs_geometry/seed_${seed}_emb_${emb_dim}_projector_${projector}"
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

        uv run "${extras[@]}" main.py "${args[@]}" 2>&1 | tee "${logfile}.log"
      done
    done
  done
done
