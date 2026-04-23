#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

for lr in 1e-4 3e-4 1e-3; do
  for lamb in 0.01 0.05 0.1; do
    for emb_dim in 64 128 256; do
      for n_views in 2 4 8; do
        for projector in true false; do
          proj_dims="none"
          if [[ "$projector" == "true" ]]; then
            proj_dims="16 32 64"
          fi

          for proj_dim in $proj_dims; do
            logfile="logs/lr_${lr}_lamb_${lamb}_emb_${emb_dim}_views_${n_views}_projector_${projector}"
            args=(
              --lr "$lr"
              --lamb "$lamb"
              --emb_dim "$emb_dim"
              --n_views "$n_views"
              --projector "$projector"
              --steps 50_000
            )

            if [[ "$projector" == "true" ]]; then
              logfile="${logfile}_projdim_${proj_dim}"
              args+=(--proj_dim "$proj_dim")
            fi

            uv run main.py "${args[@]}" 2>&1 | tee "${logfile}.log"
          done
        done
      done
    done
  done
done
