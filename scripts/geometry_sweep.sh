#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs_geometry

lr=1e-3
n_views=8
steps=50_000

extras=()
[[ "${TPU:-0}" == "1" ]] && extras=(--with "jax[tpu]")

lambs=(1e-3 3e-3 1e-2 3e-2 1e-1 3e-1)

# emb=16: (5 pd + 1 no-proj) x 6 lamb x 3 seeds = 108
# emb=64: (5 pd + 1 no-proj) x 6 lamb x 3 seeds = 108
# emb=256: (3 pd + 1 no-proj) x 6 lamb x 3 seeds = 72
# total: 288 runs

for seed in 0 1 2; do
  for emb_dim in 16 64 256; do
    if   [[ "$emb_dim" == 16  ]]; then pd_list=(4 8 16 32 64)
    elif [[ "$emb_dim" == 64  ]]; then pd_list=(16 32 64 128 256)
    else                               pd_list=(64 128 256)
    fi

    for projector in false true; do
      [[ "$projector" == true ]] && dims=("${pd_list[@]}") || dims=(none)

      for proj_dim in "${dims[@]}"; do
        for lamb in "${lambs[@]}"; do
          logfile="logs_geometry/seed_${seed}_emb_${emb_dim}_projector_${projector}"
          args=(--lr "$lr" --lamb "$lamb" --emb_dim "$emb_dim"
                --n_views "$n_views" --projector "$projector"
                --steps "$steps" --seed "$seed")

          if [[ "$projector" == true ]]; then
            logfile="${logfile}_projdim_${proj_dim}"
            args+=(--proj_dim "$proj_dim")
          fi

          logfile="${logfile}_lamb_${lamb}"

          if grep -q "^done$" "${logfile}.log" 2>/dev/null; then
            echo "skipping ${logfile}.log (already complete)"
            continue
          fi

          uv run "${extras[@]}" main.py "${args[@]}" 2>&1 | tee "${logfile}.log"
        done
      done
    done
  done
done
