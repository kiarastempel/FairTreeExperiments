#!/usr/bin/env bash

# Fixed dataset
dataset="Adult"

# Variants to loop over
variants=("threshold_gain_s" "weighted_combi" "backtracking")

echo "=== Sweeping n_rows (100..2000), with n_cols=3, max_depth=3 ==="
for variant in "${variants[@]}"; do
  echo "==> Variant: $variant"
  for n_rows in $(seq 100 100 2000); do
    echo "Running: --timed-nrows $n_rows --timed-ncols 3 --max_depth 3 --tree_variant $variant"
    python method.py --data "$dataset" \
                     --timed-run \
                     --timed-nrows "$n_rows" \
                     --timed-ncols 3 \
                     --max_depth 3 \
                     --tree_variant "$variant"
  done
done

echo "=== Sweeping n_cols (3..20), with n_rows=100, max_depth=2 ==="
for variant in "${variants[@]}"; do
  echo "==> Variant: $variant"
  for n_cols in {3..20}; do
    echo "Running: --timed-nrows 100 --timed-ncols $n_cols --max_depth 3 --tree_variant $variant"
    python method.py --data "$dataset" \
                     --timed-run \
                     --timed-nrows 100 \
                     --timed-ncols "$n_cols" \
                     --max_depth 3 \
                     --tree_variant "$variant"
  done
done

echo "=== Sweeping max_depth (2..6), with n_rows=100, n_cols=3 ==="
for variant in "${variants[@]}"; do
  echo "==> Variant: $variant"
  for depth in {2..6}; do
    echo "Running: --timed-nrows 100 --timed-ncols 3 --max_depth $depth --tree_variant $variant"
    python method.py --data "$dataset" \
                     --timed-run \
                     --timed-nrows 100 \
                     --timed-ncols 3 \
                     --max_depth "$depth" \
                     --tree_variant "$variant"
  done
done
