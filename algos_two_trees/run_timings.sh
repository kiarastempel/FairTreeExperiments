#!/usr/bin/env bash

# Fixed dataset
dataset="Adult"

echo "=== Sweeping n_rows (100..2000), with n_cols=3, max_depth=3 ==="
for n_rows in $(seq 100 100 2000); do
  echo "Running: --timed-nrows $n_rows --timed-ncols 3 --max_depth 3"
  python method.py --data "$dataset" \
                   --performance_tree_variant "sklearn" \
                   --fair_tree_variant "fairness_gain" \
                   --timed-run \
                   --timed-nrows "$n_rows" \
                   --timed-ncols 3 \
                   --max_depth 3
done

echo "=== Sweeping n_cols (3..20), with n_rows=100, max_depth=3 ==="
for n_cols in {3..20}; do
  echo "Running: --timed-nrows 100 --timed-ncols $n_cols --max_depth 3"
  python method.py --data "$dataset" \
                   --performance_tree_variant "sklearn" \
                   --fair_tree_variant "fairness_gain" \
                   --timed-run \
                   --timed-nrows 100 \
                   --timed-ncols "$n_cols" \
                   --max_depth 3
done

echo "=== Sweeping max_depth (2..6), with n_rows=100, n_cols=3 ==="
for depth in {2..6}; do
  echo "Running: --timed-nrows 100 --timed-ncols 3 --max_depth $depth"
  python method.py --data "$dataset" \
                   --performance_tree_variant "sklearn" \
                   --fair_tree_variant "fairness_gain" \
                   --timed-run \
                   --timed-nrows 100 \
                   --timed-ncols 3 \
                   --max_depth "$depth"
done
