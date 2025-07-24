#!/usr/bin/env bash

datasets=("Compas" "Adult" "German" "Dutch")
fair_variants=("fairness_gain")
hp_min_samples=(25 10 1)
hp_max_depth=(4 6 8 13)

echo "=== Run all datasets ==="
for dataset in "${datasets[@]}"; do
  echo "=== Run all variants for $dataset==="
  for fair_variant in "${fair_variants[@]}"; do
    echo "==> Variant: $fair_variant"
    for max_depth in "${hp_max_depth[@]}"; do
      for min_samples in "${hp_min_samples[@]}"; do
        for seed in {1..15}; do
          echo "Running: --seed $seed --min_samples $min_samples --max_depth $max_depth --tree_variant $fair_variant --data $dataset"
          python cv_two_trees.py --data "$dataset" \
                                 --seed $seed \
                                 --num_cv 3 \
                                 --num_gammas 50 \
                                 --max_depth $max_depth \
                                 --min_samples $min_samples \
                                 --performance_tree_variant "sklearn" \
                                 --fair_tree_variant "$fair_variant" &
        done
        wait
      done
    done
  done
done

