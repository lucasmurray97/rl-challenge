#!/bin/bash
# Parameters for grid search
num_samples=(500000 100000)
max_episodes=(100)
gammas=(0.95 0.99)
n_estimators=(100 200)
max_depths=(10 12)
envs=(True False)


# Generate commands for all combinations of parameters
for num_sample in "${num_samples[@]}"; do
  for max_episode in "${max_episodes[@]}"; do
    for gamma in "${gammas[@]}"; do
      for n_estimator in "${n_estimators[@]}"; do
        for max_depth in "${max_depths[@]}"; do
          for env in "${envs[@]}"; do
          python main.py --num_samples $num_sample \
                          --max_episode $max_episode \
                          --gamma $gamma \
                          --n_estim $n_estimator \
                          --max_depth $max_depth \
                          --env $env
          done
        done
      done
    done
  done
done
