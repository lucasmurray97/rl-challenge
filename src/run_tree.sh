#!/bin/bash
# Parameters for grid search
num_samples=(400000 600000)
max_episodes=(300 400)
envs=("false" "true")


# Generate commands for all combinations of parameters
for num_sample in "${num_samples[@]}"; do
  for max_episode in "${max_episodes[@]}"; do
          for env in "${envs[@]}"; do
          python main.py --num_samples $num_sample \
                          --max_episode $max_episode \
                          --env $env
          done
        done
      done
    done
  done
done
