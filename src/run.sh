#!/bin/bash
# Parameters for grid search
lrs=(1e-3 1e-4)
gammas=(0.95 0.99)
hidden_dims=(64 256 512)
update_target_freqs=(400)

# Fixed parameters
buffer_size=100000
epsilon_min=0.02
epsilon_max=1.0
epsilon_delay_decay=110
batch_size=750
epsilon_decay_period=20000
max_episode=500
gradient_steps=3
n_layers=6
pre_fill_buffer=False

# Generate commands for all combinations of parameters
for lr in "${lrs[@]}"; do
  for gamma in "${gammas[@]}"; do
    for hidden_dim in "${hidden_dims[@]}"; do
      for update_target_freq in "${update_target_freqs[@]}"; do
        python main.py --buffer_size $buffer_size \
                          --epsilon_min $epsilon_min \
                          --epsilon_max $epsilon_max \
                          --epsilon_delay_decay $epsilon_delay_decay \
                          --batch_size $batch_size \
                          --epsilon_decay_period $epsilon_decay_period \
                          --max_episode $max_episode \
                          --gradient_steps $gradient_steps \
                          --n_layers $n_layers \
                          --hidden_dim $hidden_dim \
                          --lr $lr \
                          --gamma $gamma \
                          --update_target_freq $update_target_freq
      done
    done
  done
done
