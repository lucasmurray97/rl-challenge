import random
import os
from pathlib import Path
import numpy as np
import argparse
import torch

from evaluate import evaluate_HIV, evaluate_HIV_population
from train import VanillaAgent, ForestAgent  # Replace DummyAgent with your agent implementation
from env_hiv_ import FastHIVPatient as HIVPatient
from gymnasium.wrappers import TimeLimit

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    parser = argparse.ArgumentParser(description="DQN Configuration Arguments")

    # Add arguments for DQN configuration
    # parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for the optimizer')
    # parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor for future rewards')
    # parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
    # parser.add_argument('--epsilon_min', type=float, default=0.995, help='Minimum epsilon for exploration')
    # parser.add_argument('--epsilon_max', type=float, default=1.0, help='Maximum epsilon for exploration')
    # parser.add_argument('--epsilon_decay_period', type=int, default=1000, help='Period over which epsilon decays')
    # parser.add_argument('--epsilon_delay_decay', type=int, default=20, help='Delay before epsilon starts decaying')
    # parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training')
    # parser.add_argument('--gradient_steps', type=int, default=1, help='Gradient steps for training')
    # parser.add_argument('--pre_fill_buffer', type=bool, default=False, help='Pre-fill buffer with random actions')
    
    # New arguments for network architecture
    # parser.add_argument('--n_layers', type=int, default=6, help='Number of hidden layers in the neural network')
    # parser.add_argument('--hidden_dim', type=int, default=128, help='Number of neurons in each hidden layer')
    parser.add_argument('--max_episode', type=int, default=10, help='Maximum number of episodes for training')
    # parser.add_argument('--update_target_freq', type=int, default=100,
                        # help='Frequency (in steps) of hard target network updates (only used if strategy is "hard")')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to evaluate the agent')
    # parser.add_argument('--n_estim', type=int, default=100, help='Number of trees in the forest')
    # parser.add_argument('--max_depth', type=int, default=10, help='Maximum depth of the trees in the forest')
    parser.add_argument("--env", type=lambda x: x.lower() == 'true', help="Environment flag")
    args = parser.parse_args()

    # Create a configuration dictionary from parsed arguments
    config = {
        # 'learning_rate': args.lr,
        # 'gamma': args.gamma,
        # 'buffer_size': args.buffer_size,
        # 'epsilon_min': args.epsilon_min,
        # 'epsilon_max': args.epsilon_max,
        # 'epsilon_decay_period': args.epsilon_decay_period,
        # 'epsilon_delay_decay': args.epsilon_delay_decay,
        # 'batch_size': args.batch_size,
        # 'n_layers': args.n_layers,
        # 'hidden_dim': args.hidden_dim,
        'max_episode': args.max_episode,
        # 'update_target_freq': args.update_target_freq,
        # 'gradient_steps': args.gradient_steps,
        # 'pre_fill_buffer': args.pre_fill_buffer,
        'num_samples': args.num_samples,
        # "n_estim": args.n_estim,
        # "max_depth": args.max_depth,
        'env': args.env
    }
    config_str = ' '.join(f'_{value}' for key, value in config.items())
    file = Path("score" + config_str + ".txt")
    env = TimeLimit(
        env=HIVPatient(domain_randomization=config["env"]), max_episode_steps=200
    ) 
    agent = ForestAgent(config, env)
    agent.load()
    # Evaluate agent and write score.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=5)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=20)
    with open(file="scores/score" + config_str + ".txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")
