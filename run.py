"""
This script demonstrates the implementation of a Deep Q-Network (DQN) with Prioritized Experience Replay (PER) to train an agent to play the LunarLander-v3 environment from Gymnasium. The goal is to land a lunar module safely on a landing pad.

Key components and functionality:
1. **Environment Setup**: Creates a LunarLander-v3 environment wrapped with custom state normalization using predefined state boundaries.
2. **Neural Network Architecture**: Defines a custom feedforward neural network with three hidden layers (256, 128, 64 neurons) using ReLU activations, initialized with He initialization.
3. **DQN Configuration**: Configures the DQN algorithm with parameters such as:
   - 1500 training episodes
   - Exponential epsilon decay for exploration (from 1.0 to 0.02)
   - Prioritized Experience Replay (PER) with alpha and beta parameters
   - Replay memory of 20,000 experiences
   - Target network updates every 5,000 steps
   - Gradient clipping and Adam optimizer with a learning rate of 0.00025
4. **Training**: Trains the DQN agent using the configured parameters, saving model checkpoints every 50 episodes and plotting rewards.
5. **Visualizing**: Runs the trained agent to demonstrate its performance.

This code serves as an educational example of implementing a DQN with PER for reinforcement learning tasks, showcasing environment interaction, neural network design, and training configuration.
"""
import argparse
import deep_reinforcement_learning as drl
from src.train import train
from src.visualize import visualize

def main():
    """Main entry point for the LunarLander DRL project."""
    parser = argparse.ArgumentParser(description="LunarLander Deep Reinforcement Learning Project")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--visualize", action="store_true", help="Visualize a single episode")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = drl.DQNConfig().load_yaml_config(args.config)

    if args.train:
        train(config)
    elif args.visualize:
        visualize(model_path="data/checkpoints/lunarlander_dqn_per.model")

if __name__ == "__main__":
    main()