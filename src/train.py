# lunarlander_dqn_per/src/train.py
import gymnasium as gym
import deep_reinforcement_learning as drl


def train(config):
    # Create the LunarLander-v3 environment
    state_min = [-1.5, -1.5, -3.0, -3.0, -3.14, -4.0, 0.0, 0.0]
    state_max = [1.5, 1.5, 3.0, 3.0, 3.14, 4.0, 1.0, 1.0]
    env = drl.EnvironmentWrapper(gym.make("LunarLander-v3"), state_max_values=state_max, state_min_values=state_min)
    
    # Define a custom neural network architecture
    layers = [
        drl.InputLayer(),
        drl.HiddenLayer(256, drl.ReLU()),
        drl.HiddenLayer(128, drl.ReLU()),
        drl.HiddenLayer(64, drl.ReLU()),
        drl.OutputLayer()
    ]
    model = drl.Model(state_size=env.state_size, action_size=env.action_size, layers=layers)
    model.init_weights(mode='he_relu')
    
    # Configure training parameters
    config = drl.DQNConfig(
        episodes=1500,                                  # Number of training episodes
        batch_size=64,                                  # Size of experience replay batch
        gamma=0.99,                                     # Discount factor
        epsilon_start=1.0,                              # Initial exploration rate
        epsilon_end=0.02,                               # Final exploration rate
        epsilon_decay_mode='exponential',               # Epsilon decay mode
        epsilon_exponential_decay=0.995,                # Exploration decay rate
        memory_size=20000,                              # Replay memory capacity
        learning_rate=0.00025,                          # Adam optimizer learning rate
        target_update=5000,                             # Frequency of target network updates
        max_grad_norm=1.0,                              # Gradient clipping norm
        save_checkpoint_every=50,                       # Save model every 50 episodes
        checkpoint_path="data/checkpoints/lunarlander_dqn_per.model",    # Path to save model
        plot_path="data/results/lunarlander_dqn_per_rewards.png",    # Path to save rewards plot
        use_per = True,                                 # Use Prioritized experience replay
        per_alpha = 0.4,                                # PER Alpha
        per_beta_start= 0.6,                            # PER initial Beta value
        per_beta_end = 1,                               # PER final Beta value
        per_beta_annealing_steps = 100_000              # Number of steps to bring Beta from beta_start to beta_end
    )

    # Create and train the DQN agent
    agent = drl.DQNAgent(env, model, config)
    rewards = agent.train()
    
    # Save the trained model
    drl.save_model(model, config.checkpoint_path)
    
    # Evaluate the trained agent
    agent.run()
    
    # Print summary
    print(f"Training completed. Average reward (last 100 episodes): {sum(rewards[-100:])/100:.2f}")