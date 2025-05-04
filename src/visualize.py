# lunarlander_dqn_per/src/visualize.py
import gymnasium as gym
import deep_reinforcement_learning as drl


def visualize(model_path):
    # Create the LunarLander-v3 environment
    state_min = [-1.5, -1.5, -3.0, -3.0, -3.14, -4.0, 0.0, 0.0]
    state_max = [1.5, 1.5, 3.0, 3.0, 3.14, 4.0, 1.0, 1.0]
    env = drl.EnvironmentWrapper(gym.make("LunarLander-v3", render_mode="human"), state_max_values=state_max, state_min_values=state_min)
    
    # Define a custom neural network architecture
    model = drl.load_model(model_path)

    # Create and train the DQN agent
    agent = drl.DQNAgent(env, model)
    
    # Evaluate the trained agent
    agent.run()