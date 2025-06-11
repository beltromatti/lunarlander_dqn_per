# LunarLander DQN PER Project

LunarLander-v3 game played by an AI agent trained with the DQN algorithm with Prioritized Experience Replay (PER) using deep-reinforcement-learning python framework.

## Overview

This project trains a DQN agent to land a starship on the moon using the LunarLander-v3 environment. The agent learns through prioritized experience replay and an epsilon-greedy policy, with periodic model checkpointing.

## Author

This project was developed by Mattia Beltrami, a student of Computer Science for Management at the University of Bologna (UNIBO).

## Project Structure

```
lunarlander_dqn_per/
├── src/
│   ├── __init__.py           # Empty file to make src a Python module
│   ├── train.py              # Training logic
│   └── visualize.py          # Visualize an episode using a trained model
├── config/
│   └── config.yaml           # Configuration parameters
├── data/
│   ├── checkpoints/          # Model checkpoints
│   └── results/              # Training plots
├── run.py                    # Main script
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
├── LICENSE                   # License file
└── .gitignore                # Git ignore file
```

## Installation

1. Install deep-reinforcement-learning framework
   ```bash
   git clone https://github.com/beltromatti/deep-reinforcement-learning.git
   cd deep-reinforcement-learning
   pip install -e .
   ```
   
2. Clone the repository:
   ```bash
   git clone https://github.com/beltromatti/lunarlander_dqn_per.git
   cd lunarlander_dqn_per
   ```

3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
Train a new model from scratch:
```bash
python run.py --train
```

### Visualization
Visualize a single episode using a saved model:
```bash
python run.py --visualize
```

### Configuration
Edit `config/config.yaml` to adjust parameters such as the number of episodes, learning rate, or visualization frequency.

## Notes

* The default configuration in `config/config.yaml` is designed to train the agent effectively, typically achieving stable performance after 1000-1500 episodes.  
* Training duration may vary depending on hardware and random seed. For consistent results, consider averaging rewards over multiple runs.  
* To improve training stability, ensure the replay memory is sufficiently large (default: 20,000 transitions) and adjust the learning rate if convergence is too slow or unstable.
* Checkpoints are saved every 50 episodes by default.

## Use of Generative AI

This project leveraged generative artificial intelligence, specifically Grok 3 developed by xAI, to assist in generating comments and documentation. The AI was used to create clear, detailed, and educational explanations for the code and configuration files, enhancing the project's clarity for learning purposes. However, every line of code, comment, and documentation was carefully reviewed and validated by the author to ensure accuracy and correctness.

## Requirements

* Python 3.8+
* deep-reinforcement-learning 1.1+
* PyTorch 2.0.0+
* Gymnasium 0.29.1+
* Box2D-py 2.3.5+
* Pygame 2.6.1+
* NumPy
* Matplotlib
* PyYAML

See `requirements.txt` for full details.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
