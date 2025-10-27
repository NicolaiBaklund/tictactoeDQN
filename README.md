# Deep Q-Learning for Tic Tac Toe

This repository contains an implementation of a Deep Q-Network (DQN) to play the game of Tic Tac Toe using a custom neural network built with NumPyML.

### Requirements
- Python 3.7+
- NumPy
- NumPyML (my ML framework library built on NumPy)

### Installation
1. Clone the repository:
2. Clone numpyML repo from my github
3. install numpy, and include numpyML in your project

### Usage
Train the dqn
run train.py
Evaluate the trained model
run test.py



## Results
With the barebones implementation, the DQN agent learns to play Tic Tac Toe reasonably well after several thousand training episodes. Further improvements could potentially be achieved by tuning hyperparameters, increasing network complexity, or implementing more advanced exploration strategies.<br>
**Current Results:**
Episodes: 100000<br>
Wins:   55704 (55.70%)<br>
Draws:  15803 (15.80%)<br>
Losses: 28493 (28.49%)