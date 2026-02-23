# Deep Q-Learning for LunarLander-v3

This project implements a **Deep Q-Network (DQN)** to solve the [LunarLander-v3](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment from **Gymnasium**. The goal is to train an AI agent to land the lunar module safely between the landing flags while minimizing horizontal drift.

---

## Table of Contents

- [Overview](#overview)  
- [Installation](#installation)  
- [Project Structure](#project-structure)  
- [Neural Network Architecture](#neural-network-architecture)  
- [Training](#training)  
- [Reward Shaping](#reward-shaping)  
- [Hyperparameters](#hyperparameters)  
- [Results](#results)  
- [Video Visualization](#video-visualization)  
  

---

## Overview

The agent uses a Deep Q-Learning algorithm with experience replay and a target network. Key features include:

- Experience replay buffer to stabilize learning.
- Target network with soft updates.
- Reward shaping to improve landing precision.
- Epsilon-greedy exploration strategy.

The agent is trained until it consistently scores above 200 points, which is considered “solving” the environment.

---

## Installation

This project requires Python 3.12+ and the following packages:

```bash
pip install gymnasium
pip install "gymnasium[box2d]"
pip install torch torchvision torchaudio
pip install imageio
```

# Deep Q-Learning for LunarLander-v3

---

## Project Structure

- **Network** — Defines the DQN neural network.  
- **ReplayMemory** — Implements the experience replay buffer.  
- **Agent** — Handles DQN agent logic, including action selection, learning, and soft updates.  
- **train.py** — Script for training the agent.  
- **video.py** — Script to visualize the trained agent.  
- **checkpoint.pth** — Saved model weights after solving the environment.  

---

## Neural Network Architecture

The DQN consists of:

- **Input Layer**: 8 nodes (state vector from LunarLander environment)  
- **Hidden Layers**: Two fully connected layers with 64 neurons each, ReLU activation  
- **Output Layer**: 4 nodes (Q-values for each action)  

```python
x = F.relu(self.fc1(state))
x = F.relu(self.fc2(x))
x = self.fc3(x)
```
---

## Training

1. Initialize the LunarLander environment.  
2. Set hyperparameters (learning rate, batch size, discount factor, etc.).  
3. Train the agent over multiple episodes using epsilon-greedy policy.  
4. Use experience replay and soft updates to improve stability.  
5. Training continues until the **average score over 100 episodes ≥ 200**.  

---

## Reward Shaping

To improve precision when landing:

- **Penalize horizontal distance from center**:  

```python
reward -= 0.2 * abs(x_pos)
```

To improve precision when landing:

- **Penalize horizontal velocity**:

```python
reward -= 0.1 * abs(x_vel)
```


## Hyperparameters

| Parameter                | Value         |
|--------------------------|---------------|
| Learning rate            | 3e-4 – 5e-4   |
| Batch size               | 64 – 100      |
| Discount factor (gamma)  | 0.99          |
| Replay buffer size       | 100,000       |
| Soft update factor (tau) | 1e-2          |
| Epsilon start            | 1.0           |
| Epsilon end              | 0.01          |
| Epsilon decay            | 0.995         |


## Results

- Environment is solved after ~340 episodes.  
- Average score: ~200+ points.  
- Agent lands consistently near the center between landing flags.  

### Sample Training Progress

```text
Episode 100  Average Score: -155
Episode 200  Average Score: -68
Episode 300  Average Score: -0.2
Episode 400  Average Score: 137
Episode 440  Average Score: 200.6
```

## Video Visualization

Trained agent performance can be visualized using `imageio`:

```python
show_video_of_model(agent, 'LunarLander-v3')
show_video()
```

This produces an mp4 video showing the agent landing successfully.
