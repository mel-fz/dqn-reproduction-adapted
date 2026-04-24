# DQN for Atari Pong with Memory-Augmented Adaptation to MiniGrid

A reproduction and extension of the seminal Deep Q-Network (DQN) algorithm from "Human-level control through deep reinforcement learning" (Mnih et al., 2015). This project demonstrates DQN learning from raw pixels on Atari Pong, then adapts the algorithm with recurrent memory mechanisms to tackle MiniGrid's MemoryEnv, a partially observable environment requiring temporal memory.

## Project Overview

### Phase 1: Canonical Reproduction
- Train DQN from scratch on Atari Pong using the ALE/Gymnasium interface
- Implement core components: experience replay, target networks, frame stacking, and ε-greedy exploration
- Reproduce the original paper's methodology with full implementation details

### Phase 2: Memory-Augmented Adaptation
- Evaluate baseline DQN on MiniGrid MemoryEnv to demonstrate memory limitations
- Implement Recurrent DQN (DRQN) with LSTM layers for explicit temporal memory
- Compare performance between feed-forward and recurrent architectures

## Key Features
- Complete DQN implementation with convolutional architecture
- Experience replay buffer with efficient sampling
- Atari preprocessing pipeline (grayscale, resize, frame stacking)
- DRQN variant with sequence-based training for partial observability
- Comprehensive logging and visualization of training metrics
- Multi-seed evaluation (3+ seeds) for statistical reliability
- Ablation studies isolating the impact of memory mechanisms

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dqn-pong-minigrid.git
cd dqn-pong-minigrid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```
# Training DQN on Atari Pong
python train_dqn_pong.py --seed 42 --episodes 1000

# Training DQN on MiniGrid MemoryEnv
python train_dqn_minigrid.py --seed 42 --episodes 5000

# Training DRQN on MiniGrid MemoryEnv
python train_drqn_minigrid.py --seed 42 --episodes 5000 --lstm-size 256
```
## Evaluation
```
python evaluate.py --model checkpoints/dqn_pong_final.pth --episodes 100
```
## Hyperparameters

### MiniGrid DQN Baseline (feed-forward)

| Hyperparameter | Value |
|---|---|
| Environment | MiniGrid-MemoryS7-v0 |
| Input shape | (3, 7, 7) — C, H, W |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Discount (γ) | 0.99 |
| Batch size | 32 |
| Replay buffer | 10,000 transitions |
| ε start → end | 1.0 → 0.1 |
| ε decay | 0.995 per episode |
| Target update freq | every 10 episodes |
| Episodes | 500 |
| Seeds | 42, 123, 456 |

### MiniGrid DRQN (recurrent)

| Hyperparameter | Value |
|---|---|
| Environment | MiniGrid-MemoryS7-v0 |
| Input shape | (3, 7, 7) — C, H, W |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Discount (γ) | 0.99 |
| Batch size | 32 |
| Replay buffer | 500 episodes |
| Sequence length | 8 |
| LSTM hidden size | 128 (ablation: 64, 128, 256) |
| ε start → end | 1.0 → 0.1 |
| ε decay | 0.995 per episode |
| Target update freq | every 10 episodes |
| Episodes | 500 |
| Seeds | 42, 123, 456 |

## Results
Results include learning curves, success rates, and qualitative demonstrations comparing vanilla DQN performance on Pong versus memory-augmented DQN on memory-dependent tasks.

[Results will be added after training completion]

