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
- Ablation study isolating the effect of LSTM hidden size (64, 128, 256)

## Key Features
- Complete DQN implementation with convolutional architecture
- Experience replay buffer with efficient sampling
- Atari preprocessing pipeline (grayscale, resize, frame stacking)
- DRQN variant with episode-based replay and sequence training for partial observability
- Comprehensive JSON-based logging and visualization of training metrics
- Multi-seed evaluation (3 seeds) for statistical reliability
- Ablation study isolating the impact of LSTM hidden size

## Installation

```bash
# Clone the repository
git clone https://github.com/mel-fz/dqn-reproduction-adapted.git
cd dqn-reproduction-adapted

# Create virtual environment
python -m venv dqn
source dqn/bin/activate  # On Windows: dqn\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Phase 1: Train DQN on Atari Pong
```bash
# Single seed
python train_dqn_pong.py

# Multi-seed via experiment runner
python run_experiments.py --mode pong --seeds 42 123 7
```

### Phase 2: Train on MiniGrid MemoryEnv

```bash
# DQN baseline (feed-forward, expected to fail)
python run_experiments.py --mode minigrid-dqn --seeds 42 123 456 --episodes 500

# DRQN (recurrent, memory-augmented)
python run_experiments.py --mode minigrid-drqn --seeds 42 123 456 --episodes 500

# DRQN with specific LSTM size and seed
python train_drqn_minigrid.py --seed 42 --episodes 500 --lstm-size 128
```

### LSTM Ablation
```bash
python train_drqn_minigrid.py --seed 42 --episodes 500 --lstm-size 64
python train_drqn_minigrid.py --seed 42 --episodes 500 --lstm-size 128
python train_drqn_minigrid.py --seed 42 --episodes 500 --lstm-size 256
```

## Reproducing Plots

All figures are generated from logs saved to `results/logs/`. Run training first, then:

### DQN vs DRQN comparison on MiniGrid
Requires: `dqn_minigrid_seed{42,123,456}.json` and `drqn_minigrid_lstm128_seed{42,123,456}.json`
```bash
python compare_minigrid.py --seeds 42 123 456
# Output: results/minigrid_comparison.png
```

### LSTM hidden size ablation
Requires: `drqn_minigrid_lstm{64,128,256}_seed42.json`
```bash
python ablation_lstm.py --seeds 42
# Output: results/ablation_lstm.png
```

### Pong multi-seed training curves
Generated automatically after running `train_dqn_pong.py`:
```bash
# Output: results/pong_multiseed_results.png
```

## Evaluation
```bash
python evaluate.py --model checkpoints/dqn_pong_seed42_final.pth --episodes 100
```

## Hyperparameters

### DQN on Atari Pong

| Hyperparameter | Value | Note |
|---|---|---|
| Environment | ALE/Pong-v5 | |
| Frame stack | 4 | agent history |
| Preprocessing | Grayscale, resize 84×84 | |
| Training budget | 500,000 frames or 1,500 episodes | whichever first |
| Learning rate | 0.00025 | |
| Discount (γ) | 0.99 | |
| Batch size | 32 | |
| Replay buffer | 100,000 transitions | |
| Learning starts | 10,000 steps | |
| Target update freq | every 10,000 steps | |
| ε start → end | 1.0 → 0.1 linear over 1,000,000 frames | |
| Optimizer | RMSprop (α=0.95, ε=0.01) | matches Mnih et al. 2015 |
| Seeds | 42, 123, 7 | |

### MiniGrid DQN Baseline (feed-forward)

| Hyperparameter | Value |
|---|---|
| Environment | MiniGrid-MemoryS7-v0 |
| Input shape | (3, 7, 7) — C, H, W |
| Architecture | Feed-forward CNN + FC (no memory) |
| Optimizer | Adam |
| Loss | MSE |
| Learning rate | 0.001 |
| Discount (γ) | 0.99 |
| Batch size | 32 |
| Replay buffer | 10,000 transitions |
| ε start → end | 1.0 → 0.1 |
| ε decay | ×0.995 per episode |
| Target update freq | every 10 episodes |
| Episodes | 500 |
| Seeds | 42, 123, 456 |

### MiniGrid DRQN (recurrent)

| Hyperparameter | Value |
|---|---|
| Environment | MiniGrid-MemoryS7-v0 |
| Input shape | (3, 7, 7) — C, H, W |
| Architecture | CNN + LSTM + FC |
| Optimizer | Adam |
| Loss | Smooth L1 (Huber) |
| Learning rate | 0.001 |
| Discount (γ) | 0.99 |
| Batch size | 32 |
| Replay buffer | 500 episodes (episode buffer, not transitions) |
| Sequence length | 8 |
| LSTM hidden size | 128 (ablation: 64, 128, 256) |
| ε start → end | 1.0 → 0.1 |
| ε decay | ×0.995 per episode |
| Target update freq | every 10 episodes |
| Episodes | 500 |
| Seeds | 42, 123, 456 |

## Compute Budget

| Experiment | Seeds | Est. compute |
|---|---|---|
| DQN on Pong | 3 × 500K frames | ~8–12 hrs CPU |
| DQN MiniGrid baseline | 3 × 500 episodes | ~30–45 min CPU |
| DRQN MiniGrid (lstm-128) | 3 × 500 episodes | ~1.5–2 hrs CPU |
| DRQN ablation (lstm-64, 256) | 1 seed each × 500 episodes | ~45–60 min CPU |
| **Total** | | **~11–16 hrs CPU** |

All experiments were conducted on CPU hardware (no GPU acceleration). The MiniGrid experiments — including the DQN baseline (3 seeds × 500 episodes), DRQN with LSTM hidden size 128 (3 seeds × 500 episodes), and the single-seed LSTM size ablation (sizes 64, 128, 256) — required approximately 2–3 hours of total compute. The Atari Pong reproduction was the dominant cost, requiring approximately 500,000 training frames per seed across 3 seeds, estimated at 8–12 hours of CPU time. Total compute across all experiments is estimated at 11–16 CPU-hours. 

## Results

Figures are saved to `results/` after running the plot scripts above.

| Figure | File | Description |
|---|---|---|
| Pong training curves | `results/pong_multiseed_results.png` | Episode reward + win rate across 3 seeds |
| DQN vs DRQN on MiniGrid | `results/minigrid_comparison.png` | Reward + success rate, mean ± std across 3 seeds |
| LSTM ablation | `results/ablation_lstm.png` | Effect of LSTM hidden size on MiniGrid performance |
