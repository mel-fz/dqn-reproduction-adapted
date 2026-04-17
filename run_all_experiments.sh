#!/bin/bash
# Quick script to run all multi-seed experiments

echo "Starting multi-seed DQN experiments..."
echo ""

# DQN on Pong (canonical reproduction)
echo "========================================"
echo "Running DQN on Atari Pong (3 seeds)..."
echo "========================================"
python run_experiments.py --mode pong --seeds 42 123 456 --max-steps 5000000

echo ""
echo "========================================"
echo "Running DQN on MiniGrid (3 seeds)..."
echo "========================================"
python run_experiments.py --mode minigrid-dqn --seeds 42 123 456 --episodes 200

echo ""
echo "========================================"
echo "Running DRQN on MiniGrid (3 seeds)..."
echo "========================================"
python run_experiments.py --mode minigrid-drqn --seeds 42 123 456 --episodes 200

echo ""
echo "All experiments complete!"
echo "Results saved to: results/logs/"
