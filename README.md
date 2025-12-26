# LeelaX

A lightweight AlphaZero-style reinforcement learning system for chess,
built from scratch with a focus on **RL systems engineering**, **self-play training**,
and **evaluation under limited compute**.

> This project is a research / engineering prototype — not a production chess engine.

---

## Overview

LeelaX implements an end-to-end reinforcement learning loop inspired by AlphaZero:

- Neural network with **policy + value heads**
- **Monte Carlo Tree Search (MCTS)** for move selection
- **Self-play** to generate training data
- Supervised-style training from self-play targets
- **Checkpoint-based evaluation** via arena matches

The system is designed to be **fully configurable** and to run **CPU-only**,
making it suitable for experimentation without large compute budgets.

---

## Key Features

- 🧠 **Policy–Value Network**
  - Configurable model sizes (e.g. `128x6`)
  - Residual CNN architecture
- 🌲 **MCTS**
  - Adjustable simulations per move
  - Temperature-controlled exploration
- ♻️ **Self-Play Training Loop**
  - Replay buffer
  - Multi-cycle training
  - Checkpointing & resume support
- ⚔️ **Arena Evaluation**
  - Checkpoint vs checkpoint matches
  - Neutral evaluation mode (no shaping)
  - Elo-style scoring
- 📊 **Diagnostics**
  - TensorBoard logging
  - PGN & FEN export for qualitative analysis

---

## Current Status

- The model plays **legal, structured chess** and can:
  - Develop pieces
  - Find checks and basic mates
  - Avoid trivial repetitions more reliably than early runs
- Still **tactically weak**:
  - Material is often dropped
  - Long-term planning is limited
- Main limiting factors:
  - CPU-only training
  - Relatively small networks
  - Limited self-play volume

This behavior is expected for the current training regime and compute budget.

---

## Example: Training Run

```bash
python scripts/run_cycle.py \
  --cycles 30 \
  --games-per-cycle 180 \
  --train-steps 2000 \
  --simulations 40 \
  --max-moves 320 \
  --device cpu \
  --model-size 128x6 \
  --log-dir runs/exp_K_128x6_baseline \
  --checkpoint-dir checkpoints/exp_K_128x6_baseline
```
