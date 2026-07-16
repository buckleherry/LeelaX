# LeelaX

A lightweight AlphaZero-style reinforcement learning system for chess.  

The name is a nod to Leela Chess Zero (Lc0) — this project is unaffiliated and built independently from scratch.

Built from scratch with a focus on:
- **RL systems engineering**
- **self-play training**
- **evaluation under limited compute**

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
  - Play first games which lead to checkmate in under 50 moves
  - Avoid trivial repetitions more reliably than early runs
- Still **tactically weak**:
  - Material advantage not yet fully understood (thus still hanging a lot of pieces)
  - Some positions still end in senseless repetition
  - King walks (especially with black it seems)
  - Long-term planning is limited
- Main limiting factors:
  - CPU-only training
  - Relatively small networks
  - Limited self-play volume

This behavior is expected for the current training regime and compute budget.

I am currently training on my MacBook Air M1 from 2020.  
Generating a couple of thousand games takes somewhere between 12 and 24 hours, depending on the chosen model size.  
To reach a significantly stronger state, RL self-play models in this category usually need a vastly larger number of games played.  
Currently, this is not feasible and also not the main scope of the project.  

---

## Setup

LeelaX is installed as an editable package, so the `leelax` module is importable from anywhere in the repository (scripts, tests, notebooks).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
pytest   # verify the installation (22 tests)
```

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

## Example: Arena Evaluation
```bash
python scripts/arena.py \
  --a checkpoints/exp_K_128x6_baseline/model_cycle_020.pt \
  --b checkpoints/exp_K_128x6_baseline/model_cycle_030.pt \
  --games 80 \
  --simulations 64 \
  --max-moves 220 \
  --neutral \
  --save-pgns \
  --out-dir arena/exp_K_c20_vs_c30
```

---

## Project Goals

This project focuses on:

- Reinforcement learning system design
- Debugging and stabilizing self-play pipelines
- Understanding exploration, collapse, and evaluation pitfalls
- Working under realistic compute constraints


## Future Work

Possible extensions:

- Reward shaping for aggressive play styles
- Reward shaping for a quicker learning process (rewards for material gains, checks, king exposure...)
- Knowledge distillation from stronger engines
- GPU training and larger networks
- Improved endgame handling

--- 

## Disclaimer

LeelaX is intended for educational and research purposes.
It is not a competitive chess engine.


## Author

Built by Sven Kohl  
_AI Engineer_

More projects & background: https://skohl.io

