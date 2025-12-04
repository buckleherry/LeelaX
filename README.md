# 🧠 LeelaX – AlphaZero-inspired Chess Reinforcement Learning Engine

## Overview

LeelaX is a modular, research-oriented implementation of an **AlphaZero-style Reinforcement Learning system** for chess.  
It is inspired by Leela Chess Zero, focusing on clarity, modularity, and experimental flexibility.

The goal is **not** to maximize playing strength, but to build a *transparent, analyzable RL pipeline* suitable for modern AI research and portfolio demonstration.

---

## 🧩 Core Modules

- **Environment (`leelax.env`)**  
  Based on `python-chess`, providing canonical (side-to-move) board encoding, legality masks, and reward shaping hooks.

- **Network (`leelax.net`)**  
  Compact ResNet-like architecture with shared trunk and two heads (policy + value).  
  Input: 24x8x8 tensor, Output: policy logits (4864) and value scalar.

- **MCTS (`leelax.mcts`)**  
  Implements the PUCT variant with neural priors, Dirichlet noise, and temperature control.

- **Self-Play (`leelax.selfplay`)**  
  Generates games through MCTS-guided play. Stores (state, policy, value) tuples into a replay buffer.

- **Replay Buffer & Dataset (`leelax.selfplay.replay_buffer`, `leelax.train.replay_dataset`)**  
  In-memory ring buffer + PyTorch Dataset wrapper for efficient sampling and training integration.

- **Training (`leelax.train`)**  
  Implements AlphaZero-style optimization:
  - Loss = CE(π, logits) + MSE(z, value)
  - Adam optimizer
  - DataLoader-based batching
  - extensible for schedulers, logging, and mixed precision.

## Model Sizes

LeelaX currently implements 3 model variants:

- **small** → 64×4 residual trunk (fastest CPU mode)
- **base** → 96×6 trunk (balanced)
- **128x6** → 128×6 trunk (strongest, recommended)

The architecture follows AlphaZero/Leela-style design:
trunk → policy head (76×8×8 logits) → value head.

---

## 🔁 Training & Evaluation

### Minimal Loop

1. Generate self-play games via MCTS:
   ```python
   samples = worker.play_game()
   buffer.add_many(samples)
   ```
2. Train for N steps:
    ```python
    from leelax.train.loop import train_for_n_steps
    train_for_n_steps(model, buffer, n_steps=1000)
    ```
3. (Optional) Evaluate vs older checkpoints or Stockfish (arena mode planned).

## Research & Logging

- The project is designed for experimental analysis:
- Reward shaping (aggressiveness, initiative, control)
- Policy evolution tracking
- Replay visualization
- Evaluation across training epochs

## 🧰 Roadmap

- [x] Environment & canonical encoding
- [x] Network definition
- [x] MCTS (PUCT)
- [x] Self-play loop
- [x] Replay buffer & DataLoader integration
- [x] Training scheduler + logging
- [ ] Evaluation & Arena system
- [ ] Reward shaping extensions
- [ ] Long-run experiment analysis

## ⚙️ Setup

```python
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
```

## Licence

MIT License — open for academic and personal research use.
