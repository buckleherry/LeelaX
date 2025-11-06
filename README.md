# 🧠 LeelaX — A Minimal AlphaZero-Style Reinforcement Learning System for Chess

**LeelaX** is a modular, research-oriented reimplementation of the *Leela Chess Zero / AlphaZero* learning loop, designed for clarity, reproducibility, and experimentation with modern Reinforcement Learning (RL) ideas.

This project is built as a **scientific prototype** — not to achieve peak Elo, but to **demonstrate and explore the principles** behind self-play learning, Monte Carlo Tree Search (MCTS), and neural policy/value training in complex domains like chess.

This repository currently demonstrates a full inference path:
FEN → encoded tensor → neural policy/value → MCTS (PUCT) → improved policy → move.
The next step is to wrap this into a self-play loop and start producing training data.

---

## 🎯 Project Goals

- **Research transparency** — implement a minimal but complete AlphaZero pipeline in Python, with clear code and full test coverage.
- **Modularity** — independent components for state encoding, move policy indexing, neural architecture, and MCTS.
- **Reward shaping** — experiment with *aggressive* play incentives (initiative, tempo, piece activity) beyond simple win/loss.
- **Scientific clarity** — serve as a portfolio project and teaching reference for modern deep RL.
- **Reproducibility** — lightweight setup, local execution, Docker-ready for later reproducible training.

---

## ⚙️ Core Architecture

| Module | Description |
|:-------|:-------------|
| `leelax.env.encode` | Converts `python-chess` board states into `(24, 8, 8)` tensor planes. Includes side-to-move, castling rights, repetition, and in-check indicators. |
| `leelax.env.policy_index` | Defines the **8×8×73 AlphaZero policy mapping**, legal move mask generation, and move↔index conversions. |
| `leelax.net.model` | (Planned) Small ResNet-style network with dual heads (policy + value). |
| `leelax.mcts` | (Planned) PUCT-based search with Dirichlet noise, temperature schedule, and policy priors. |
| `leelax.selfplay` | (Planned) Self-play training loop producing `(state, policy, value)` triplets. |
| `leelax.train` | (Planned) Supervised RL training pipeline with replay buffer and optimizer schedule. |
| `leelax.eval` | (Planned) Arena evaluator between checkpoints or versus Stockfish (low depth). |

---

## 🧩 Current Status

| Component | Progress | Notes |
|:-----------|:----------|:------|
| **LXL-001 – State Encoder** | ✅ Complete | 24-channel board encoding. |
| **LXL-002 – Policy Indexing (8×8×73)** | ✅ Complete | Action space defined and tested. |
| **LXL-003 – CLI / visual debug tools** | ⏳ Planned | Human-readable dumps of encoded states and masks. |
| **LXL-010 → LXL-013 – Network + Training** | 🚧 Upcoming | Policy-Value network, loss functions, replay buffer, optimizer. |
| **LXL-020 – Self-Play + MCTS** | 🚧 Upcoming | Full AlphaZero-style RL loop. |

## Current Status

- ✅ Environment
  - 24-plane board encoding (`leelax.env.encode`)
  - AlphaZero 8×8×73 policy space + legal move mask (`leelax.env.policy_index`)
  - CLI for inspecting encodings (`python -m leelax.env.inspect --fen ...`)
- ✅ Neural Network
  - Small ResNet backbone (4×64) with policy (4672 logits) + value head (`leelax.net.model`)
  - Masked softmax + action selection utilities (`leelax.net.inference`)
- ✅ Search
  - PUCT-based MCTS that uses the network for priors and value (`leelax.mcts.puct`)
  - Returns visit-based policy (training target) and a chosen move
- ⏳ Next
  - Self-play worker to generate (state, policy, value) triplets
  - Replay buffer
  - Training loop (CE + MSE) and checkpointing
  - Evaluation / arena

---

## 📚 Scientific Relevance

LeelaX is structured as a **research instrument** — each subsystem is isolated and documented for independent inspection and benchmarking.  
The aim is to make *deep RL for board games* accessible to students, practitioners, and employers seeking examples of:

- Applied Reinforcement Learning with PyTorch  
- MCTS guided by neural priors  
- Reward shaping and style conditioning  
- Transparent, test-driven scientific software development  

This project could support a **short research paper** or **technical blog** on topics such as:
> *“Reward Shaping and Aggression in Self-Play Reinforcement Learning for Chess.”*

---

## 🧰 Tech Stack

- **Language:** Python 3.10+  
- **Libraries:** `torch`, `numpy`, `python-chess`, `pytest`, `wandb` (optional)  
- **Environment:** macOS / Linux, virtualenv or Docker  
- **Editor:** Vim / VS Code / Jupyter for experimentation  

---

## 🚀 Quickstart

```bash
# clone and setup
git clone https://github.com/<yourname>/LeelaX.git
cd LeelaX
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# run all tests
pytest -q

# inspect encoding & policy
python -m leelax.env.inspect
```

## Licence
MIT License — open for academic and personal research use.
