# LeelaX – AlphaZero-inspiriertes RL für Schach (MVP)

Ein bewusst **schlanker** Prototyp nach dem AlphaZero-/Leela-Chess-Prinzip:
- Self-Play + **MCTS (PUCT)**
- kleines **ResNet** mit **Policy**- und **Value-Head**
- **Replay-Buffer**, Training mit **CE+MSE**
- **Evaluation/Arena** + Metriken (inkl. Aggressions-Analysen)
- Fokus: **Nachvollziehbarkeit & Experimente** (nicht maximale Elo)

---

## Projektstatus
MVP im Aufbau. Ziel: lauffähiger End-to-End-Loop mit kleinen Settings (CPU-tauglich).

---

## Repo-Struktur (MVP)
leelax/
leelax/
env/ # python-chess Wrapper, Encodings, Legalitäts-Masken
net/ # ResNet + Policy/Value-Heads
mcts/ # PUCT-Suche, Dirichlet am Root
selfplay/ # Worker, Replay-Buffer
train/ # Dataset/Loader, Trainingsloop
eval/ # Arena, Aggressionsmetriken
utils/ # Logging, Checkpoints, Config/CLI
scripts/ # Shell-Einstiegspunkte
notebooks/ # Debug/Analyse (optional)
README.md
requirements.txt

---

## Installation (macOS, venv)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# if Torch issues
pip install torch --index-url https://download.pytorch.org/whl/cpu
