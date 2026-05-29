# RunPod Training Guide

## Recommended GPU

| GPU | VRAM | Price/hr | Good for? | Notes |
|-----|------|----------|-----------|-------|
| **RTX 4090** | 24 GB | ~$0.34 | Best bang/buck | batch_size 512, fast MCTS, full model |
| **RTX 3090** | 24 GB | ~$0.21 | Budget choice | batch_size 512, slightly slower inference |
| **A5000** | 24 GB | ~$0.39 | Solid mid-range | batch_size 512, good throughput |
| **A100 40GB** | 40 GB | ~$0.79 | Max throughput | batch_size 1024+, deep nets |
| **L40S** | 48 GB | ~$1.09 | Overkill | batch_size 1024+, 20-block model |

**Recommended: RTX 4090 or RTX 3090** for single-GPU training. The 4090 has faster tensor cores which help the attention blocks.

CPU doesn't matter much — MCTS runs on CPU in between GPU eval calls. Any modern Xeon/EPYC with 8+ cores is fine.

## Pod Setup

### 1. Start a Pod
- **Container**: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`
- **Disk**: 50 GB (checkpoints + data + replay buffer)
- **Template**: "RunPod PyTorch 2.x" (community template)

### 2. Provision
Run this on pod start (paste or save as `setup.sh`):

```bash
# Install git & clone repo
apt-get update && apt-get install -y git python3-pip
git clone https://github.com/andersonniu08-boop/gomoku-ai.git
cd gomoku-ai

# Install deps (torch is already in the container)
pip install numpy flask

# Create directories
mkdir -p checkpoints data game_examples
```

### 3. Bootstrap a seed checkpoint

```bash
python -c "
from neural.model import GomokuNet
from selfplay.train import save_model_checkpoint
save_model_checkpoint(GomokuNet(), 'checkpoints/best.pt')
save_model_checkpoint(GomokuNet(), 'checkpoints/latest.pt')
print('Seeded')
"
```

## Launch Training

### Basic (recommended for first run)

```bash
cd /root/gomoku-ai

python -m selfplay.train \
  --num-iterations 200 \
  --games-per-iteration 50 \
  --batch-size 512 \
  --mcts-simulations 400 \
  --eval-simulations 400 \
  --eval-frequency 5 \
  --eval-games 50 \
  --learning-rate 0.001
```

### Multi-worker (faster)

Pod A — Trainer (runs on the GPU):
```bash
python -m selfplay.train \
  --num-iterations 200 \
  --games-per-iteration 0 \
  --batch-size 512 \
  --mcts-simulations 400 \
  --eval-frequency 10 \
  --eval-games 100
```
(`games_per_iteration=0` means it only ingests worker files)

Pod B–E — Workers (no GPU needed, or cheap CPU pods):
```bash
for i in {1..4}; do
  python -m selfplay.worker \
    --checkpoint-dir /mnt/shared/checkpoints \
    --output-dir /mnt/shared/game_examples \
    --num-simulations 400 \
    --worker-id "worker-$i" &
done
```

## Expected Timings

On a **single RTX 4090** with `games_per_iteration=50` and `mcts_simulations=400`:

| Step | Time | Notes |
|------|------|-------|
| 1 game (400 sims) | ~15–20s | GPU eval + MCTS |
| 50 games | ~15 min | Self-play phase |
| Training (512 batch) | ~10s | 512 batch × ~10 batches |
| Evaluation (50 games) | ~25 min | 400 sims, temp=0 |
| **1 full iteration** | **~40 min** | Self-play + train + eval (every 5th iter) |
| **200 iterations** | **~3-4 days** | Full training run |

## Monitoring

Training logs print every iteration:
```
  loss=4.2134  policy=3.5214  value=0.6920  entropy=3.4212  gnorm=5.00  lr=0.000976  buf=10240  sims=400  t=40.2s
  replay: 10240 pos, 834 openings (8.1% unique), moves={'0-9': ...}, vals={'win': 5120, 'draw': 0, 'loss': 5120}
```

Key signals:
- **loss** should trend down over time
- **entropy** should decrease (policy becomes more confident)
- **gnorm** should be near `max_grad_norm` (5.0) — means clipping is active
- **vals** should be roughly balanced (win ≈ loss)
- **unique openings** should increase (diversity is good)

## Keep-Alive

RunPod kills idle pods. Prevent disconnect:

```bash
# Keep SSH alive
echo "ClientAliveInterval 60" >> /etc/ssh/sshd_config
service ssh restart

# Or run training in tmux
tmux new -s train
python -m selfplay.train ...  # inside tmux
tmux detach  # Ctrl+B, D — reattach with tmux attach -t train
```

## Saving Checkpoints

Checkpoints auto-save to `checkpoints/`:
- `latest.pt` — updated every iteration
- `best.pt` — updated on promotion
- `best_iterXXX_winYY.pt` — named promotion snapshots

Sync to S3 / Google Drive periodically:
```bash
# Every 10 iterations, sync checkpoints
while true; do
  sleep 3600
  aws s3 sync checkpoints/ s3://my-bucket/gomoku/ --exclude "*.tmp"
done
```
