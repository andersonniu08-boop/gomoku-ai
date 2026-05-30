# Project Status

**Date:** May 2026
**Status:** Complete — Portfolio-Ready

## Summary

NeuralGomoku is a fully functional neural-network-driven Gomoku engine with a
complete training pipeline, distributed worker system, and web-based gameplay
interface. All core components are implemented, tested (440 tests), and
operational.

Designed and presented as a portfolio demonstration of end-to-end ML systems
engineering: PyTorch model architecture, GPU-optimized MCTS inference,
distributed self-play, and web integration.

## Component Status

| Component | Status | Notes |
|---|---|---|
| Board engine | Complete | 15x15, NumPy-backed, incremental win detection |
| Threat detection | Complete | Pattern-based (FIVE, OPEN_FOUR, CLOSED_FOUR, OPEN_THREE) |
| Tactical solver | Complete | Deterministic forced-line search |
| Neural network | Complete | 10-block residual CNN, attention, SE/CBAM, DropPath, ~3.7M params |
| Inference wrapper | Complete | Checkpoint loading, device placement, batch eval |
| MCTS search | Complete | PUCT, batched descent, virtual loss, tree reuse, threat override |
| Zobrist eval cache | Complete | Cached neural evaluations for board positions |
| Self-play generation | Complete | Temperature stages, Dirichlet noise, resignation heuristics |
| Replay buffer | Complete | 500K FIFO, D4 symmetry augmentation, persistence |
| Training loop | Complete | Mixed precision, gradient clipping, CosineAnnealingLR |
| Model evaluation | Complete | Elo tracking, win-rate-based promotion, checkpoint management |
| Distributed workers | Complete | Checkpoint polling, game file generation, graceful shutdown |
| Web UI | Complete | Flask server, Canvas board, search tree visualization |
| Explainability | Complete | Saliency maps, activation visualization, human-vs-AI comparison |
| Benchmarking | Complete | MCTS sims/sec, neural architecture benchmarks, tactical suite |
| Test suite | Complete | 440 tests |

## Trained Model

- **Training hardware:** NVIDIA A100-80GB (RunPod) + local RTX 3050
- **Checkpoints:** `checkpoints/best.pt` and `checkpoints/latest.pt` (~15 MiB each)
- **Architecture:** 10 blocks, 128 channels, ~3.7M parameters
- **Model loads and plays via UI:** Yes
- **Further training possible:** Yes — resumes from checkpoint

## Development Timeline

| Phase | Description | Status |
|---|---|---|
| Core Training | Self-play, replay buffer, training loop, evaluation | Complete |
| Performance | Batched GPU inference, virtual loss, profiling | Complete |
| Architecture | Attention, SE/CBAM, stochastic depth, dilated pyramid | Complete |
| Explainability | Saliency maps, activation visualization, move comparison | Complete |
| UI & Infrastructure | Web interface, search tree viz, Elo tracking | Complete |
| Distribution | Multi-worker self-play, file-based coordination | Complete |

## Known Limitations

- Model plays at amateur-intermediate strength after 25+ iterations.
  Professional-strength Gomoku engines typically require 100+ iterations with
  larger batch sizes and deeper networks.
- Web UI runs synchronously (no WebSocket); blocks during AI search.
- Multi-worker training requires manual process management.
- Checkpoints stored locally; no cloud artifact storage.

## Intended Audience

This project is designed for:

- **ML Engineers** — evaluating systems design, training pipelines, GPU optimization
- **Recruiters** — assessing end-to-end project delivery capability
- **Students** — studying reinforcement learning and MCTS implementation
- **Hobbyists** — playing Gomoku against a neural-network-powered AI
