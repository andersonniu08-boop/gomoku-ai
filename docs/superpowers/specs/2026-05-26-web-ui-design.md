# Phase 5 — Web UI and Visualization

## Overview

Build a web-based UI for playing Gomoku against the AlphaZero-style AI engine,
with toggleable visualizations for MCTS search statistics and policy heatmaps.

## Scope

**In scope:**
- Web-based game UI: 15×15 board, play against the AI
- MCTS search tree visualization (floating panel, toggleable)
- Policy/probability heatmap overlay on the board (toggleable)

**Deferred to future:**
- Value landscape visualization
- Search tree depth expansion / subtree exploration

## Architecture

```
┌─────────────┐    POST /api/search     ┌──────────────────┐
│  Browser     │ ◄───────────────────── │  Flask Server     │
│  (Canvas UI) │    JSON response       │  (ui/server.py)   │
│              │                        │                  │
│  index.html  │                        │  GomokuWrapper    │
│  style.css   │                        │  MCTS             │
│  app.js      │                        │  Board            │
└─────────────┘                        └──────────────────┘
```

### Import Flow

```
server.py
  ├── from neural.wrapper import GomokuInferenceWrapper
  ├── from selfplay.mcts import MCTS
  └── from engine.board import Board, Player
```

Zero changes to `engine/`, `neural/`, or `selfplay/`. The server is a new
consumer of existing interfaces.

### Dependencies

- `flask` — only new pip dependency (goes into `requirements.txt`)

Everything else (numpy, torch) already in the project.

## Backend API

### `GET /`

Serves `ui/static/index.html` as the single-page application entry point.

### `POST /api/new-game`

Reset the server-side board state for a new game.

**Request:** (empty body)

**Response:**
```json
{
  "board": [[0,0,0,...], ...],
  "current_player": 1,
  "legal_moves": [[7,7]],
  "winner": null
}
```

### `POST /api/search`

Run MCTS search from the current position and return results.

**Request:**
```json
{
  "move_history": [[7,7], [6,6]],
  "current_player": -1
}
```

**Response:**
```json
{
  "board": [[0,0,1,...], ...],
  "current_player": 1,
  "winner": null,
  "legal_moves": [[r,c], ...],
  "search": {
    "visit_counts": { "7,7": 247, "8,6": 89, ... },
    "q_values":     { "7,7": 0.42, "8,6": 0.18, ... },
    "priors":       { "7,7": 0.15, "8,6": 0.09, ... },
    "total_simulations": 400
  },
  "ai_move": [7, 7]
}
```

The server reconstructs the board from `move_history`, runs `MCTS.search_with_stats()`,
converts `SearchResult` to JSON (mapping tuple keys to "r,c" strings), and returns the
result alongside the board state.

### Session Architecture

Board state lives in memory on the server. A single client is assumed — no session
management, no database. The server resets state on `/api/new-game`.

## Frontend

### File Structure

```
ui/
├── __init__.py
├── server.py              # Flask app
└── static/
    ├── index.html         # Single-page UI
    ├── style.css          # Board, panel, toggle styles
    └── app.js             # Canvas board renderer, API, toggle logic
```

### Board Rendering (`app.js`)

- HTML Canvas element, ~450×450px
- 15×15 grid with wood-colored background (`#DEB361`)
- Grid lines at evenly spaced intersections with 20px padding
- 9 star-point markers (positions 3, 7, 11 in both coordinates)
- Stones rendered with radial gradients (black: `#555`→`#111`, white: `#fff`→`#ccc`)
- Click handler: mouse position → nearest intersection → validate empty → POST search

### Game Flow

1. **Page load:** POST `/api/new-game`, render empty board
2. **Human move:** Click intersection → POST `/api/search` with updated history
3. **Server response:** Returns MCTS results + AI's chosen move
4. **Render:** Show human stone, brief "thinking..." indicator, then AI stone
5. **State updates:** If heatmap toggle on → overlay colors; if search tree on → update panel
6. **Game over:** If `winner` is non-null, disable board clicks, highlight winning line

### Heatmap Overlay

- Rendered on the same Canvas after stones
- Translucent circles on empty intersections only
- Color: green (high visit count) → red (low visit count)
- Normalized across the root children's visit counts
- Toggled via a button, on by default

### Search Tree Panel

- Floating `div` positioned to the right of the board
- Shows top-10 moves ranked by visit count
- Each row: move coordinates, stone icon, visit count, Q-value
- Rows colored with a left-border accent (green = best, blue = second, etc.)
- "✕" close button; toggle button reopens it

### Toggle Buttons

Both toggles sit between the board and the status bar. Both on by default.

- **Heatmap** button — line: "■ Heatmap", active border color: `#e67e22`
- **Search Tree** button — line: "◨ Search Tree", active border color: `#2980b9`

Each button has an active/inactive visual state (different border opacity / background).

### Styling (`style.css`)

Minimal, clean, dark-ish terminal aesthetic:

- Dark background: `#1a1a2e`
- Card/panel backgrounds: `#16213e`
- Text: light gray (`#e0e0e0`)
- Toggle buttons: outlined style with colored border when active

## MCTS API Contract

The frontend depends on `SearchResult` attributes only:

| Field | Type | Used for |
|-------|------|----------|
| `visit_counts` | `dict[tuple, int]` | Heatmap coloring, search tree ranking |
| `q_values` | `dict[tuple, float]` | Search tree display |
| `priors` | `dict[tuple, float]` | Search tree display |
| `total_simulations` | `int` | Status display |

The frontend never reads MCTS internal nodes, never traverses the tree, and
never depends on `MCTSNode` fields. This decouples the UI from MCTS
implementation details. Future optimizations (batching, virtual loss,
GPU inference) do not affect the UI as long as `search_with_stats()` still
returns a `SearchResult`.

## Out of Scope

- Value landscape visualization
- Subtree expansion / interactive tree browsing
- Game history navigation / replay
- Multi-game management or session persistence
- User authentication or multiplayer
- Mobile-responsive layout
- Accessibility beyond basic semantic HTML
- The `explain/` module (saliency, activations, comparison) is not called by
  this phase. Those features may be integrated in a follow-up but are intentionally
  excluded from the initial ship.
