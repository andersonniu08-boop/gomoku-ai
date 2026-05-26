# Phase 5 — Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web-based UI to play Gomoku against the AI with toggleable MCTS visualizations.

**Architecture:** Flask server loads the trained model + MCTS at startup. Single-page Canvas-based frontend communicates via JSON API. Two API endpoints: new-game and search. Visualizations (heatmap, search tree) are toggleable overlays rendered client-side from search stats.

**Tech Stack:** Flask (Python), vanilla HTML/CSS/JS (no build step), HTML Canvas for board rendering.

---

## File Structure

```
Files to create:
  ui/__init__.py          # Empty package marker
  ui/server.py            # Flask app: load model, serve API + static files
  ui/static/index.html    # Single-page UI skeleton
  ui/static/style.css     # Board + panel + toggle styling
  ui/static/app.js        # Canvas board renderer, API calls, toggle logic

Files to modify:
  requirements.txt        # Add flask
```

**Design dependencies — no existing code changes needed:**
- `server.py` imports `GomokuInferenceWrapper` from `neural.wrapper`
- `server.py` imports `MCTS` from `selfplay.mcts`
- `server.py` imports `Board, Player` from `engine.board`
- The `SearchResult` dataclass is used as-is from `selfplay.mcts`

---

### Task 1: Add Flask dependency

**Files:**
- Modify: `requirements.txt:1`

- [ ] **Step 1: Add flask to requirements**

Append to `requirements.txt`:
```
flask
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add flask dependency for web UI"
```

---

### Task 2: Create Flask server

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/server.py`

The server:
- Loads `GomokuInferenceWrapper` from `checkpoints/best.pt`
- Creates an `MCTS` instance (400 simulations, threat_override=True)
- Maintains a `Board` in memory per game session
- Serves `ui/static/index.html` at `GET /`
- `POST /api/new-game` — resets board, returns initial state
- `POST /api/search` — accepts `{move: [r,c]}`, applies the human move, runs MCTS, returns search results + AI response
- Parses `"r,c"` string keys from JSON (since tuple keys don't survive JSON serialization)

- [ ] **Step 1: Create `ui/__init__.py`**

Empty file.

```python
```

- [ ] **Step 2: Create `ui/server.py`**

```python
"""Flask server — loads model + MCTS, serves web UI API."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from engine.board import Board, Player
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------
# Global engine state — loaded once at import time.
# ---------------------------------------------------------------------------

wrapper = GomokuInferenceWrapper(str(CHECKPOINT_DIR / "best.pt"))
mcts = MCTS(wrapper, num_simulations=400, threat_override=True)

# Current game board — reset on new-game.
board: Board = Board()


def _board_state_json() -> dict:
    """Serialise the current board for API responses."""
    return {
        "board": board.grid.tolist(),
        "current_player": int(board.current_player),
        "legal_moves": board.get_legal_moves(),
        "winner": int(board.check_win()) if board.check_win() else None,
    }


def _search_result_json(result) -> dict:
    """Convert SearchResult tuples to "r,c" string keys for JSON."""
    if not result.visit_counts:
        return {}
    return {
        "visit_counts": {f"{r},{c}": v for (r, c), v in result.visit_counts.items()},
        "q_values": {f"{r},{c}": v for (r, c), v in result.q_values.items()},
        "priors": {f"{r},{c}": v for (r, c), v in result.priors.items()},
        "total_simulations": result.total_simulations,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/new-game", methods=["POST"])
def new_game():
    global board
    board = Board()
    return jsonify(_board_state_json())


@app.route("/api/search", methods=["POST"])
def search():
    global board

    data = request.get_json()
    if not data or "move" not in data:
        return jsonify({"error": "missing move"}), 400

    r, c = data["move"]

    # Validate and apply the human move.
    if board.grid[r, c] != 0:
        return jsonify({"error": "cell occupied"}), 400
    if board.check_win() is not None:
        return jsonify({"error": "game already over"}), 400

    board.make_move(r, c)

    # Check if human won with that move.
    if board.check_win() is not None:
        return jsonify({**_board_state_json(), "search": {}, "ai_move": None})

    # Run MCTS.
    result = mcts.search_with_stats(board)
    ai_move = max(result.visit_counts, key=result.visit_counts.get)
    board.make_move(*ai_move)

    response = {
        **_board_state_json(),
        "search": _search_result_json(result),
        "ai_move": list(ai_move),
    }
    return jsonify(response)


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

- [ ] **Step 3: Quick smoke test — verify the server starts**

```bash
cd /home/anderson/projects/gomoku-ai
python -c "from ui.server import app; print('OK')"
```

Expected: prints "OK" with no import errors.

- [ ] **Step 4: Commit**

```bash
git add ui/__init__.py ui/server.py
git commit -m "feat: add Flask server with game API"
```

---

### Task 3: Create index.html

**Files:**
- Create: `ui/static/index.html`

- [ ] **Step 1: Create `ui/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gomoku AI</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <div id="app">
    <header>
      <h1>Gomoku AI</h1>
    </header>

    <main>
      <div id="board-container">
        <canvas id="board-canvas" width="480" height="480"></canvas>
        <div id="status-bar">
          <span id="status-player"></span>
          <div id="toggle-bar">
            <button id="toggle-heatmap" class="toggle active">■ Heatmap</button>
            <button id="toggle-tree" class="toggle active">◨ Search Tree</button>
            <button id="btn-new-game">New Game</button>
          </div>
          <span id="status-info"></span>
        </div>
      </div>

      <aside id="search-panel" class="visible">
        <div id="panel-header">
          <span>Search Tree</span>
          <button id="panel-close">✕</button>
        </div>
        <div id="panel-subtitle"></div>
        <div id="panel-list"></div>
      </aside>
    </main>

    <div id="thinking" class="hidden">thinking...</div>
  </div>

  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add ui/static/index.html
git commit -m "feat: add HTML skeleton for web UI"
```

---

### Task 4: Create style.css

**Files:**
- Create: `ui/static/style.css`

- [ ] **Step 1: Create `ui/static/style.css`**

```css
/* Dark terminal aesthetic — clean, minimal, functional. */

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  background: #1a1a2e;
  color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 24px;
}

#app {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

header h1 {
  font-size: 20px;
  font-weight: 600;
  letter-spacing: 0.5px;
  color: #e0e0e0;
}

header h1::before {
  content: "◆ ";
  color: #4CAF50;
}

main {
  display: flex;
  gap: 20px;
  align-items: flex-start;
}

#board-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

#board-canvas {
  border-radius: 4px;
  cursor: pointer;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
}

/* Status bar */
#status-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  margin-top: 8px;
  font-size: 13px;
  color: #aaa;
}

#toggle-bar {
  display: flex;
  gap: 8px;
}

.toggle {
  padding: 4px 12px;
  border: 2px solid #444;
  border-radius: 4px;
  background: transparent;
  color: #aaa;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.toggle.active {
  color: #e0e0e0;
}

.toggle#toggle-heatmap.active {
  border-color: #e67e22;
  color: #e67e22;
}

.toggle#toggle-tree.active {
  border-color: #2980b9;
  color: #2980b9;
}

#btn-new-game {
  padding: 4px 16px;
  border: none;
  border-radius: 4px;
  background: #4CAF50;
  color: white;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
}

#btn-new-game:hover {
  background: #43a047;
}

/* Thinking indicator */
#thinking {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: #16213e;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 12px 24px;
  font-size: 14px;
  color: #e0e0e0;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
  z-index: 100;
}

#thinking.hidden {
  display: none;
}

/* Search tree panel */
#search-panel {
  width: 270px;
  background: #16213e;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  display: none;
  flex-direction: column;
  gap: 6px;
}

#search-panel.visible {
  display: flex;
}

#panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
  font-weight: 600;
}

#panel-close {
  background: none;
  border: none;
  color: #888;
  font-size: 16px;
  cursor: pointer;
}

#panel-close:hover {
  color: #e0e0e0;
}

#panel-subtitle {
  font-size: 11px;
  color: #888;
}

#panel-list {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.move-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 5px 8px;
  background: #0f3460;
  border-radius: 3px;
  font-size: 13px;
  border-left: 3px solid #555;
}

.move-row .move-coord {
  font-weight: 500;
}

.move-row .move-stats {
  font-size: 11px;
  color: #aaa;
}

.move-row.rank-1 { border-left-color: #4CAF50; }
.move-row.rank-2 { border-left-color: #2980b9; }
.move-row.rank-3 { border-left-color: #f39c12; }
```

- [ ] **Step 2: Commit**

```bash
git add ui/static/style.css
git commit -m "feat: add dark-themed CSS for board UI"
```

---

### Task 5: Create app.js — Canvas board renderer + game logic

**Files:**
- Create: `ui/static/app.js`

This is the main file. Key functions:

- `drawBoard(state)` — renders grid, stones, heatmap overlay
- `drawGrid(ctx)` — grid lines, star points
- `drawStone(ctx, row, col, player)` — gradient stones
- `drawHeatmap(ctx, visitCounts)` — colored translucent circles
- `async makeMove(r, c)` — POST `/api/search`, update state, redraw
- `resetGame()` — POST `/api/new-game`, redraw
- `updateSearchPanel(result)` — populate the search tree panel
- `updateStatus(state)` — turn indicator, move count
- Click handler on canvas → nearest intersection → `makeMove()`

- [ ] **Step 1: Create `ui/static/app.js`**

```javascript
// ──────────────────────────────────────────────────────────────
//  Gomoku AI — Canvas Board Renderer & Game Client
// ──────────────────────────────────────────────────────────────

const BOARD_SIZE = 15;
const CANVAS_SIZE = 480;
const PADDING = 24;
const CELL = (CANVAS_SIZE - 2 * PADDING) / (BOARD_SIZE - 1);
const STONE_RADIUS = CELL * 0.42;

const canvas = document.getElementById('board-canvas');
const ctx = canvas.getContext('2d');

// DOM refs
const statusPlayer = document.getElementById('status-player');
const statusInfo = document.getElementById('status-info');
const thinkingEl = document.getElementById('thinking');
const searchPanel = document.getElementById('search-panel');
const panelList = document.getElementById('panel-list');
const panelSubtitle = document.getElementById('panel-subtitle');

// State
let boardState = null;       // last /api/search response
let heatmapOn = true;
let treeOn = true;
let thinking = false;

// Star points for 15×15 board
const STAR_POINTS = [
  [3,3], [3,7], [3,11],
  [7,3], [7,7], [7,11],
  [11,3], [11,7], [11,11],
];

// ─── Coordinate helpers ───

function gridFromPixel(px, py) {
  const c = Math.round((px - PADDING) / CELL);
  const r = Math.round((py - PADDING) / CELL);
  if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) return null;
  return [r, c];
}

function gridToPixel(r, c) {
  return { x: PADDING + c * CELL, y: PADDING + r * CELL };
}

// ─── Drawing ───

function drawBoard(state) {
  const grid = state.board;
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // Wood background
  ctx.fillStyle = '#DEB361';
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // Grid lines
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 0.8;
  for (let i = 0; i < BOARD_SIZE; i++) {
    const p = PADDING + i * CELL;
    ctx.beginPath(); ctx.moveTo(PADDING, p); ctx.lineTo(CANVAS_SIZE - PADDING, p); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(p, PADDING); ctx.lineTo(p, CANVAS_SIZE - PADDING); ctx.stroke();
  }

  // Star points
  ctx.fillStyle = '#333';
  for (const [r, c] of STAR_POINTS) {
    const { x, y } = gridToPixel(r, c);
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
  }

  // Heatmap overlay (before stones, on empty cells)
  if (heatmapOn && state.search && state.search.visit_counts) {
    drawHeatmap(state.search.visit_counts, grid);
  }

  // Stones
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (grid[r][c] !== 0) {
        drawStone(r, c, grid[r][c] === 1);
      }
    }
  }

  // Last-move marker (small red dot on the most recent stone)
  if (state.last_move) {
    const { x, y } = gridToPixel(state.last_move[0], state.last_move[1]);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#e74c3c';
    ctx.fill();
  }
}

function drawStone(row, col, isBlack) {
  const { x, y } = gridToPixel(row, col);
  const grad = ctx.createRadialGradient(x - 3, y - 3, 1, x, y, STONE_RADIUS);
  if (isBlack) {
    grad.addColorStop(0, '#555');
    grad.addColorStop(1, '#111');
  } else {
    grad.addColorStop(0, '#fff');
    grad.addColorStop(1, '#ccc');
  }
  ctx.beginPath();
  ctx.arc(x, y, STONE_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = isBlack ? '#000' : '#aaa';
  ctx.lineWidth = 0.5;
  ctx.stroke();
}

function drawHeatmap(visitCounts, grid) {
  const visits = Object.values(visitCounts);
  const maxVisits = Math.max(...visits, 1);

  for (const [key, count] of Object.entries(visitCounts)) {
    const [r, c] = key.split(',').map(Number);
    if (grid[r][c] !== 0) continue;  // occupied

    const { x, y } = gridToPixel(r, c);
    const intensity = Math.min(count / maxVisits, 1);
    const red = Math.floor(255 * (1 - intensity));
    const green = Math.floor(200 * intensity);
    ctx.beginPath();
    ctx.arc(x, y, STONE_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${red}, ${green}, 0, 0.35)`;
    ctx.fill();
  }
}

// ─── Search tree panel ───

function updateSearchPanel(result, aiMove) {
  if (!result || !result.visit_counts || Object.keys(result.visit_counts).length === 0) {
    searchPanel.classList.remove('visible');
    return;
  }

  panelSubtitle.textContent = `${result.total_simulations} sims`;

  // Sort moves by visit count desc
  const entries = Object.entries(result.visit_counts)
    .map(([key, visits]) => ({
      key,
      r: parseInt(key.split(',')[0]),
      c: parseInt(key.split(',')[1]),
      visits,
      q: result.q_values[key] || 0,
      prior: result.priors[key] || 0,
    }))
    .sort((a, b) => b.visits - a.visits)
    .slice(0, 10);

  panelList.innerHTML = '';
  entries.forEach((entry, i) => {
    const row = document.createElement('div');
    row.className = `move-row rank-${Math.min(i + 1, 3)}`;
    const stoneType = aiMove && entry.r === aiMove[0] && entry.c === aiMove[1] ? '●' : '○';
    row.innerHTML = `
      <span class="move-coord">${stoneType} (${entry.r},${entry.c})</span>
      <span class="move-stats">${entry.visits} · Q=${entry.q.toFixed(2)}</span>
    `;
    panelList.appendChild(row);
  });

  if (treeOn) searchPanel.classList.add('visible');
}

// ─── Status bar ───

function updateStatus(state) {
  const playerName = state.current_player === 1 ? 'Black' : 'White';
  statusPlayer.textContent = `● ${playerName}'s turn`;
  if (state.winner !== null) {
    const winner = state.winner === 1 ? 'Black' : 'White';
    statusPlayer.textContent = `🏆 ${winner} wins!`;
  }
  const moveCount = state.board.flat().filter(v => v !== 0).length;
  statusInfo.textContent = `Move #${Math.ceil(moveCount / 2)}`;
  if (state.search && state.search.total_simulations) {
    statusInfo.textContent += ` · ${state.search.total_simulations} sims`;
  }
}

// ─── API calls ───

async function makeMove(r, c) {
  if (thinking) return;

  const resp = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ move: [r, c] }),
  });
  if (!resp.ok) return;

  state = await resp.json();
  boardState = state;

  // Show thinking before AI move
  if (state.ai_move) {
    thinking = true;
    thinkingEl.classList.remove('hidden');
    drawBoard({ ...state, last_move: [r, c] });
    await new Promise(r => setTimeout(r, 300));  // brief pause for readability
    thinkingEl.classList.add('hidden');
    thinking = false;
  }

  drawBoard(state);
  updateStatus(state);
  updateSearchPanel(state.search, state.ai_move);
}

async function resetGame() {
  const resp = await fetch('/api/new-game', { method: 'POST' });
  state = await resp.json();
  boardState = state;
  drawBoard(state);
  updateStatus(state);
  searchPanel.classList.remove('visible');
}

// ─── Canvas click handler ───

canvas.addEventListener('click', (e) => {
  if (thinking || (boardState && boardState.winner !== null)) return;
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;
  const cell = gridFromPixel(px, py);
  if (!cell) return;
  const [r, c] = cell;

  // Check empty and legal
  if (!boardState) return;
  const isLegal = boardState.legal_moves.some(([lr, lc]) => lr === r && lc === c);
  if (!isLegal || boardState.board[r][c] !== 0) return;

  makeMove(r, c);
});

// ─── Toggle handlers ───

document.getElementById('toggle-heatmap').addEventListener('click', () => {
  heatmapOn = !heatmapOn;
  document.getElementById('toggle-heatmap').classList.toggle('active');
  if (boardState) drawBoard(boardState);
});

document.getElementById('toggle-tree').addEventListener('click', () => {
  treeOn = !treeOn;
  document.getElementById('toggle-tree').classList.toggle('active');
  if (treeOn && boardState && boardState.search) {
    searchPanel.classList.add('visible');
  } else {
    searchPanel.classList.remove('visible');
  }
});

document.getElementById('panel-close').addEventListener('click', () => {
  treeOn = false;
  document.getElementById('toggle-tree').classList.remove('active');
  searchPanel.classList.remove('visible');
});

document.getElementById('btn-new-game').addEventListener('click', resetGame);

// ─── Init ───

resetGame();
```

- [ ] **Step 2: Commit**

```bash
git add ui/static/app.js
git commit -m "feat: add Canvas-based board renderer and game client"
```

---

### Task 6: Integration smoke test

**Files:**
- Test: run the server and verify the UI loads

- [ ] **Step 1: Start the server and verify**

```bash
cd /home/anderson/projects/gomoku-ai
python -c "
from ui.server import app
import json

with app.test_client() as client:
    # New game
    resp = client.post('/api/new-game')
    data = json.loads(resp.data)
    assert data['board'][7][7] == 0
    assert len(data['legal_moves']) == 1
    assert data['legal_moves'][0] == [7, 7]
    print('new-game OK')

    # Make first move
    resp = client.post('/api/search', json={'move': [7, 7]})
    data = json.loads(resp.data)
    assert data['board'][7][7] == 1  # Black's stone
    assert 'search' in data
    assert 'ai_move' in data
    print('search OK')

    # Invalid move
    resp = client.post('/api/search', json={'move': [7, 7]})
    assert resp.status_code == 400
    print('error handling OK')

    # HTML renders
    resp = client.get('/')
    assert resp.status_code == 200
    print('index.html OK')

    # Static files
    resp = client.get('/style.css')
    assert resp.status_code == 200
    resp = client.get('/app.js')
    assert resp.status_code == 200
    print('static files OK')

print('All integration checks passed')
"
```

- [ ] **Step 2: Commit any final tweaks**

```bash
git commit -am "fix: minor integration adjustments"
```
