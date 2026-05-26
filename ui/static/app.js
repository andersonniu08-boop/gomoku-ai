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
let state = null;
let heatmapOn = true;
let treeOn = true;
let thinking = false;

// Star points for 15×15 board (positions 3, 7, 11 in both axes)
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
    ctx.beginPath();
    ctx.moveTo(PADDING, p);
    ctx.lineTo(CANVAS_SIZE - PADDING, p);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p, PADDING);
    ctx.lineTo(p, CANVAS_SIZE - PADDING);
    ctx.stroke();
  }

  // Star points
  ctx.fillStyle = '#333';
  for (const [r, c] of STAR_POINTS) {
    const { x, y } = gridToPixel(r, c);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
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
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
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
    if (grid[r][c] !== 0) continue;

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
    const isAIMove = aiMove && entry.r === aiMove[0] && entry.c === aiMove[1];
    const stoneType = isAIMove ? '●' : '○';
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
    const winnerName = state.winner === 1 ? 'Black' : 'White';
    statusPlayer.textContent = `🏆 ${winnerName} wins!`;
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
  if (!resp.ok) {
    const err = await resp.json();
    console.warn('search API error:', err);
    return;
  }

  const newState = await resp.json();
  state = newState;

  // Show thinking indicator before AI move
  if (newState.ai_move) {
    thinking = true;
    thinkingEl.classList.remove('hidden');
    drawBoard({ ...newState, last_move: [r, c] });
    await new Promise(r => setTimeout(r, 300));
    thinkingEl.classList.add('hidden');
    thinking = false;
  }

  drawBoard(newState);
  updateStatus(newState);
  updateSearchPanel(newState.search, newState.ai_move);
}

async function resetGame() {
  const resp = await fetch('/api/new-game', { method: 'POST' });
  const newState = await resp.json();
  state = newState;
  drawBoard(newState);
  updateStatus(newState);
  searchPanel.classList.remove('visible');
}

// ─── Canvas click handler ───

canvas.addEventListener('click', (e) => {
  if (thinking) return;
  if (state && state.winner !== null) return;

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;
  const cell = gridFromPixel(px, py);
  if (!cell) return;

  const [r, c] = cell;
  if (!state) return;
  if (state.board[r][c] !== 0) return;

  const isLegal = state.legal_moves.some(([lr, lc]) => lr === r && lc === c);
  if (!isLegal) return;

  makeMove(r, c);
});

// ─── Toggle handlers ───

document.getElementById('toggle-heatmap').addEventListener('click', () => {
  heatmapOn = !heatmapOn;
  document.getElementById('toggle-heatmap').classList.toggle('active');
  if (state) drawBoard(state);
});

document.getElementById('toggle-tree').addEventListener('click', () => {
  treeOn = !treeOn;
  document.getElementById('toggle-tree').classList.toggle('active');
  if (treeOn && state && state.search) {
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
