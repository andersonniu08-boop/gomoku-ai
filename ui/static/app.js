// ──────────────────────────────────────────────────────────────
//  NeuralGomoku — Canvas Board Renderer & Game Client
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
const infoBar = document.getElementById('info-bar');
const thinkingEl = document.getElementById('thinking');
const searchPanel = document.getElementById('search-panel');
const panelList = document.getElementById('panel-list');
const panelSubtitle = document.getElementById('panel-subtitle');

const sideBlackBtn = document.getElementById('btn-side-black');
const sideWhiteBtn = document.getElementById('btn-side-white');
const strengthFastBtn = document.getElementById('btn-strength-fast');
const strengthMedBtn = document.getElementById('btn-strength-medium');
const strengthStrongBtn = document.getElementById('btn-strength-strong');

// State
let state = null;
let humanPlayer = 1;    // 1 = Black, -1 = White — synced from server
let currentStrength = 'medium';
let heatmapOn = true;
let treeOn = true;
let thinking = false;

// Star points for 15x15 board (positions 3, 7, 11 in both axes)
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

  // Wood background with gradient
  const woodGrad = ctx.createLinearGradient(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  woodGrad.addColorStop(0, '#c8a45c');
  woodGrad.addColorStop(0.3, '#d4b06a');
  woodGrad.addColorStop(0.6, '#c49a4a');
  woodGrad.addColorStop(1, '#b8923e');
  ctx.fillStyle = woodGrad;
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  // Subtle wood grain lines
  ctx.strokeStyle = 'rgba(0,0,0,0.04)';
  ctx.lineWidth = 0.5;
  for (let gy = 0; gy < CANVAS_SIZE; gy += 7) {
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(CANVAS_SIZE, gy);
    ctx.stroke();
  }

  // Grid lines
  ctx.strokeStyle = '#4a3a28';
  ctx.lineWidth = 0.6;
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

  // Outer border
  ctx.strokeStyle = '#3a2a18';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(PADDING, PADDING, CANVAS_SIZE - 2 * PADDING, CANVAS_SIZE - 2 * PADDING);

  // Star points
  ctx.fillStyle = '#3a2a18';
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

  // Stone shadow
  ctx.beginPath();
  ctx.arc(x + 1, y + 1, STONE_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  ctx.fill();

  const grad = ctx.createRadialGradient(x - 2.5, y - 2.5, 1, x, y, STONE_RADIUS);
  if (isBlack) {
    grad.addColorStop(0, '#666');
    grad.addColorStop(0.6, '#1a1a1a');
    grad.addColorStop(1, '#0a0a0a');
  } else {
    grad.addColorStop(0, '#fff');
    grad.addColorStop(0.5, '#f0f0f0');
    grad.addColorStop(1, '#c0c0c0');
  }
  ctx.beginPath();
  ctx.arc(x, y, STONE_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = isBlack ? '#000' : '#999';
  ctx.lineWidth = 0.4;
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
    const alpha = 0.15 + intensity * 0.3;
    const blue = Math.floor(88 + 70 * intensity);
    const green = Math.floor(166 - 50 * intensity);
    ctx.beginPath();
    ctx.arc(x, y, STONE_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${88 - 30 * intensity}, ${green}, ${blue}, ${alpha})`;
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
  const youLabel = humanPlayer === 1 ? 'Black' : 'White';

  if (state.winner !== null) {
    const winnerLabel = state.winner === humanPlayer ? 'You' : 'AI';
    statusPlayer.textContent = `${winnerLabel} wins! (${state.winner === 1 ? 'Black' : 'White'})`;
  } else {
    statusPlayer.textContent = `Your turn ● ${youLabel}`;
  }

  const moveCount = state.board.flat().filter(v => v !== 0).length;
  const sims = state.simulations || 0;
  statusInfo.textContent = `You: ${youLabel} · Move #${Math.ceil(moveCount / 2)} · ${sims} sims`;
}

// ─── Side & strength selection ───

function selectSide(side) {
  sideBlackBtn.classList.toggle('active', side === 'black');
  sideWhiteBtn.classList.toggle('active', side === 'white');
  startNewGame(side, currentStrength);
}

function selectStrength(strength) {
  strengthFastBtn.classList.toggle('active', strength === 'fast');
  strengthMedBtn.classList.toggle('active', strength === 'medium');
  strengthStrongBtn.classList.toggle('active', strength === 'strong');
  currentStrength = strength;
  // A strength change applies on next new game — no instant restart.
  updateStatus(state);
}

// ─── API calls ───

async function makeMove(r, c) {
  if (thinking) return;

  // Show thinking overlay for the REAL duration of the search.
  thinking = true;
  thinkingEl.classList.remove('hidden');

  const resp = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ move: [r, c] }),
  });

  thinkingEl.classList.add('hidden');
  thinking = false;

  if (!resp.ok) {
    const err = await resp.json();
    console.warn('search API error:', err);
    return;
  }

  const newState = await resp.json();
  state = newState;
  humanPlayer = newState.human_player;
  currentStrength = newState.strength || currentStrength;

  drawBoardWithHover(newState);
  updateStatus(newState);
  updateSearchPanel(newState.search, newState.ai_move);
}

async function startNewGame(side, strength) {
  // Show thinking while server computes AI opening (only relevant for White side).
  thinking = true;
  thinkingEl.classList.remove('hidden');

  const resp = await fetch('/api/new-game', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ side, strength }),
  });

  thinkingEl.classList.add('hidden');
  thinking = false;

  const newState = await resp.json();
  state = newState;
  humanPlayer = newState.human_player;
  currentStrength = newState.strength || strength;

  drawBoardWithHover(newState);
  updateStatus(newState);
  updateSearchPanel(newState.search, newState.ai_move);
}

// ─── Hover preview ───

let hoverCell = null;   // [r, c] or null

function drawBoardWithHover(state) {
  drawBoard(state);

  if (!hoverCell || thinking) return;
  if (!state || state.winner !== null) return;

  const [r, c] = hoverCell;
  if (state.board[r][c] !== 0) return;
  const isLegal = state.legal_moves.some(([lr, lc]) => lr === r && lc === c);
  if (!isLegal) return;

  const { x, y } = gridToPixel(r, c);

  // Ghost stone (translucent orb)
  const isBlack = state.current_player === 1;
  const grad = ctx.createRadialGradient(x - 3, y - 3, 1, x, y, STONE_RADIUS);
  if (isBlack) {
    grad.addColorStop(0, 'rgba(85, 85, 85, 0.55)');
    grad.addColorStop(1, 'rgba(17, 17, 17, 0.55)');
  } else {
    grad.addColorStop(0, 'rgba(255, 255, 255, 0.55)');
    grad.addColorStop(1, 'rgba(204, 204, 204, 0.55)');
  }
  ctx.beginPath();
  ctx.arc(x, y, STONE_RADIUS, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();
}

canvas.addEventListener('mousemove', (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;
  const cell = gridFromPixel(px, py);

  const prev = hoverCell;
  hoverCell = cell;

  // Only redraw if the hover cell actually changed
  if (state) {
    if (prev && cell && prev[0] === cell[0] && prev[1] === cell[1]) return;
    drawBoardWithHover(state);
  }
});

canvas.addEventListener('mouseleave', () => {
  if (hoverCell && state) {
    hoverCell = null;
    drawBoard(state);
  }
});

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
  updateInfoBar();
  if (state) drawBoardWithHover(state);
});

document.getElementById('toggle-tree').addEventListener('click', () => {
  treeOn = !treeOn;
  document.getElementById('toggle-tree').classList.toggle('active');
  updateInfoBar();
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

document.getElementById('btn-new-game').addEventListener('click', () => {
  const side = sideBlackBtn.classList.contains('active') ? 'black' : 'white';
  startNewGame(side, currentStrength);
});

sideBlackBtn.addEventListener('click', () => {
  if (!sideBlackBtn.classList.contains('active')) {
    selectSide('black');
  }
});

sideWhiteBtn.addEventListener('click', () => {
  if (!sideWhiteBtn.classList.contains('active')) {
    selectSide('white');
  }
});

strengthFastBtn.addEventListener('click', () => {
  if (!strengthFastBtn.classList.contains('active')) {
    selectStrength('fast');
  }
});

strengthMedBtn.addEventListener('click', () => {
  if (!strengthMedBtn.classList.contains('active')) {
    selectStrength('medium');
  }
});

strengthStrongBtn.addEventListener('click', () => {
  if (!strengthStrongBtn.classList.contains('active')) {
    selectStrength('strong');
  }
});

// ─── Info bar ───

function updateInfoBar() {
  const parts = [];
  if (heatmapOn) {
    parts.push('Heatmap shows neural-network-guided search priorities — brighter cells indicate higher MCTS visit counts.');
  }
  if (treeOn) {
    parts.push('Tree panel displays the MCTS search tree ranked by simulation visits during the AI\'s turn.');
  }
  infoBar.textContent = parts.join(' ');
}

// ─── Init ───

startNewGame('black', 'medium');
updateInfoBar();
