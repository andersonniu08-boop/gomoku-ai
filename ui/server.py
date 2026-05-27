"""Flask server — loads model + MCTS, serves web UI API."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so engine/, neural/, selfplay/ resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

# Strength presets — each configures a dedicated MCTS instance.
STRENGTH_PRESETS: dict[str, dict] = {
    "fast": {
        "num_simulations": 200,
        "label": "Fast",
        "description": "Quick casual play",
    },
    "medium": {
        "num_simulations": 800,
        "label": "Medium",
        "description": "Balanced strength and speed",
    },
    "strong": {
        "num_simulations": 2000,
        "label": "Strong",
        "description": "Strongest tactical play",
    },
}

# Cache one MCTS instance per preset.
_mcts_instances: dict[str, MCTS] = {}


def _get_mcts(strength: str) -> MCTS:
    """Return cached MCTS instance for a given strength preset."""
    if strength not in _mcts_instances:
        params = STRENGTH_PRESETS[strength]
        _mcts_instances[strength] = MCTS(
            wrapper,
            num_simulations=params["num_simulations"],
            threat_override=True,
        )
    return _mcts_instances[strength]


# Current game state.
board: Board = Board()
human_player: Player = Player.BLACK
current_strength: str = "medium"


def _board_state_json() -> dict:
    """Serialise the current board for API responses."""
    return {
        "board": board.grid.tolist(),
        "current_player": int(board.current_player),
        "legal_moves": board.get_legal_moves(),
        "winner": int(board.check_win()) if board.check_win() else None,
        "human_player": int(human_player),
        "strength": current_strength,
        "simulations": _get_mcts(current_strength).num_simulations,
    }


def _search_result_json(result) -> dict:
    """Convert SearchResult tuple keys to "r,c" strings for JSON."""
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


@app.route("/api/config", methods=["GET"])
def api_config():
    """Return available strength presets (frontend populates selector)."""
    return jsonify({
        "strength_presets": {
            k: {"label": v["label"], "description": v["description"]}
            for k, v in STRENGTH_PRESETS.items()
        },
        "default_strength": "medium",
    })


@app.route("/api/new-game", methods=["POST"])
def new_game():
    global board, human_player, current_strength

    data = request.get_json(silent=True) or {}
    side = data.get("side", "black").lower()
    strength = data.get("strength", current_strength).lower()

    if strength not in STRENGTH_PRESETS:
        strength = "medium"

    human_player = Player.BLACK if side == "black" else Player.WHITE
    current_strength = strength
    mcts = _get_mcts(strength)

    board = Board()

    # If the human chose White, AI (Black) makes the opening move.
    if human_player == Player.WHITE:
        result = mcts.search_with_stats(board)
        if not result.visit_counts:
            ai_move = (7, 7)
        else:
            ai_move = max(result.visit_counts, key=result.visit_counts.get)
        board.make_move(*ai_move)
        return jsonify({
            **_board_state_json(),
            "last_move": list(ai_move),
            "search": _search_result_json(result),
            "ai_move": list(ai_move),
        })

    return jsonify({
        **_board_state_json(),
        "search": {},
        "ai_move": None,
    })


@app.route("/api/search", methods=["POST"])
def search():
    global board

    data = request.get_json()
    if not data or "move" not in data:
        return jsonify({"error": "missing move"}), 400

    r, c = data["move"]

    if board.grid[r, c] != 0:
        return jsonify({"error": "cell occupied"}), 400
    if board.check_win() is not None:
        return jsonify({"error": "game already over"}), 400
    if board.current_player != human_player:
        return jsonify({"error": "not your turn"}), 400

    board.make_move(r, c)

    if board.check_win() is not None:
        return jsonify({
            **_board_state_json(), "search": {}, "ai_move": None,
            "last_move": data["move"],
        })

    mcts = _get_mcts(current_strength)
    result = mcts.search_with_stats(board)
    ai_move = max(result.visit_counts, key=result.visit_counts.get)
    board.make_move(*ai_move)

    response = {
        **_board_state_json(),
        "last_move": list(ai_move),
        "search": _search_result_json(result),
        "ai_move": list(ai_move),
    }
    return jsonify(response)


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
