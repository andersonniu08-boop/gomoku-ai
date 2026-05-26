"""Flask server — loads model + MCTS, serves web UI API."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from engine.board import Board
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

    if board.grid[r, c] != 0:
        return jsonify({"error": "cell occupied"}), 400
    if board.check_win() is not None:
        return jsonify({"error": "game already over"}), 400

    board.make_move(r, c)

    if board.check_win() is not None:
        return jsonify({**_board_state_json(), "search": {}, "ai_move": None})

    result = mcts.search_with_stats(board)
    ai_move = max(result.visit_counts, key=result.visit_counts.get)
    board.make_move(*ai_move)

    response = {
        **_board_state_json(),
        "last_move": data["move"],
        "search": _search_result_json(result),
        "ai_move": list(ai_move),
    }
    return jsonify(response)


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
