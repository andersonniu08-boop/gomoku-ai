"""Tests for ui.server — verify side selection, turn handling, strength, and game flow."""

import pytest

from ui.server import app, STRENGTH_PRESETS


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _new_game(client, side="black", strength="fast"):
    """Helper: start a new game with the given side and strength."""
    return client.post("/api/new-game", json={"side": side, "strength": strength})


def _search(client, move, **kw):
    """Helper: make a move via search endpoint."""
    return client.post("/api/search", json={"move": move, **kw})


# ─── New game with side selection ───


def test_new_game_black(client):
    """Side=black: empty board, human=Black, no AI move."""
    resp = _new_game(client, "black")
    data = resp.get_json()
    assert data["human_player"] == 1
    assert data["current_player"] == 1
    assert data["ai_move"] is None
    stones = sum(1 for row in data["board"] for cell in row if cell != 0)
    assert stones == 0


def test_new_game_white(client):
    """Side=white: AI opens, human=White, one stone on board."""
    resp = _new_game(client, "white")
    data = resp.get_json()
    assert data["human_player"] == -1
    assert data["current_player"] == -1
    assert data["ai_move"] is not None
    stones = sum(1 for row in data["board"] for cell in row if cell != 0)
    assert stones == 1
    assert data["last_move"] == data["ai_move"]


def test_new_game_default_side(client):
    """Default side (no param) is Black for backward compatibility."""
    resp = client.post("/api/new-game", json={})
    data = resp.get_json()
    assert data["human_player"] == 1
    assert data["ai_move"] is None


# ─── Human as Black (first) ───


def test_black_turn_ordering(client):
    """Black side: human moves, AI responds, back to human."""
    _new_game(client, "black")

    data = _search(client, [7, 7]).get_json()
    assert data["human_player"] == 1
    assert data["ai_move"] is not None
    assert data["current_player"] == 1

    data = _search(client, [7, 8]).get_json()
    assert data["human_player"] == 1
    assert data["ai_move"] is not None
    assert data["current_player"] == 1


def test_black_stone_counts(client):
    """Black side: after each round stones increase by 2."""
    _new_game(client, "black")

    data = _search(client, [7, 7]).get_json()
    stones = sum(1 for row in data["board"] for cell in row if cell != 0)
    assert stones == 2

    data = _search(client, [7, 8]).get_json()
    stones = sum(1 for row in data["board"] for cell in row if cell != 0)
    assert stones == 4


# ─── Human as White (second) ───


def test_white_ai_opens_first(client):
    """White side: AI (Black) places first stone."""
    data = _new_game(client, "white").get_json()
    blacks = sum(1 for row in data["board"] for cell in row if cell == 1)
    assert blacks == 1


def test_white_turn_ordering(client):
    """White side: AI opens -> human moves -> AI responds -> human."""
    _new_game(client, "white")

    data = _search(client, [7, 6]).get_json()
    assert data["human_player"] == -1
    assert data["ai_move"] is not None
    assert data["current_player"] == -1

    data = _search(client, [6, 7]).get_json()
    assert data["human_player"] == -1
    assert data["ai_move"] is not None
    assert data["current_player"] == -1


def test_white_stone_counts(client):
    """White side: starts with 1, +2 per round."""
    data = _new_game(client, "white").get_json()
    stones = sum(1 for row in data["board"] for cell in row if cell != 0)
    assert stones == 1

    data = _search(client, [7, 6]).get_json()
    stones = sum(1 for row in data["board"] for cell in row if cell != 0)
    assert stones == 3

    data = _search(client, [6, 7]).get_json()
    stones = sum(1 for row in data["board"] for cell in row if cell != 0)
    assert stones == 5


# ─── Game resets ───


def test_new_game_resets_board(client):
    """Starting a new game clears all stones."""
    _new_game(client, "white")
    _search(client, [7, 6])

    data = _new_game(client, "black").get_json()
    stones = sum(1 for row in data["board"] for cell in row if cell != 0)
    assert stones == 0


def test_reset_preserves_side(client):
    """Reset with same side preserves human_player."""
    _new_game(client, "white")
    data = _search(client, [7, 6]).get_json()
    assert data["human_player"] == -1

    data = _new_game(client, "white").get_json()
    assert data["human_player"] == -1
    assert data["ai_move"] is not None


# ─── Strength presets ───


def test_strength_presets_exist(client):
    """All three strength presets are available and have distinct simulation counts."""
    sims = set()
    for name in ("fast", "medium", "strong"):
        assert name in STRENGTH_PRESETS, f"missing preset: {name}"
        sims.add(STRENGTH_PRESETS[name]["num_simulations"])
    assert len(sims) == 3, "presets must have distinct sim counts"


def test_strength_param_reflected_in_response(client):
    """The strength parameter is returned in the API response."""
    data = _new_game(client, "black", "fast").get_json()
    assert data["strength"] == "fast"
    assert data["simulations"] == STRENGTH_PRESETS["fast"]["num_simulations"]

    data = _new_game(client, "black", "strong").get_json()
    assert data["strength"] == "strong"
    assert data["simulations"] == STRENGTH_PRESETS["strong"]["num_simulations"]


def test_strength_retains_last_setting(client):
    """When strength is omitted, the previous setting persists."""
    _new_game(client, "black", "strong")
    # No strength param — should retain "strong" from above.
    resp = client.post("/api/new-game", json={"side": "black"})
    data = resp.get_json()
    # After a strong game, omit param → stays strong (current_strength global).
    assert data["strength"] == "strong"
    assert data["simulations"] == STRENGTH_PRESETS["strong"]["num_simulations"]


def test_config_endpoint(client):
    """The /api/config endpoint returns available presets."""
    resp = client.get("/api/config")
    data = resp.get_json()
    assert "strength_presets" in data
    assert "default_strength" in data
    for name in ("fast", "medium", "strong"):
        assert name in data["strength_presets"]
        assert "label" in data["strength_presets"][name]
        assert "description" in data["strength_presets"][name]


def test_strength_persists_across_search(client):
    """The strength setting persists into search responses."""
    _new_game(client, "black", "medium")
    data = _search(client, [7, 7]).get_json()
    assert data["strength"] == "medium"
    assert data["simulations"] == STRENGTH_PRESETS["medium"]["num_simulations"]


def test_cached_mcts_reused(client):
    """Calling new-game with same strength reuses cached MCTS instance."""
    from ui.server import _get_mcts

    m1 = _get_mcts("fast")
    m2 = _get_mcts("fast")
    assert m1 is m2, "MCTS instance should be cached"

    m3 = _get_mcts("strong")
    assert m3 is not m1, "different presets = different instances"


# ─── Error handling ───


def test_occupied_cell_rejected(client):
    """Moving on an occupied cell returns 400."""
    _new_game(client, "black")
    _search(client, [7, 7])
    resp = _search(client, [7, 7])
    assert resp.status_code == 400


def test_missing_move_rejected(client):
    """Request without move returns 400."""
    _new_game(client, "black")
    resp = client.post("/api/search", json={})
    assert resp.status_code == 400
