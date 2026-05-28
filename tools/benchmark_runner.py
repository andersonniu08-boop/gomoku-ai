#!/usr/bin/env python3
"""Tactical benchmark runner."""
from __future__ import annotations
import argparse, json, sys, tempfile, time
from pathlib import Path
from typing import Optional
import torch
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path: sys.path.insert(0, str(_proj_root))
from engine.board import Board
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS

WIN_IN_1 = [("open_four_both_ends",[(7,2),(0,0),(7,3),(0,1),(7,4),(0,2),(7,5),(0,3)],{(7,1),(7,6)},""),
    ("contiguous_closed_four",[(7,2),(7,1),(7,3),(8,0),(7,4),(8,2),(7,5),(8,4)],{(7,6)},""),
    ("split_closed_four_xx_xx",[(7,2),(0,0),(7,3),(0,1),(7,5),(0,2),(7,6),(0,3)],{(7,4)},""),
    ("split_closed_four_xxx_x",[(7,2),(0,0),(7,3),(0,1),(7,4),(0,2),(7,6),(0,3)],{(7,5)},""),
    ("vertical_open_four",[(3,7),(0,0),(4,7),(0,1),(5,7),(0,2),(6,7),(0,3)],{(2,7),(7,7)},""),
    ("diagonal_open_four",[(3,3),(0,0),(4,4),(0,1),(5,5),(0,2),(6,6),(0,3)],{(2,2),(7,7)},""),
    ("anti_diagonal_open_four",[(3,6),(0,0),(4,5),(0,1),(5,4),(0,2),(6,3),(0,3)],{(2,7),(7,2)},"")]
FORCED_DEFENSE = [("block_open_four",[(2,2),(7,2),(4,4),(7,3),(6,6),(7,4),(8,8),(7,5)],{(7,1),(7,6)},""),
    ("block_contiguous_closed_four",[(7,1),(7,2),(8,0),(7,3),(8,2),(7,4),(8,4),(7,5)],{(7,6)},""),
    ("block_split_closed_four",[(10,0),(7,2),(12,3),(7,3),(10,6),(7,5),(12,9),(7,6)],{(7,4)},""),
    ("block_split_closed_four_xxx_x",[(10,0),(7,2),(12,3),(7,3),(10,6),(7,4),(12,9),(7,6)],{(7,5)},""),
    ("win_takes_priority",[(7,2),(10,0),(7,3),(10,1),(7,4),(10,2),(7,5),(10,3)],{(7,1),(7,6)},"")]
DOUBLE_THREAT = [("create_double_open_three",[(7,3),(13,0),(7,4),(13,2),(5,5),(13,4),(6,5),(13,6)],{(7,5)},"",False),
    ("open_four_plus_open_three",[(7,2),(1,0),(7,3),(1,2),(7,4),(1,4),(7,5),(1,6),(5,9),(1,8),(6,9),(1,10)],{(7,1),(7,6)},"",True)]
EDGE_CASES = [("near_edge_open_four",[(0,0),(7,0),(0,1),(7,1),(0,2),(7,2),(0,3),(7,3)],{(0,4)},"")]

def _make_wrapper(checkpoint=None):
    if checkpoint: return GomokuInferenceWrapper(Path(checkpoint), device="cpu")
    model = GomokuNet(board_size=15, in_channels=3, num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f); tmp_path = Path(f.name)
    w = GomokuInferenceWrapper(tmp_path, device="cpu", num_res_blocks=5, num_hidden_channels=64, use_se=False, use_attention=False)
    tmp_path.unlink(); return w

def run_scenario(wrapper, setup, expected, sims, exact=True):
    mcts = MCTS(wrapper, num_simulations=sims, threat_override=True)
    b = Board()
    for r,c in setup: b.make_move(r,c)
    t0 = time.perf_counter(); d = mcts.search(b); elapsed = time.perf_counter()-t0
    actual = set(d.keys()); passed = actual==expected if exact else expected.issubset(actual)
    return passed, {"expected": sorted(expected), "actual": sorted(actual)}, elapsed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", choices=["win-in-1","forced-defense","double-threat","edge-cases","all"], default="all")
    p.add_argument("--checkpoint", default=None); p.add_argument("--simulations", type=int, default=10); p.add_argument("--json", default=None)
    args = p.parse_args()
    cats = {"win-in-1":("Win-in-1",WIN_IN_1),"forced-defense":("Forced Defense",FORCED_DEFENSE),"double-threat":("Double Threat",DOUBLE_THREAT),"edge-cases":("Edge Cases",EDGE_CASES)}
    selected = list(cats.items()) if args.category=="all" else [(k,v) for k,v in cats.items() if k==args.category]
    wrapper = _make_wrapper(args.checkpoint)
    cat_results, total_passed, total_sc, total_time = [], 0, 0, 0.0
    for key,(name,scenarios) in selected:
        results, tt = [], 0.0
        for sc in scenarios:
            nm, setup, exp, desc = sc[:4]; exact = sc[4] if len(sc)>4 else True
            ok, det, elapsed = run_scenario(wrapper, setup, exp, args.simulations, exact)
            tt += elapsed; results.append({"name":nm,"passed":ok,"details":det,"time_sec":round(elapsed,4)})
        n = len(scenarios)
        cat_results.append({"category":name,"total":n,"passed":sum(1 for r in results if r["passed"]),"failed":n-sum(1 for r in results if r["passed"]),"avg_time_ms":round(tt/n*1000,2),"scenarios":results})
        total_passed += sum(1 for r in results if r["passed"]); total_sc += n; total_time += tt
    print(json.dumps({"overall":{"total":total_sc,"passed":total_passed,"pass_rate":total_passed/max(total_sc,1)},"categories":cat_results},indent=2))
    if args.json: Path(args.json).write_text(json.dumps({"overall":{"total":total_sc,"passed":total_passed,"pass_rate":total_passed/max(total_sc,1)},"categories":cat_results},indent=2))
    return 0 if total_passed==total_sc else 1

if __name__ == "__main__": sys.exit(main())
