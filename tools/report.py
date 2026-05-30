#!/usr/bin/env python3
"""Reporting tools for NeuralGomoku evaluation."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Optional
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path: sys.path.insert(0, str(_proj_root))
from selfplay.elo import EloTracker

def generate_elo_report(path):
    t=EloTracker(); t.load(path)
    sr=dict(sorted(t._ratings.items(),key=lambda x:-x[1]))
    return {"num_checkpoints":len(t._ratings),"num_matches":len(t.match_history),"ratings":sr,"top_5":list(sr.items())[:5],
        "history":[{"timestamp":time.strftime("%Y-%m-%d %H:%M",time.localtime(m.timestamp)),"challenger":m.model_a,"baseline":m.model_b,"score":m.score_a,"num_games":m.num_games,"delta":m.delta_a} for m in t.match_history[-20:]]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--elo-history",default=None); p.add_argument("--benchmark",default=None)
    p.add_argument("--regression",default=None); p.add_argument("--output","-o",default=None)
    args=p.parse_args(); report={}
    if args.elo_history and Path(args.elo_history).exists():
        er=generate_elo_report(args.elo_history)
        report.update(er)
        print(f"Elo: {er['num_matches']} matches, {er['num_checkpoints']} checkpoints")
        for name,rating in er.get("top_5",[]): print(f"  {rating:>7.1f}  {name}")
    if args.benchmark and Path(args.benchmark).exists():
        d=json.loads(Path(args.benchmark).read_text())
        report["benchmarks"]=[{"category":c.get("category",""),"pass_rate":c.get("passed",0)/max(c.get("total",1),1),"passed":c.get("passed",0),"total":c.get("total",0)} for c in d.get("categories",[])]
    if args.regression and Path(args.regression).exists():
        d=json.loads(Path(args.regression).read_text())
        report["regression"]=d
    if args.output: Path(args.output).write_text(json.dumps(report,indent=2))
    return 0

if __name__=="__main__": sys.exit(main())
