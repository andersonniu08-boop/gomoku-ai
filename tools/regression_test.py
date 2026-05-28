#!/usr/bin/env python3
"""Regression testing for Gomoku AI."""
from __future__ import annotations
import argparse, json, sys, tempfile, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import torch
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path: sys.path.insert(0, str(_proj_root))
from engine.board import Board
from neural.model import GomokuNet
from neural.wrapper import GomokuInferenceWrapper
from selfplay.mcts import MCTS

@dataclass
class RegressionResult:
    name: str; passed: bool; category: str; details: dict = field(default_factory=dict)

@dataclass
class RegressionReport:
    results: list = field(default_factory=list); baseline_name: str = "reference"; new_name: str = "current"
    def add(self, r): self.results.append(r)
    @property
    def passed(self): return sum(1 for r in self.results if r.passed)
    @property
    def failed(self): return sum(1 for r in self.results if not r.passed)
    @property
    def total(self): return len(self.results)
    def to_dict(self): return {"baseline":self.baseline_name,"new":self.new_name,"total":self.total,"passed":self.passed,"failed":self.failed,"pass_rate":self.passed/max(self.total,1),"results":[asdict(r) for r in self.results]}

def _make_wrapper(checkpoint=None):
    if checkpoint: return GomokuInferenceWrapper(Path(checkpoint), device="cpu")
    model = GomokuNet(board_size=15,in_channels=3,num_res_blocks=5,num_hidden_channels=64,use_se=False,use_attention=False)
    with tempfile.NamedTemporaryFile(suffix=".pt",delete=False) as f:
        torch.save(model.state_dict(),f); tmp_path = Path(f.name)
    w = GomokuInferenceWrapper(tmp_path,device="cpu",num_res_blocks=5,num_hidden_channels=64,use_se=False,use_attention=False)
    tmp_path.unlink(); return w

TACTICAL = [("win_open_four",[(7,2),(0,0),(7,3),(0,1),(7,4),(0,2),(7,5),(0,3)],{(7,1),(7,6)}),
    ("win_closed_four",[(7,2),(7,1),(7,3),(8,0),(7,4),(8,2),(7,5),(8,4)],{(7,6)}),
    ("win_split_xx_xx",[(7,2),(0,0),(7,3),(0,1),(7,5),(0,2),(7,6),(0,3)],{(7,4)}),
    ("win_split_xxx_x",[(7,2),(0,0),(7,3),(0,1),(7,4),(0,2),(7,6),(0,3)],{(7,5)}),
    ("block_open_four",[(2,2),(7,2),(4,4),(7,3),(6,6),(7,4),(8,8),(7,5)],{(7,1),(7,6)}),
    ("block_split_closed",[(10,0),(7,2),(12,3),(7,3),(10,6),(7,5),(12,9),(7,6)],{(7,4)}),
    ("block_contiguous_closed",[(7,1),(7,2),(8,0),(7,3),(8,2),(7,4),(8,4),(7,5)],{(7,6)})]

def check_tactical(wrapper, sims=10):
    results=[]; mcts=MCTS(wrapper,num_simulations=sims,threat_override=True)
    for n,s,e in TACTICAL:
        b=Board()
        for r,c in s: b.make_move(r,c)
        results.append(RegressionResult(name=f"tactical/{n}",passed=set(mcts.search(b).keys())==e,category="tactical",details={"expected":sorted(e),"actual":sorted(set(mcts.search(b).keys()))}))
    return results

def check_eval(wrapper):
    results=[]
    for name,setup in [("empty",[]),("one_stone",[(7,7)]),("four_stones",[(7,7),(8,7),(7,8),(8,8)])]:
        b=Board()
        for r,c in setup: b.make_move(r,c)
        mp,v=wrapper.evaluate(b)
        if abs(sum(p for _,p in mp)-1.0)>1e-4: results.append(RegressionResult(name=f"eval/{name}_norm",passed=False,category="evaluation",details={"total":sum(p for _,p in mp)}))
        if not(-1.05<=v<=1.05): results.append(RegressionResult(name=f"eval/{name}_value",passed=False,category="evaluation",details={"value":v}))
    if not results: results.append(RegressionResult(name="eval/all_checks",passed=True,category="evaluation",details={}))
    return results

def check_search(wrapper, sims=50):
    results=[]; mcts=MCTS(wrapper,num_simulations=sims,threat_override=True)
    for name,setup in [("empty",[]),("one_stone",[(7,7)]),("four_stones",[(7,7),(0,0),(8,8),(0,1)])]:
        b=Board()
        for r,c in setup: b.make_move(r,c)
        d=mcts.search(b); legal=set(b.get_legal_moves())
        for move in d:
            if move not in legal: results.append(RegressionResult(name=f"search/{name}_illegal",passed=False,category="search",details={"move":move}))
        if abs(sum(d.values())-1.0)>=1e-4 and d: results.append(RegressionResult(name=f"search/{name}_norm",passed=False,category="search",details={}))
    if not results: results.append(RegressionResult(name="search/all_checks",passed=True,category="search",details={}))
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--new",default=None); p.add_argument("--json",default=None)
    p.add_argument("--category",nargs="+",choices=["tactical","evaluation","search","all"],default=["all"])
    p.add_argument("--simulations",type=int,default=50)
    args=p.parse_args()
    cats=["tactical","evaluation","search"] if "all" in args.category else args.category
    wrapper=_make_wrapper(args.new)
    report=RegressionReport(new_name=args.new or "untrained_current")
    if "tactical" in cats:
        for r in check_tactical(wrapper,args.simulations): report.add(r)
    if "evaluation" in cats:
        for r in check_eval(wrapper): report.add(r)
    if "search" in cats:
        for r in check_search(wrapper,args.simulations): report.add(r)
    print(f"Regression Report: {report.passed}/{report.total} passed ({report.passed/max(report.total,1):.1%})")
    if report.failed>0: print(f"REGRESSIONS: {report.failed} failed")
    if args.json: Path(args.json).write_text(json.dumps(report.to_dict(),indent=2))
    return 0 if report.failed==0 else 1

if __name__=="__main__": sys.exit(main())
