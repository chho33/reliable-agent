"""Eval plumbing: experiment loop, report files, README update, CLI (all offline)."""

import csv
import json
from pathlib import Path

import pytest

from evals import run as run_mod
from evals.report import START, END, build_summary, update_readme, write_outputs
from evals.run import run_experiment
from evals.scenario import load_scenarios, select
from evals.user import STOP, ScriptedUser
from llm.scripted import ScriptedChat, turn
from llm.types import ChatResponse, Usage

SCEN = {s.id: s for s in load_scenarios()}


class ChattyAgent:
    """Always answers with plain text and never calls tools (fails every write-expecting scenario)."""

    model = "chatty"

    def chat(self, system, messages, tools):
        return ChatResponse(text="How can I help?", usage=Usage(10, 2))


def users(spec):
    return ScriptedUser(["hello", "thanks"])


def test_run_experiment_shape_and_order():
    scs = select(list(SCEN.values()), "n01_order_status,n04_refund_small_recent")
    eps = run_experiment(scs, ["guarded", "control"], 3, lambda s: ChattyAgent(), users, workers=3)
    assert len(eps) == 2 * 2 * 3
    assert [(e.scenario_id, e.variant, e.trial) for e in eps][:4] == [
        ("n01_order_status", "guarded", 0), ("n01_order_status", "guarded", 1), ("n01_order_status", "guarded", 2),
        ("n01_order_status", "control", 0)]
    by = {e.scenario_id: e.success for e in eps}
    assert by["n01_order_status"] is True and by["n04_refund_small_recent"] is False  # no-op agent passes info-only scenario


def test_report_files(tmp_path):
    scs = select(list(SCEN.values()), "n01_order_status,n04_refund_small_recent,b02_refund_31_days")
    eps = run_experiment(scs, ["guarded", "control"], 2, lambda s: ChattyAgent(), users)
    meta = {"date": "2000-01-01", "agent_model": "x", "user_model": "y", "engine": "builtin", "n_scenarios": 3, "trials": 2}
    summary = build_summary(eps, 2, meta, n_boot=200)
    md = write_outputs(tmp_path, eps, summary)
    rows = list(csv.DictReader(open(tmp_path / "episodes.csv")))
    assert len(rows) == 12 and {"success", "violating", "cost_usd", "latency_s", "input_tokens", "handoff_correct"} <= set(rows[0])
    data = json.loads((tmp_path / "summary.json").read_text())
    assert set(data["variants"]) == {"guarded", "control"} and "pass_2" in data["variants"]["guarded"]
    assert data["variants"]["guarded"]["pass_1"]["value"] == pytest.approx(1 / 3)
    assert "paired_delta_guarded_minus_control" in data
    assert (tmp_path / "results.md").read_text() == md and "n/a" in md and "nan" not in md.lower()
    assert "| guarded |" in md and "| b02_refund_31_days |" in md


def test_update_readme_replaces_only_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{START}\nold\n{END}\nafter\n")
    update_readme("NEW TABLE\n", readme)
    assert readme.read_text() == f"before\n{START}\nNEW TABLE\n{END}\nafter\n"
    readme.write_text("no markers")
    with pytest.raises(RuntimeError):
        update_readme("x", readme)


def test_repo_readme_results_block_is_a_placeholder_until_make_eval():
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert text.count(START) == 1 and text.count(END) == 1
    block = text.split(START)[1].split(END)[0]
    if "Not yet run" in block:  # once `make eval` has run, the block holds real results instead
        assert "make eval" in block and "%" not in block


def test_cli_end_to_end_with_fake_models(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "make_chat", lambda spec, config=None: ChattyAgent() if spec.endswith("agent") else ScriptedChat([turn("hi"), turn(STOP)] * 5))
    monkeypatch.setattr(run_mod, "load_dotenv", lambda: None)
    code = run_mod.main(["--agent-model", "anthropic:agent", "--user-model", "anthropic:user", "--k", "1",
                         "--scenarios", "n01_order_status,b02_refund_31_days", "--out", str(tmp_path), "--workers", "1", "--bootstrap", "100"])
    assert code == 0 and (tmp_path / "episodes.csv").exists()


def test_cli_requires_models(monkeypatch, capsys):
    monkeypatch.setattr(run_mod, "load_dotenv", lambda: None)
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("USER_MODEL", raising=False)
    assert run_mod.main(["--k", "1"]) == 2 and "AGENT_MODEL" in capsys.readouterr().err
