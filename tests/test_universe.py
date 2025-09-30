from __future__ import annotations

from src.tools.universe import save_universe, load_universe


def test_universe_save_load(tmp_path):
    # save to a temp-like name (the util uses project data dir but ensures file exists)
    name = "test_universe"
    path = save_universe(name, ["AAA", "BBB"])  # returns path under repo data/universes
    syms = load_universe(name)
    assert "AAA" in syms and "BBB" in syms