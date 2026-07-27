import time

from tenor.demo import main


def test_toy_runs_end_to_end_quickly(capsys):
    t0 = time.time()
    assert main([]) == 0
    out = capsys.readouterr().out
    assert time.time() - t0 < 120
    assert "wrong decisions: TENOR" in out
    lines = [ln for ln in out.splitlines() if ln[:2].isdigit()]
    hacked = next(ln for ln in lines if " hacked " in ln)
    assert "WRONG" in hacked.split("ok")[-1] or hacked.rstrip().endswith("WRONG")  # the proxy ships the hack
    assert "ship" not in hacked.split()[3]                                          # TENOR does not
