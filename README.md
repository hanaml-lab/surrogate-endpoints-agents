# TENOR: valid evaluation of long-running agents from early surrogates and late outcomes

[![ci](https://github.com/hanaml-lab/surrogate-endpoints-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/hanaml-lab/surrogate-endpoints-agents/actions/workflows/ci.yml)

Reference implementation of *TENOR: Valid Evaluation of Long-Running Agents from Early Surrogates and Late Outcomes*
(Agarwal, hanaML technical report, 2026). Report: https://hanaml.com/research/papers/surrogate-endpoints-agents

An agent's pull request succeeds when it has been merged and survived a fortnight without revert; a reconciliation
succeeds when it passes the month-end audit. Release decisions cannot wait that long, so they rest on proxies such as
a judge's score or tests passed at submission, which an optimised agent can move without moving the outcome. TENOR
decides whether to ship a new agent version at any hour after release, with an interval for its effect on the
outcome rate that is valid at every check:

* **Two clocks.** The release's effect is the change in a surrogate index (learned within hours from six early
  signals) plus a residual effect (learned from outcomes as they mature, with a censoring-weighted rectifier).
* **Transport, validated.** Before outcomes mature, the residual is bounded by a margin learned from the lab's past
  agent changes (a split-conformal bound), with a slope that corrects the index's attenuation.
* **A screen for the paradox.** Transport is licensed only while the signals the agent controls (judge, visible
  tests, completion verifier) and those it does not (reviewer comments, perturbed re-checks) agree about the change.
  A reward hack, which moves the first without the second, is flagged and waits for its outcomes.
* **Anytime-valid.** Betting and asymptotic confidence sequences, so the gate may be checked every hour.

## Install

```bash
pip install -e ".[dev]"      # numpy and pyyaml at run time; pytest and ruff for development
pip install -e ".[paper]"    # + matplotlib, to redraw the report's figures
```

## Quickstart (CPU, a few seconds)

```bash
python -m tenor.demo          # or: python scripts/run_toy.py --config configs/toy.yaml
pytest -q                     # 21 tests
python examples/decide_on_your_runs.py
```

## Toy run (real output)

`python -m tenor.demo` fits TENOR on thirty simulated past changes of a pull-request-like environment (two of them
reward-hacked, one flattering the judge) and replays seven releases hour by hour. Waiting for the outcome and deciding
on a proxy run on the same simulated runs. This is the output of the reference implementation on this machine:

```text
TENOR toy: pull requests (toy), 30 past changes, 1500 runs/day/version, landmark 14 days
slope gamma 1.109: the index explains 90% of an effect
margin kappa 2.45 pts, screen tolerance tau 3.49 pts; 25 past changes pass the screen
past changes the screen flags: 3, 4, 9, 18, 19

release  kind       effect   TENOR                 wait for the outcome  proxy only
31       honest      -2.2    escalate              block    20.5d  ok    block     3.3d  ok
32       honest      +5.3    ship      4.2d  ok    ship     15.2d  ok    ship      0.5d  ok
33       honest      +3.7    ship     27.7d  ok    ship     15.2d  ok    ship      0.8d  ok
34       hacked      -4.4    block    26.8d  ok    block    14.9d  ok    ship      0.1d  WRONG
35       unhack      +1.9    escalate              ship     18.5d  ok    block     0.5d  WRONG
36       honest     +21.2    ship      2.9d  ok    ship     14.2d  ok    ship      0.1d  ok
37       latent      -5.6    ship      4.8d  WRONG block    14.5d  ok    ship      0.5d  WRONG

wrong decisions: TENOR 1, waiting 0, proxy 3 (of 7); 3.5s on CPU
```

TENOR ships the two clear improvements on days 4.2 and 2.9, about eleven days before waiting would. It never ships
the hacked change, which the proxy ships within hours; once the screen stops it trusting the index it waits for
outcomes and blocks on day 26.8, later than waiting (day 14.9), because its outcome-only interval spends less error
than the waiting baseline's. It escalates two changes it cannot settle by day 28, and it is wrong once: on the change
whose effect runs through a path none of the six signals measures. That failure is the method's stated limit.

## Using TENOR on your own runs

`examples/decide_on_your_runs.py` shows the whole path: fit the index and the maturation model on past runs, validate
on past changes, then run the gate on a release.

```python
from tenor import SurrogateIndex, MaturationModel, PastChange, validate, ReleaseGate, Streams
index = SurrogateIndex.fit(past_runs)                 # P(outcome | six signals) on matured runs
v = validate(past_changes)                            # slope, margin kappa, screen tolerance tau
trace = ReleaseGate(v).run(streams, check_times)      # ship / block / wait, with the interval at every check
```

## Layout

```
src/tenor/   confseq.py (betting and asymptotic confidence sequences), index.py (surrogate index),
             censoring.py (maturation model, rectifier), validate.py (slope, margin, screen),
             gate.py (Algorithm 1 and the baselines), simulate.py (delayed-outcome simulator),
             replay.py (fit on history, replay a release), demo.py (the CPU toy)
configs/     toy.yaml (the toy), paper.yaml (the report's four environments)
scripts/     run_toy.py, reproduce_paper.sh
examples/    decide_on_your_runs.py
paper/       make_data.py, figures.py: the seeded simulation behind every number, table and figure of the report;
             check_results.py, results.sha256: the check of its numbers; toy_run.md
tests/       pytest suite
```

## Results (from the report)

These numbers come from the report's seeded simulation and are illustrative. Sixty releases in four delayed-outcome
environments (pull requests, reconciliation, research replication, a business simulation), five seeds each; the
gate ships once a change is shown to cost less than two points and blocks once it is shown to cost anything.

| Method | Wrong decisions (%) | Regressions shipped (%) | Median days to decision | Anytime coverage (%) |
|---|---|---|---|---|
| Wait for the outcome | 0.0 | 0.0 | 14.4 | 97.0 |
| Proxy only (tests pass and judge approves) | 22.0 | 41.0 | 0.3 | 37.7 |
| Surrogate index, no validation | 17.3 | 37.0 | 2.2 | 50.7 |
| PPI with the judge as predictor (PARC-style) | 0.0 | 0.0 | 14.9 | 99.3 |
| **TENOR** | 4.0 | 7.0 | 3.2 | 94.7 |

`bash scripts/reproduce_paper.sh` regenerates the tables and figures that the report's seeded simulation produces
(the 300 release replays take about 30 CPU-minutes, spread over every core). On every push, CI reruns
`paper/make_data.py` with pinned numerics (numpy 2.3.5, one OpenBLAS thread) and checks the tables and results it
writes against `paper/results.sha256` (`python paper/check_results.py`).

## Citation

```bibtex
@techreport{agarwal2026tenor,
  title       = {{TENOR}: Valid Evaluation of Long-Running Agents from Early Surrogates and Late Outcomes},
  author      = {Agarwal, Manav},
  institution = {hanaML},
  year        = {2026},
  url         = {https://github.com/hanaml-lab/surrogate-endpoints-agents}
}
```

## License

Apache-2.0. See `LICENSE`.
