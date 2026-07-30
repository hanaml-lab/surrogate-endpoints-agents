"""Every figure of the TENOR report, drawn from results.json (written by make_data.py)."""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "results.json")))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "grid.linewidth": 0.4, "grid.alpha": 0.35, "axes.grid": True, "lines.linewidth": 1.2, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
# Okabe-Ito
C = dict(tenor="#0072B2", wait="#7a7a7a", proxy="#D55E00", naive="#E69F00", parc="#009E73", truth="#000000",
         honest="#7a7a7a", hacked="#D55E00", unhack="#E69F00", presentation="#CC79A7", latent="#56B4E9")
LAB = dict(tenor="TENOR", wait="Wait for the outcome", proxy="Proxy only", naive="Index, no validation",
           parc="PPI, judge as predictor")
MARK = dict(honest="o", hacked="^", unhack="s", presentation="D", latent="P")
KIND = dict(honest="honest", hacked="hacked", unhack="removes a hack", presentation="flatters the judge",
            latent="unmeasured path")
DELTA = 2.0
ENV = {k: R["envs"][k]["name"] for k in R["envs"]}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# 1. Teaser: intervals over time for one honest improvement and one hacked change ------------------------------
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.55), sharey=False)
for ax, key, title in zip(axs, ("honest", "hacked"), ("(a) An honest improvement", "(b) A reward-hacked change")):
    T = R["teaser"][key]
    ts = np.array(T["ts"])
    show = ts <= 21.0
    for m, z, a in (("wait", 1, 0.16), ("proxy", 2, 0.30), ("tenor", 3, 0.20)):
        lo, hi = (100 * np.array(x) for x in T[m])
        informative = (hi - lo) < 150
        lo, hi = np.clip(lo, -30, 30), np.clip(hi, -30, 30)
        sel = show & informative
        ax.fill_between(ts[sel], lo[sel], hi[sel], color=C[m], alpha=a, lw=0, zorder=z)
        ax.plot(np.where(sel, ts, np.nan), np.where(sel, lo, np.nan), color=C[m], lw=0.9, zorder=z)
        ax.plot(np.where(sel, ts, np.nan), np.where(sel, hi, np.nan), color=C[m], lw=0.9, zorder=z)
    ax.axhline(100 * T["truth"], color=C["truth"], lw=0.9, ls="--", zorder=4)
    ax.axhline(-DELTA, color="#444444", lw=0.6, ls=":", zorder=4)
    ax.axhline(0, color="#444444", lw=0.5, zorder=4)
    notes = []
    for m, t_key, d_key in (("tenor", "t_tenor", "dec_tenor"), ("wait", "t_wait", "dec_wait"), ("proxy", "t_proxy", "dec_proxy")):
        t, d = T[t_key], T[d_key]
        ax.plot([t], [-28.5], marker="^", color=C[m], ms=4.5, zorder=6, clip_on=False)
        notes.append((C[m], f"{LAB[m]}: {'ships' if d == 1 else 'blocks'} at day {t:.1f}"))
    for i, (col, txt) in enumerate(notes):
        ax.text(20.6, 27.5 - 4.4 * i, txt, color=col, fontsize=6.1, ha="right", va="top", zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6))
    if T.get("t_flag"):
        ax.axvline(T["t_flag"], color=C["tenor"], lw=0.6, ls="--")
        ax.text(T["t_flag"] + 0.3, -12, "screen flags the change:\nTENOR waits for outcomes", fontsize=6.0, color=C["tenor"], va="center",
                zorder=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6))
    if T.get("t_lic") and key == "honest":
        ax.axvline(T["t_lic"], color=C["tenor"], lw=0.6, ls="--")
        ax.text(T["t_lic"] + 0.3, -12, "screen licenses transport", fontsize=6.0, color=C["tenor"], va="center",
                zorder=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6))
    ax.set_xlim(0, 21); ax.set_ylim(-30, 30)
    ax.set_xlabel("Days since release"); ax.set_title(title, loc="left")
axs[0].set_ylabel("Effect on outcome rate (points)")
handles = [Line2D([0], [0], color=C[m], lw=4, alpha=0.5) for m in ("tenor", "wait", "proxy")] + \
          [Line2D([0], [0], color="k", ls="--", lw=0.9), Line2D([0], [0], color="#444444", ls=":", lw=0.6),
           Line2D([0], [0], marker="^", color="w", markerfacecolor="#555555", ms=5)]
fig.legend(handles, ["TENOR", "Wait for the outcome", "Proxy only", "True effect", "Gate margin $-\\delta$", "Decision"],
           loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.12))
save(fig, "fig_teaser.pdf")

# 2. Coverage and width over time --------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.2))
days = np.arange(1, 43)
for m in ("wait", "proxy", "naive", "parc", "tenor"):
    ot = R["over_time"][m]
    axs[0].plot(days, ot["coverage"], color=C[m], lw=1.6 if m == "tenor" else 1.0, label=LAB[m])
    w = np.array(ot["width"])
    axs[1].plot(days, np.where(w >= 199.9, np.nan, w), color=C[m], lw=1.6 if m == "tenor" else 1.0, label=LAB[m])
axs[0].axhline(95, color="#444444", ls=":", lw=0.6)
axs[0].set_ylim(0, 102); axs[0].set_xlim(0, 42)
axs[0].set_xlabel("Days since release"); axs[0].set_ylabel("Covered at every day so far (%)")
axs[0].set_title("(a) Anytime coverage", loc="left")
axs[1].set_yscale("log"); axs[1].set_xlim(0, 42)
axs[1].set_xlabel("Days since release"); axs[1].set_ylabel("Median interval width (points)")
axs[1].set_title("(b) Width (blank: uninformative)", loc="left")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.17))
save(fig, "fig_coverage.pdf")

# 3. The paradox on the forty past changes ------------------------------------------------------------------------
H = R["history"]
kappa, gamma = 100 * R["cal"]["kappa"], R["cal"]["gamma"]
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.75))
for ax, xkey, xlab, title in ((axs[0], "proxy", "Effect on the proxy (points)", "(a) Proxy: tests pass and judge approves"),
                              (axs[1], "dS", "Effect on the index, $\\hat\\gamma\\,\\Delta_S$ (points)", "(b) Surrogate index, screened")):
    lim = 30
    ax.fill_between([0, lim], -lim, 0, color=C["hacked"], alpha=0.07, lw=0)
    ax.fill_between([-lim, 0], 0, lim, color=C["hacked"], alpha=0.07, lw=0)
    ax.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.5)
    if xkey == "dS":
        xs = np.array([-lim, lim])
        ax.fill_between(xs, xs - kappa, xs + kappa, color=C["tenor"], alpha=0.18, lw=0)
    for h in H:
        x = 100 * h[xkey] * (gamma if xkey == "dS" else 1.0)
        y = 100 * h["dY"]
        hollow = xkey == "dS" and not h["passed"]
        ax.scatter([x], [y], marker=MARK[h["kind"]], s=18, facecolor="white" if hollow else C[h["kind"]],
                   edgecolor=C[h["kind"]], lw=0.8, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(xlab); ax.set_title(title, loc="left")
    ax.text(lim * 0.95, -lim * 0.9, "proxy up,\noutcome down", ha="right", va="bottom", fontsize=6.2, color=C["hacked"])
    ax.text(-lim * 0.95, lim * 0.9, "proxy down,\noutcome up", ha="left", va="top", fontsize=6.2, color=C["hacked"])
axs[0].set_ylabel("Effect on the outcome (points)")
kinds = [k for k in ("honest", "hacked", "unhack", "presentation") if any(h["kind"] == k for h in H)]
handles = [Line2D([0], [0], marker=MARK[k], color="w", markerfacecolor=C[k], markeredgecolor=C[k], ms=5) for k in kinds]
handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#555555", ms=5),
            Line2D([0], [0], color=C["tenor"], lw=5, alpha=0.3)]
fig.legend(handles, [KIND[k] for k in kinds] + ["flagged by the screen", "margin $\\pm\\kappa$"], loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.11))
save(fig, "fig_paradox.pdf")

# 4. Variance ratio against the index's R^2 -----------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.35), sharey=True)
cmap = plt.get_cmap("viridis")
for ax, k in zip(axs, ("pr", "recon")):
    V = R["variance"][k]
    names = [n for n in V["r2"] if n != "const"]
    r2s = np.array([max(V["r2"][n], 0) for n in names])
    grid = np.linspace(0, 0.5, 50)
    for i, (t, rho) in enumerate(zip(V["times"], V["rho"])):
        col = cmap(0.1 + 0.8 * i / (len(V["times"]) - 1))
        ax.plot(grid, 1 - (1 - rho) * grid, color=col, lw=0.8, ls="--")
        ax.scatter(r2s, [V["ratio"][n][i] for n in names], color=col, s=12, zorder=3,
                   label=f"day {t:g} ($\\rho$ = {rho:.2f})")
    ax.set_xlim(0, 0.5); ax.set_ylim(0.4, 1.05)
    ax.set_xlabel("Outcome $R^2$ of the index"); ax.set_title(f"({'a' if k == 'pr' else 'b'}) {ENV[k]}", loc="left")
    ax.legend(loc="lower left", fontsize=6.0)
axs[0].set_ylabel("Variance relative to outcome only")
save(fig, "fig_variance.pdf")

# 5. Ablation of the surrogate's signals ---------------------------------------------------------------------------
SIG = dict(judge="judge score", tests="tests passed", ccv="completion prob.", comments="reviewer comments",
           rollback="rollbacks", perturb="perturbed re-checks")
order = list(SIG)
base = R["ablations"]["tenor"]
fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.1), sharey=True)
ys = np.arange(len(order) + 1)[::-1]
labels = ["none (TENOR)"] + [SIG[s] for s in order]
days_ = [base["median_days"]] + [R["signals"][s]["median_days"] for s in order]
kap = [100 * R["cal"]["kappa"]] + [R["signals"][s]["kappa"] for s in order]
wr = [base["wrong"]] + [R["signals"][s]["wrong"] for s in order]
cols = [C["tenor"]] + ["#8aa6c8"] * len(order)
for ax, vals, xl, title in ((axs[0], days_, "Median days to decision", "(a) Speed"),
                            (axs[1], kap, "Margin $\\kappa$ (points)", "(b) Margin"),
                            (axs[2], wr, "Wrong decisions (%)", "(c) Errors")):
    ax.barh(ys, vals, color=cols, height=0.62)
    for y, v in zip(ys, vals):
        ax.text(v + max(vals) * 0.02, y, f"{v:.1f}", va="center", fontsize=6.2)
    ax.set_xlabel(xl); ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.25)
axs[0].set_yticks(ys); axs[0].set_yticklabels(["removed: " + x if i else x for i, x in enumerate(labels)])
save(fig, "fig_signals.pdf")

# 6. Time to decision -------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.25))
for m in ("proxy", "naive", "tenor", "wait", "parc"):
    t = np.array(R["times"][m])
    axs[0].step(np.concatenate([[0], t]), np.concatenate([[0], np.arange(1, len(t) + 1) / len(t)]) * 100, where="post",
                color=C[m], lw=1.6 if m == "tenor" else 1.0, label=LAB[m])
axs[0].set_xscale("symlog", linthresh=1); axs[0].set_xlim(0, 48)
axs[0].set_xticks([0, 1, 2, 3, 5, 10, 14, 30, 48]); axs[0].set_xticklabels(["0", "1", "2", "3", "5", "10", "14", "30", "48"])
axs[0].set_xlabel("Days since release"); axs[0].set_ylabel("Releases decided (%)")
axs[0].set_title("(a) All environments", loc="left"); axs[0].legend(loc="upper left", fontsize=6.0)
styles = dict(pr="-", recon="--", repl=":", sim="-.")
for k in R["times_env"]:
    for m in ("tenor", "wait"):
        t = np.array(R["times_env"][k][m])
        axs[1].step(np.concatenate([[0], t]), np.concatenate([[0], np.arange(1, len(t) + 1) / len(t)]) * 100, where="post",
                    color=C[m], ls=styles[k], lw=1.1)
axs[1].set_xscale("symlog", linthresh=1); axs[1].set_xlim(0, 48)
axs[1].set_xticks([0, 1, 2, 3, 5, 10, 14, 30, 48]); axs[1].set_xticklabels(["0", "1", "2", "3", "5", "10", "14", "30", "48"])
axs[1].set_xlabel("Days since release"); axs[1].set_title("(b) By environment: TENOR (blue), waiting (grey)", loc="left")
handles = [Line2D([0], [0], color="#333333", ls=styles[k], lw=1.0) for k in styles]
axs[1].legend(handles, [ENV[k] for k in styles], loc="lower right", fontsize=6.0)
save(fig, "fig_times.pdf")

# 7. Appendix: calibration of the index ----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(6.5, 1.8), sharey=True)
for ax, k in zip(axs, R["envs"]):
    rel = [r for r in R["envs"][k]["rel"] if r[2] >= 50]
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.6, ls="--")
    ax.plot([r[0] for r in rel], [r[1] for r in rel], "-o", ms=2.5, color=C["tenor"], lw=1.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(f"{ENV[k]}\nECE {R['envs'][k]['ece']:.3f}", fontsize=7)
    ax.set_xlabel("Index $S(x)$")
axs[0].set_ylabel("Outcome rate")
save(fig, "fig_calibration.pdf")

# 8. Appendix: the margin against the size of the history --------------------------------------------------------
hs = R["history_size"]
ms = sorted(int(m) for m in hs)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.0))
kap = np.array([100 * hs[str(m)]["kappa"] for m in ms])
q = np.array([[100 * x for x in hs[str(m)]["kappa_q"]] for m in ms])
axs[0].plot(ms, kap, "-o", ms=3, color=C["tenor"])
axs[0].fill_between(ms, q[:, 0], q[:, 1], color=C["tenor"], alpha=0.2, lw=0)
axs[0].set_xlabel("Past changes used for validation"); axs[0].set_ylabel("Margin $\\kappa$ (points)")
axs[0].set_title("(a) The margin (band: 10th to 90th percentile)", loc="left")
axs[1].plot(ms, [100 * hs[str(m)]["coverage"] for m in ms], "-o", ms=3, color=C["tenor"], label="held-out changes covered")
axs[1].plot(ms, [100 * hs[str(m)]["nominal"] for m in ms], "--", color="#555555", lw=0.8, label="nominal $n/(n+1)$")
axs[1].set_xlabel("Past changes used for validation"); axs[1].set_ylabel("Residual within $\\kappa$ (%)")
axs[1].set_title("(b) What the margin covers", loc="left"); axs[1].legend(loc="lower right")
save(fig, "fig_history.pdf")
print("figures written")
