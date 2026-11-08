"""WitDraw vs. CmdStan on three of the Stan examples.

Needs cmdstanpy (with CmdStan installed), arviz, numpy and pandas, and sbt for
the WitDraw side. Run from the repository root:

    python examples/benchmark/bench.py all --out <dir> [--chains 4]

Both engines run the same number of chains, from the same starting points
(drawn as Stan draws its own: uniform in (-2, 2) in unconstrained space):
CmdStan as parallel processes, WitDraw side by side in one compiled loop
(vmap). Times are wall-clock times of all chains together; for CmdStan, that
of its slowest chain. Each run is repeated, and the median time reported. ESS
and R-hat are computed over all chains, of the first repeat.

Steps (each can be run on its own):
  prep     simulate the data sets, written as Stan JSON data files, and draw
           the starting points of the chains
  stan     compile and run each model with CmdStan, in two configurations:
             stan-default  diagonal mass matrix, adapted step size (Stan's default)
             stan-unit     identity mass matrix, adapted step size
  witdraw  run each model with WitDraw's NUTS, on CPU and GPU, in two
           configurations:
             witdraw-<device>        identity mass matrix and no adaptation, at
                                     the median step size Stan adapted for
                                     stan-unit, each chain starting where
                                     stan-unit's ended its warmup: a
                                     comparison of the sampling phase only
             witdraw-adapt-<device>  NUTS.adaptChains: diagonal mass matrix and
                                     step size adapted per chain (BlackJax
                                     window adaptation)
  report   effective sample sizes and R-hat (ArviZ, the same estimators for
           every run), timings, and posterior means side by side
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MODELS = ["linreg", "mixture", "hmm"]
N_WARMUP = 1000
N_SAMPLES = 1000
N_CHAINS = 4
REPEATS = 3


# -- data ---------------------------------------------------------------------


def simulate_linreg(rng):
    n, k = 1000, 5
    x = rng.normal(size=(n, k))
    beta = np.array([1.5, -2.0, 0.5, 0.0, 3.0])
    y = 1.0 + x @ beta + rng.normal(scale=0.5, size=n)
    return {"N": n, "K": k, "x": x.tolist(), "y": y.tolist()}


def simulate_mixture(rng):
    n = 500
    theta = np.array([0.5, 0.3, 0.2])
    mu = np.array([-4.0, 0.0, 4.0])
    sigma = np.array([0.7, 1.0, 0.5])
    z = rng.choice(3, size=n, p=theta)
    y = rng.normal(mu[z], sigma[z])
    return {"K": 3, "N": n, "y": y.tolist()}


def simulate_hmm(rng):
    n = 200
    mu = np.array([-2.0, 0.0, 2.0])
    sigma = 0.5
    rho = np.array([0.8, 0.1, 0.1])
    gamma = np.full((3, 3), 0.05) + np.eye(3) * 0.85
    z = np.empty(n, dtype=int)
    z[0] = rng.choice(3, p=rho)
    for t in range(1, n):
        z[t] = rng.choice(3, p=gamma[z[t - 1]])
    y = rng.normal(mu[z], sigma)
    return {"N": n, "y": y.tolist()}


# -- starting points ----------------------------------------------------------
# Stan's transforms from unconstrained values u, each uniform in (-2, 2).


def positive(u):
    return np.exp(u)


def ordered(u):
    return np.cumsum(np.concatenate([u[:1], np.exp(u[1:])]))


def simplex(u):
    """Stan's stick-breaking transform: K-1 values to a K-simplex."""
    k = len(u) + 1
    x, rest = [], 1.0
    for i, ui in enumerate(u):
        z = 1 / (1 + np.exp(-(ui - np.log(k - 1 - i))))
        x.append(rest * z)
        rest -= x[-1]
    return np.array(x + [rest])


def init_linreg(data, u):
    k = data["K"]
    return {"alpha": u(1)[0], "beta": u(k), "sigma": positive(u(1))[0]}


def init_mixture(data, u):
    k = data["K"]
    return {"theta": simplex(u(k - 1)), "mu": ordered(u(k)), "sigma": positive(u(k))}


def init_hmm(data, u):
    return {
        "gamma_arr": [simplex(u(2)) for _ in range(3)],
        "rho": simplex(u(2)),
        "mu": ordered(u(3)),
        "sigma": positive(u(1))[0],
    }


def to_json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return [to_json(v) for v in x]
    return float(x)


def inits_file(out, m, chain, kind=""):
    return out / "inits" / f"{m}{kind}_{chain}.json"


def values_at(names, columns, row):
    """The values of the variables `names` in one draw `row` of CmdStan's
    `columns`, e.g. `gamma_arr[2,3]`, as nested lists for Stan's JSON format."""
    values = {}
    for name in names:
        if name in columns:
            values[name] = float(row[columns.index(name)])
            continue
        entries = [
            (tuple(int(i) - 1 for i in c[len(name) + 1 : -1].split(",")), row[j])
            for j, c in enumerate(columns)
            if c.startswith(name + "[")
        ]
        shape = tuple(max(idx[d] for idx, _ in entries) + 1 for d in range(len(entries[0][0])))
        array = np.zeros(shape)
        for idx, v in entries:
            array[idx] = v
        values[name] = array.tolist()
    return values


def save_warmup_ends(out, m, fit, chains):
    """Where each chain of `fit` ended its warmup, as starting points."""
    names = list(json.loads(inits_file(out, m, 0).read_text()))
    columns = list(fit.column_names)
    last = fit.draws(inc_warmup=True)[N_WARMUP - 1]  # (chain, column)
    for c in range(chains):
        values = values_at(names, columns, last[c])
        inits_file(out, m, c, "_unit_end").write_text(json.dumps(values))


def prep(out, chains):
    rng = np.random.default_rng(2026)
    sims = {"linreg": simulate_linreg, "mixture": simulate_mixture, "hmm": simulate_hmm}
    inits = {"linreg": init_linreg, "mixture": init_mixture, "hmm": init_hmm}
    (out / "inits").mkdir(exist_ok=True)
    for m in MODELS:
        data = sims[m](rng)
        (out / f"{m}.json").write_text(json.dumps(data))
        for c in range(chains):
            init = inits[m](data, lambda n: rng.uniform(-2, 2, size=n))
            inits_file(out, m, c).write_text(json.dumps({k: to_json(v) for k, v in init.items()}))
    print(f"data and starting points of {chains} chains written to {out}")


# -- stan ---------------------------------------------------------------------


def stan(out, chains):
    import cmdstanpy

    results = {}
    for m in MODELS:
        t0 = time.perf_counter()
        model = cmdstanpy.CmdStanModel(stan_file=HERE / "stan" / f"{m}.stan", force_compile=True)
        compile_s = time.perf_counter() - t0
        results[m] = {"compileSeconds": compile_s}
        for config, metric in [("stan-default", "diag_e"), ("stan-unit", "unit_e")]:
            runs = []
            for rep in range(REPEATS):
                fit = model.sample(
                    data=str(out / f"{m}.json"),
                    chains=chains,
                    parallel_chains=chains,
                    inits=[str(inits_file(out, m, c)) for c in range(chains)],
                    iter_warmup=N_WARMUP,
                    iter_sampling=N_SAMPLES,
                    metric=metric,
                    save_warmup=True,
                    seed=1 + rep,
                    show_progress=False,
                    output_dir=str(out / "stan_csv" / f"{m}_{config}_{rep}"),
                )
                # the chains run in parallel: the wall-clock time is that of
                # the slowest one
                times = fit.time
                runs.append(
                    {
                        "wallSeconds": max(t["warmup"] + t["sampling"] for t in times),
                        "samplingWallSeconds": max(t["sampling"] for t in times),
                        "chainSeconds": [t["warmup"] + t["sampling"] for t in times],
                        "stepSize": float(np.median(fit.step_size)),
                        "stepSizes": [float(x) for x in fit.step_size],
                    }
                )
                if rep == 0:
                    if config == "stan-unit":
                        save_warmup_ends(out, m, fit, chains)
                    draws = fit.draws_pd(inc_warmup=True)
                    is_warmup = draws["iter__"] <= N_WARMUP
                    draws[~is_warmup].to_csv(out / f"{config}_{m}_draws.csv", index=False)
                    # leapfrog steps of all chains, of warmup and of sampling
                    results[m].setdefault(config, {})["leapfrogSteps"] = {
                        "warmup": draws["n_leapfrog__"][is_warmup].astype(int).tolist(),
                        "sampling": draws["n_leapfrog__"][~is_warmup].astype(int).tolist(),
                    }
                    results[m][config]["divergences"] = int(
                        draws["divergent__"][~is_warmup].sum()
                    )
            results[m][config]["runs"] = runs
            print(m, config, runs)
    results["nChains"] = chains
    (out / "stan_results.json").write_text(json.dumps(results, indent=2))


# -- witdraw ------------------------------------------------------------------


def witdraw(out, devices, configs, chains):
    stan_results = json.loads((out / "stan_results.json").read_text())
    for device in devices:
        cmds = []
        for config, m in [(c, m) for c in configs for m in MODELS]:
            if config == "adapt":
                step, dev_out = "adapt", out / f"witdraw-adapt-{device}"
                inits, warmup = out / "inits" / m, N_WARMUP
            else:
                step = stan_results[m]["stan-unit"]["runs"][0]["stepSize"]
                dev_out = out / f"witdraw-{device}"
                # from where stan-unit ended its warmup, so no warmup of its own
                inits, warmup = out / "inits" / f"{m}_unit_end", 0
            cmds.append(
                f"examples/runMain witdraw.examples.stan.stanBenchmark {m} "
                f"{out / (m + '.json')} {step} {dev_out} {warmup} {N_SAMPLES} {chains} "
                f"{inits} {REPEATS}"
            )
        env = dict(os.environ)
        if device == "cpu":
            env["JAX_PLATFORMS"] = "cpu"
        else:
            env.pop("JAX_PLATFORMS", None)
        print(f"running WitDraw on {device}")
        subprocess.run(["sbt", "-batch", *cmds], cwd=REPO, env=env, check=True)


# -- report -------------------------------------------------------------------


def diagnostics(draws, columns):
    """Bulk and tail ESS, and R-hat, of every column, over all chains."""
    per_chain = [g for _, g in draws.groupby("chain__", sort=True)]
    n = min(len(g) for g in per_chain)
    chains = {c: np.stack([g[c].to_numpy()[:n] for g in per_chain]) for c in columns}  # (chain, draw)
    bulk = {c: float(az.ess(x, method="bulk")) for c, x in chains.items()}
    # tail ESS as in Stan: the smaller of the 5% and 95% quantile ESS
    tail = {
        c: min(float(az.ess(x, method="quantile", prob=p)) for p in (0.05, 0.95))
        for c, x in chains.items()
    }
    # rank-normalised split R-hat; not defined for one chain
    rhat = {c: float(az.rhat(x)) if len(per_chain) > 1 else np.nan for c, x in chains.items()}
    return pd.Series(bulk), pd.Series(tail), pd.Series(rhat)


def report(out, devices, configs):
    stan_results = json.loads((out / "stan_results.json").read_text())
    n_chains = stan_results.get("nChains", 1)
    rows, means = [], {}
    for m in MODELS:
        columns = None
        prefixes = {"fixed": "witdraw", "adapt": "witdraw-adapt"}
        engines = [f"{prefixes[c]}-{d}" for c in configs for d in devices]
        engines += ["stan-unit", "stan-default"]
        for engine in engines:
            if engine.startswith("witdraw"):
                info_file = out / engine / f"witdraw_{m}_info.json"
                if not info_file.exists():
                    continue
                info = json.loads(info_file.read_text())
                draws = pd.read_csv(out / engine / f"witdraw_{m}_draws.csv")
                columns = [c for c in draws.columns if c != "chain__"]
                if info.get("nChains", 1) != n_chains:
                    print(f"warning: {engine} {m} ran {info.get('nChains', 1)} chains, Stan {n_chains}")
                overhead = info["setupSeconds"] + info["compileSeconds"]
                totals = [w + s for w, s in zip(info["warmupRuns"], info["samplingRuns"])]
                run_s = float(np.median(totals))
                run_spread = f"{min(totals):.2f}-{max(totals):.2f}"
                if not info["adapted"]:
                    # the warmup was stan-unit's: only the sampling phase is timed
                    run_s, run_spread = np.nan, ""
                # the timed sampling loop of all chains, and the integration
                # steps of its draws
                loop_s = float(np.median(info["samplingRuns"]))
                leapfrog = info["leapfrogSteps"]
                divergences = info["divergences"]
                step = info["stepSize"]
            else:
                r = stan_results[m][engine]
                draws = pd.read_csv(out / f"{engine}_{m}_draws.csv")[["chain__", *columns]]
                overhead = stan_results[m]["compileSeconds"]
                totals = [x["wallSeconds"] for x in r["runs"]]
                run_s = float(np.median(totals))
                run_spread = f"{min(totals):.2f}-{max(totals):.2f}"
                # the sampling phase of the slowest chain, and the integration
                # steps of the sampling draws of all chains
                loop_s = float(np.median([x["samplingWallSeconds"] for x in r["runs"]]))
                leapfrog = r["leapfrogSteps"]["sampling"]
                divergences = r["divergences"]
                step = r["runs"][0]["stepSize"]
            bulk, tail, rhat = diagnostics(draws, columns)
            grads = int(np.sum(leapfrog))  # in the sampling phase, all chains
            rows.append(
                {
                    "model": m,
                    "engine": engine,
                    "step size": round(step, 4),
                    "compile/JIT s": round(overhead, 2),
                    "warmup+sampling s": round(run_s, 2),
                    "sampling s": round(loop_s, 2),
                    f"range ({REPEATS} runs)": run_spread,
                    "sampling leapfrogs": grads,
                    # wall-clock time per gradient, over all chains
                    "µs / gradient": round(1e6 * loop_s / grads, 1),
                    "divergences": divergences,
                    "min bulk ESS": round(bulk.min()),
                    "min tail ESS": round(tail.min()),
                    "max R-hat": round(rhat.max(), 3),
                    "min bulk ESS / s": round(bulk.min() / run_s, 1),
                    "min bulk ESS / sampling s": round(bulk.min() / loop_s, 1),
                }
            )
            means[(m, engine)] = draws[columns].mean()
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.to_string(index=False))
    table.to_csv(out / "summary.csv", index=False)

    print("\nposterior means")
    for m in MODELS:
        df = pd.DataFrame({e: s for (mm, e), s in means.items() if mm == m})
        print(f"\n{m}\n{df.round(3).to_string()}")
    pd.DataFrame(
        [{"model": m, "engine": e, **s.to_dict()} for (m, e), s in means.items()]
    ).to_csv(out / "means.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["prep", "stan", "witdraw", "report", "all"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--devices", nargs="+", default=["cpu", "gpu"])
    p.add_argument("--configs", nargs="+", choices=["fixed", "adapt"], default=["fixed", "adapt"])
    p.add_argument("--chains", type=int, default=N_CHAINS)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.step in ("prep", "all"):
        prep(out, a.chains)
    if a.step in ("stan", "all"):
        stan(out, a.chains)
    if a.step in ("witdraw", "all"):
        witdraw(out, a.devices, a.configs, a.chains)
    if a.step in ("report", "all"):
        report(out, a.devices, a.configs)


if __name__ == "__main__":
    main()
