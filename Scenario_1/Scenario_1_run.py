"""
Scenario 1 — Synthetic numerical benchmarks (paper Section 2.2.1)

Paper spec:
  - 24 problems = 6 functions × 4 dims (p ∈ {20, 100, 200, 500})
  - Functions: Sphere, PA-1, PA-10, CPA, Ackley, Rastrigin
    * PA(x) = ||x - x*||  with  x*=1  or  x*=10·1
    * CPA   = min over 5 points on a circle (radius 2, centre 1) in the
              (x₁, x₂)-plane (other dims fixed at 1)
  - n_rep = 30, n_evals = 10 000
  - Init: uniform [-1, 1]^p for ALL EAs (including CMA-ES x₀)
  - 9 EAs with paper-exact hyperparameters:
        CMA-ES   : vanilla defaults (Hansen)
        DE       : DE/rand/1, NP=15, F=0.5, CR=0.8
        PSO      : n_pop=100, w=0.8, φ_p=φ_g=1.5
        ES-σ     : basic (μ,λ) = (30, 30), μ_parents=10, Gaussian noise
                   of variance σ²·I around mean of parents
                   σ ∈ {0.02, 0.25, 0.5}
        GA-σ     : n_pop=100, n_tour=5, p_xo=0.8
                   crossover: x₁ + α(x₂-x₁) + ε, α∼U[0,1] scalar
                   mutation : x + ε; ε∼N(0, σ²·I)
                   σ ∈ {0.02, 0.25, 0.5}

CLI:
  python Scenario_1_run.py                     # full run
  python Scenario_1_run.py --quick             # smoke test
  python Scenario_1_run.py --problems ea.p.s.sphere-20 --n_rep 2
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from multiprocessing import Pool, cpu_count, get_context
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG (paper defaults)
# ============================================================
N_EVALS_DEFAULT = 10_000
N_REP_DEFAULT   = 30
N_CORES_DEFAULT = max(1, cpu_count() - 1)
DIMS_DEFAULT    = [20, 100, 200, 500]

SOLVER_NAME_MAP = {
    "CMA-ES":  "cmaEs",
    "DE":      "differentialEvolution",
    "PSO":     "pso",
    "ES-0.02": "es-0.02",
    "ES-0.25": "es-0.25",
    "ES-0.5":  "es-0.5",
    "GA-0.02": "ga-0.02",
    "GA-0.25": "ga-0.25",
    "GA-0.5":  "ga-0.5",
}


# ============================================================
# BENCHMARK FUNCTIONS (paper definitions only)
# ============================================================
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


def point_aiming(x: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(x - target))


def circular_point_aiming(x: np.ndarray, targets: List[np.ndarray]) -> float:
    return float(min(np.linalg.norm(x - t) for t in targets))


def ackley(x: np.ndarray) -> float:
    d = len(x)
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / d))
        - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / d)
        + 20.0 + np.e
    )


def rastrigin(x: np.ndarray) -> float:
    d = len(x)
    return float(10.0 * d + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def _cpa_targets(dim: int, seed: int = 42) -> List[np.ndarray]:
    """5 points on circle (radius=2, centre=1) in the (x1,x2) plane,
    remaining dims equal 1. Fixed per `dim` via a deterministic seed (paper)."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, 5)
    ts = []
    for a in angles:
        t = np.ones(dim)
        t[0] += 2.0 * np.cos(a)
        if dim > 1:
            t[1] += 2.0 * np.sin(a)
        ts.append(t)
    return ts


# ============================================================
# RECORD (same schema as Scenario 2/3)
# ============================================================
def _record(iteration, total_evals, births, best_fit, t0, pop, fitness,
            n_firsts=1, n_lasts=None):
    n_pop   = len(pop)
    n_lasts = n_lasts if n_lasts is not None else n_pop
    rounded = np.round(pop, 6)
    geno_uni = len(np.unique(rounded, axis=0)) / n_pop
    fit_uni  = len(np.unique(np.round(fitness, 8))) / n_pop
    return {
        "iterations":  iteration,
        "evals":       total_evals,
        "births":      births,
        "elapsed":     round(time.time() - t0, 4),
        "all_size":    n_pop,
        "firsts_size": n_firsts,
        "lasts_size":  n_lasts,
        "geno_uni":    geno_uni,
        "sol_uni":     geno_uni,
        "fit_uni":     fit_uni,
        "best_fitness": float(best_fit),
    }


# ============================================================
# SOLVERS — paper-exact (same as Scenario 2/3)
# ============================================================
def run_cma_es(fitness_fn, dim, seed, n_evals: int):
    """Vanilla CMA-ES with paper init: x0 ~ U(-1,1)^p, default sigma."""
    import cma
    t0  = time.time()
    rng = np.random.default_rng(seed)
    x0  = rng.uniform(-1.0, 1.0, dim)
    es  = cma.CMAEvolutionStrategy(
        x0, 0.5, {"seed": int(seed), "verbose": -9, "maxfevals": int(n_evals)}
    )
    records, total_evals, births, iteration = [], 0, 0, 0
    while not es.stop() and total_evals < n_evals:
        solutions = es.ask()
        fits = np.array([fitness_fn(s) for s in solutions])
        es.tell(solutions, fits.tolist())
        n = len(solutions)
        total_evals += n; births += n; iteration += 1
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   float(np.min(fits)), t0,
                                   np.array(solutions), fits, n_lasts=n))
    return records


def run_de(fitness_fn, dim, seed, n_evals: int):
    """DE/rand/1: NP=15, F=0.5, CR=0.8 (paper)."""
    t0, rng = time.time(), np.random.default_rng(seed)
    NP, F, CR = 15, 0.5, 0.8
    pop     = rng.uniform(-1.0, 1.0, (NP, dim))
    fitness = np.array([fitness_fn(ind) for ind in pop])
    total_evals, iteration = NP, 0
    records = [_record(0, NP, NP, np.min(fitness), t0, pop, fitness)]
    while total_evals < n_evals:
        iteration += 1
        for i in range(NP):
            choices    = [j for j in range(NP) if j != i]
            r1, r2, r3 = rng.choice(choices, 3, replace=False)
            mutant     = pop[r1] + F * (pop[r2] - pop[r3])
            mask       = rng.random(dim) < CR
            if not np.any(mask): mask[rng.integers(dim)] = True
            trial      = np.where(mask, mutant, pop[i])
            f_trial    = fitness_fn(trial); total_evals += 1
            if f_trial <= fitness[i]:
                pop[i], fitness[i] = trial, f_trial
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, total_evals,
                                   np.min(fitness), t0, pop, fitness))
    return records


def run_pso(fitness_fn, dim, seed, n_evals: int):
    """PSO (Clerc 2012 w/out constriction): n_pop=100, w=0.8, φp=φg=1.5 (paper)."""
    t0, rng = time.time(), np.random.default_rng(seed)
    n_pop, w, phi_p, phi_g = 100, 0.8, 1.5, 1.5
    pos = rng.uniform(-1.0, 1.0, (n_pop, dim))
    vel = rng.uniform(-0.5, 0.5, (n_pop, dim))
    fit = np.array([fitness_fn(p) for p in pos])
    pbest_pos, pbest_fit = pos.copy(), fit.copy()
    gbest_idx = np.argmin(fit)
    gbest_fit = fit[gbest_idx]; gbest_pos = pos[gbest_idx].copy()
    total_evals, iteration = n_pop, 0
    records = [_record(0, n_pop, n_pop, gbest_fit, t0, pos, fit)]
    while total_evals < n_evals:
        iteration += 1
        r1, r2 = rng.random((n_pop, dim)), rng.random((n_pop, dim))
        vel = w * vel + phi_p * r1 * (pbest_pos - pos) + phi_g * r2 * (gbest_pos - pos)
        pos = pos + vel
        fit = np.array([fitness_fn(p) for p in pos])
        total_evals += n_pop
        improved = fit < pbest_fit
        pbest_pos[improved] = pos[improved]; pbest_fit[improved] = fit[improved]
        idx = np.argmin(pbest_fit)
        if pbest_fit[idx] < gbest_fit:
            gbest_fit = pbest_fit[idx]; gbest_pos = pbest_pos[idx].copy()
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, total_evals,
                                   gbest_fit, t0, pos, fit))
    return records


def run_es(fitness_fn, dim, seed, sigma: float, n_evals: int):
    """Basic ES: n_pop=30, n_parents=⌊0.33·n_pop⌋=10, σ identity, NO self-adapt (paper)."""
    t0, rng = time.time(), np.random.default_rng(seed)
    n_pop     = 30
    n_parents = int(np.floor(0.33 * n_pop))
    pop = rng.uniform(-1.0, 1.0, (n_pop, dim))
    fit = np.array([fitness_fn(ind) for ind in pop])
    total_evals, births, iteration = n_pop, n_pop, 0
    records = [_record(0, n_pop, n_pop, np.min(fit), t0, pop, fit)]
    std = float(np.sqrt(sigma))
    while total_evals < n_evals:
        iteration += 1
        order   = np.argsort(fit)
        mu_vec  = pop[order[:n_parents]].mean(axis=0)
        new_pop = np.empty_like(pop)
        new_pop[0]  = pop[order[0]]                            # elitism (best kept)
        new_pop[1:] = mu_vec + rng.normal(0.0, std, size=(n_pop - 1, dim))
        pop = new_pop
        fit = np.array([fitness_fn(ind) for ind in pop])
        total_evals += n_pop; births += n_pop
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   np.min(fit), t0, pop, fit, n_lasts=n_pop))
    return records


def run_ga(fitness_fn, dim, seed, sigma: float, n_evals: int):
    """GA: n_pop=100, n_tour=5, p_xo=0.8; geometric segment crossover + Gaussian mut (paper)."""
    t0, rng = time.time(), np.random.default_rng(seed)
    n_pop, n_tour, p_xo = 100, 5, 0.8
    pop = rng.uniform(-1.0, 1.0, (n_pop, dim))
    fit = np.array([fitness_fn(ind) for ind in pop])
    total_evals, births, iteration = n_pop, n_pop, 0
    records = [_record(0, n_pop, n_pop, np.min(fit), t0, pop, fit)]
    std = float(np.sqrt(sigma))

    def tour_select():
        parts = rng.choice(n_pop, n_tour, replace=False)
        return pop[parts[np.argmin(fit[parts])]]

    while total_evals < n_evals:
        iteration += 1
        offspring = np.empty_like(pop)
        for k in range(n_pop):
            if rng.random() < p_xo:
                x1, x2 = tour_select(), tour_select()
                alpha  = rng.random()                          # scalar (paper)
                child  = x1 + alpha * (x2 - x1) + rng.normal(0.0, std, size=dim)
            else:
                child  = tour_select() + rng.normal(0.0, std, size=dim)
            offspring[k] = child
        off_fit = np.array([fitness_fn(o) for o in offspring])
        births += n_pop; total_evals += n_pop
        merged_pop = np.vstack([pop, offspring])
        merged_fit = np.concatenate([fit, off_fit])
        best_idx   = np.argsort(merged_fit)[:n_pop]
        pop, fit   = merged_pop[best_idx], merged_fit[best_idx]
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   np.min(fit), t0, pop, fit, n_lasts=n_pop))
    return records


# ============================================================
# WORKER
# ============================================================
def _make_fitness(fn_tag: str, dim: int):
    if fn_tag == "sphere":
        return sphere
    if fn_tag == "ackley":
        return ackley
    if fn_tag == "rastrigin":
        return rastrigin
    if fn_tag == "pa1":
        target = np.ones(dim)
        return lambda x, t=target: point_aiming(x, t)
    if fn_tag == "pa10":
        target = np.full(dim, 10.0)
        return lambda x, t=target: point_aiming(x, t)
    if fn_tag == "cpa":
        ts = _cpa_targets(dim)
        return lambda x, tts=ts: circular_point_aiming(x, tts)
    raise ValueError(f"Unknown fn_tag: {fn_tag}")


def _worker(args):
    prob_key, jname, fn_tag, dim, obj, solver_key, sigma, seed, n_evals = args
    fn = _make_fitness(fn_tag, dim)

    if solver_key == "CMA-ES":
        records = run_cma_es(fn, dim, seed, n_evals)
    elif solver_key == "DE":
        records = run_de(fn, dim, seed, n_evals)
    elif solver_key == "PSO":
        records = run_pso(fn, dim, seed, n_evals)
    elif solver_key.startswith("ES-"):
        records = run_es(fn, dim, seed, float(sigma), n_evals)
    elif solver_key.startswith("GA-"):
        records = run_ga(fn, dim, seed, float(sigma), n_evals)
    else:
        records = []

    j_solv = SOLVER_NAME_MAP[solver_key]
    return [
        {
            **r,
            "seed":          seed + 1,
            "problem":       jname,
            "solver_sigma":  j_solv,
            "objective":     obj,
            "best→fitness":  r["best_fitness"],
            "genotype_size": dim,
        }
        for r in records
    ]


# ============================================================
# MAIN
# ============================================================
def build_problems(dims):
    """Return list of (prob_key, tag, jname) — exactly 6 functions per dim (paper)."""
    out = []
    for d in dims:
        out.extend([
            (f"Sphere-{d}",    "sphere",    f"ea.p.s.sphere-{d}"),
            (f"PA-1-{d}",      "pa1",       f"ea.p.s.pointAiming-{d}-1"),
            (f"PA-10-{d}",     "pa10",      f"ea.p.s.pointAiming-{d}-10"),
            (f"CPA-{d}",       "cpa",       f"ea.p.s.circularPointAiming-{d}"),
            (f"Ackley-{d}",    "ackley",    f"ea.p.s.ackley-{d}"),
            (f"Rastrigin-{d}", "rastrigin", f"ea.p.s.rastrigin-{d}"),
        ])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scenario 1 — synthetic benchmarks")
    p.add_argument("--n_evals", type=int, default=N_EVALS_DEFAULT)
    p.add_argument("--n_rep",   type=int, default=N_REP_DEFAULT)
    p.add_argument("--cores",   type=int, default=N_CORES_DEFAULT)
    p.add_argument("--dims",    type=int, nargs="*", default=None,
                   help=f"Dims to run (default {DIMS_DEFAULT})")
    p.add_argument("--problems", nargs="*", default=None,
                   help="Subset of problem names, e.g. ea.p.s.sphere-20")
    p.add_argument("--quick",   action="store_true",
                   help="Smoke test: 2 reps, 500 evals, dims=[20], 1 problem.")
    return p.parse_args()


def main():
    args     = parse_args()
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dims = args.dims if args.dims else DIMS_DEFAULT
    if args.quick:
        args.n_rep   = min(args.n_rep, 2)
        args.n_evals = min(args.n_evals, 500)
        args.cores   = 1
        dims         = [20]

    all_problems = build_problems(dims)
    if args.problems:
        wanted = set(args.problems)
        all_problems = [p for p in all_problems if p[2] in wanted]
        if not all_problems:
            raise ValueError("No matching problems. Example: ea.p.s.sphere-20")

    tasks = []
    for prob_key, tag, jname in all_problems:
        # Dim = first numeric token in the problem name
        # ea.p.s.sphere-20 → 20 ; ea.p.s.pointAiming-20-1 → 20
        dim_candidates = [int(tok) for tok in jname.split("-") if tok.isdigit()]
        dim = dim_candidates[0]
        for sk in SOLVER_NAME_MAP.keys():
            sig = float(sk.split("-")[1]) if sk.startswith("ES-") or sk.startswith("GA-") else None
            for s in range(args.n_rep):
                tasks.append((prob_key, jname, tag, dim, "minimize",
                              sk, sig, s, args.n_evals))

    all_rows: list = []
    if args.cores <= 1:
        for t in tqdm(tasks, total=len(tasks)):
            all_rows.extend(_worker(t))
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=args.cores) as pool:
            for res in tqdm(
                pool.imap_unordered(_worker, tasks, chunksize=4),
                total=len(tasks),
            ):
                all_rows.extend(res)

    out_csv = results_dir / "Scenario_1_Novel_Final.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, sep=";", index=False)
    print(f"Done! Saved to {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
