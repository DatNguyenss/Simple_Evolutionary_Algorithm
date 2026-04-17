"""
Scenario 3 — ANN-based 2D Navigation (paper Section 2.2.3)

Robot: differential-drive, circular, radius=5cm, max speed=10cm/s
Arena: 1m x 1m, 3 variants (Small / Large / Maze)
Sensors: m proximity sensors (frontal half) + 2 (distance+angle to target)
ANN: tanh, input=(m+2), hidden=3(m+2), output=2
Fitness: mean distance robot–target over 60s simulation (minimize)
Problems: 3 arenas × 3 sensor configs (m∈{3,5,9}) = 9
p ∈ {122, 212, 464}  (as reported by paper)

Performance notes:
  - Simulation is pure-Python by necessity (sequential time steps).
  - --quick mode uses 1 sim seed and reduced sim duration for fast testing.
  - For full run use --n_evals 10000 --n_rep 30 --cores N.
"""

import argparse
import math
import time
import warnings
from dataclasses import dataclass
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG (paper defaults)
# ============================================================
N_EVALS_DEFAULT  = 10_000
N_REP_DEFAULT    = 30
N_CORES_DEFAULT  = max(1, cpu_count() - 1)
N_SIM_SEEDS_FULL = 3   # avg over 3 random inits (reduces stochastic noise)
N_SIM_SEEDS_QUICK = 1  # for smoke tests

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
# RECORD (same schema as Scenario 1/2)
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
# SOLVERS (exact paper spec — same as Scenario_2/new.py)
# ============================================================
def run_cma_es(fitness_fn, dim, seed, n_evals: int):
    import cma
    t0  = time.time()
    rng = np.random.default_rng(seed)
    x0  = rng.uniform(-1, 1, dim)
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
    t0, rng = time.time(), np.random.default_rng(seed)
    NP, F, CR = 15, 0.5, 0.8
    pop     = rng.uniform(-1, 1, (NP, dim))
    fitness = np.array([fitness_fn(ind) for ind in pop])
    total_evals, iteration = NP, 0
    records = [_record(0, NP, NP, np.min(fitness), t0, pop, fitness)]
    while total_evals < n_evals:
        iteration += 1
        for i in range(NP):
            choices      = [j for j in range(NP) if j != i]
            r1, r2, r3   = rng.choice(choices, 3, replace=False)
            mutant       = pop[r1] + F * (pop[r2] - pop[r3])
            mask         = rng.random(dim) < CR
            if not np.any(mask): mask[rng.integers(dim)] = True
            trial        = np.where(mask, mutant, pop[i])
            f_trial      = fitness_fn(trial); total_evals += 1
            if f_trial <= fitness[i]:
                pop[i], fitness[i] = trial, f_trial
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, total_evals,
                                   np.min(fitness), t0, pop, fitness))
    return records


def run_pso(fitness_fn, dim, seed, n_evals: int):
    t0, rng = time.time(), np.random.default_rng(seed)
    n_pop, w, phi_p, phi_g = 100, 0.8, 1.5, 1.5
    pos = rng.uniform(-1, 1, (n_pop, dim))
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
    t0, rng = time.time(), np.random.default_rng(seed)
    n_pop    = 30
    n_parents = int(np.floor(0.33 * n_pop))
    pop      = rng.uniform(-1, 1, (n_pop, dim))
    fit      = np.array([fitness_fn(ind) for ind in pop])
    total_evals, births, iteration = n_pop, n_pop, 0
    records  = [_record(0, n_pop, n_pop, np.min(fit), t0, pop, fit)]
    std      = float(np.sqrt(sigma))
    while total_evals < n_evals:
        iteration += 1
        order   = np.argsort(fit)
        mu_vec  = pop[order[:n_parents]].mean(axis=0)
        new_pop = np.empty_like(pop)
        new_pop[0]  = pop[order[0]]
        new_pop[1:] = mu_vec + rng.normal(0.0, std, size=(n_pop - 1, dim))
        pop  = new_pop
        fit  = np.array([fitness_fn(ind) for ind in pop])
        total_evals += n_pop; births += n_pop
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   np.min(fit), t0, pop, fit, n_lasts=n_pop))
    return records


def run_ga(fitness_fn, dim, seed, sigma: float, n_evals: int):
    t0, rng = time.time(), np.random.default_rng(seed)
    n_pop, n_tour, p_xo = 100, 5, 0.8
    pop = rng.uniform(-1, 1, (n_pop, dim))
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
                x1, x2  = tour_select(), tour_select()
                alpha    = rng.random()
                child    = x1 + alpha * (x2 - x1) + rng.normal(0.0, std, size=dim)
            else:
                child    = tour_select() + rng.normal(0.0, std, size=dim)
            offspring[k] = child
        off_fit       = np.array([fitness_fn(o) for o in offspring])
        births       += n_pop; total_evals += n_pop
        merged_pop    = np.vstack([pop, offspring])
        merged_fit    = np.concatenate([fit, off_fit])
        best_idx      = np.argsort(merged_fit)[:n_pop]
        pop, fit      = merged_pop[best_idx], merged_fit[best_idx]
        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   np.min(fit), t0, pop, fit, n_lasts=n_pop))
    return records


# ============================================================
# 2D NAVIGATION SIMULATION (paper Section 2.2.3)
# ============================================================
# Arena dimensions and physics constants (paper)
ARENA_W   = 1.0    # m
ARENA_H   = 1.0    # m
R_ROBOT   = 0.05   # m (5 cm)
V_MAX     = 0.10   # m/s (10 cm/s)
SENSOR_R  = 1.0    # m proximity range
DT          = 0.1   # s time step
T_SIM       = 60.0  # s total duration (paper)
T_SIM_QUICK = 10.0  # s for quick/smoke test mode
TARGET      = np.array([0.50, 0.15])

# Arena barriers: list of line segments [(x1,y1,x2,y2), ...]
ARENAS = {
    "Small": [
        (0.40, 0.30, 0.60, 0.30),   # one short barrier in front of target
    ],
    "Large": [
        (0.20, 0.30, 0.80, 0.30),   # one long barrier
    ],
    "Maze": [
        (0.20, 0.40, 0.60, 0.40),   # first barrier
        (0.40, 0.20, 0.80, 0.20),   # second barrier — forms simple maze
    ],
}


def _seg_dist(px: float, py: float, ax: float, ay: float,
              bx: float, by: float) -> float:
    """Minimum distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    len2   = dx * dx + dy * dy
    if len2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _ray_seg_dist(ox: float, oy: float, angle: float,
                  ax: float, ay: float, bx: float, by: float,
                  max_r: float) -> float:
    """Distance along ray from (ox,oy) in direction `angle` to segment, or max_r."""
    dx, dy   = math.cos(angle), math.sin(angle)
    ex, ey   = bx - ax, by - ay
    denom    = dx * ey - dy * ex
    if abs(denom) < 1e-12:
        return max_r
    t = ((ax - ox) * ey - (ay - oy) * ex) / denom
    u = ((ax - ox) * dy - (ay - oy) * dx) / denom
    if t >= 0 and 0 <= u <= 1:
        return min(t, max_r)
    return max_r


def _proximity_reading(ox: float, oy: float, angle: float,
                       barriers: list, max_r: float) -> float:
    """Returns 1 when nothing in range, 0 on direct contact, interpolated otherwise."""
    d = max_r
    for seg in barriers:
        d = min(d, _ray_seg_dist(ox, oy, angle, *seg, max_r))
    # wall boundaries (4 walls)
    for wall in [
        (0.0, 0.0, ARENA_W, 0.0),
        (ARENA_W, 0.0, ARENA_W, ARENA_H),
        (ARENA_W, ARENA_H, 0.0, ARENA_H),
        (0.0, ARENA_H, 0.0, 0.0),
    ]:
        d = min(d, _ray_seg_dist(ox, oy, angle, *wall, max_r))
    return d / max_r          # 1 = nothing in range, 0 = contact


def _ann_forward(theta: np.ndarray, inputs: np.ndarray,
                 n_in: int, n_hid: int, n_out: int) -> np.ndarray:
    """Forward pass: tanh hidden + tanh output."""
    idx = 0
    W1  = theta[idx: idx + n_in * n_hid].reshape(n_in, n_hid); idx += n_in * n_hid
    b1  = theta[idx: idx + n_hid];                              idx += n_hid
    W2  = theta[idx: idx + n_hid * n_out].reshape(n_hid, n_out); idx += n_hid * n_out
    b2  = theta[idx: idx + n_out]
    h   = np.tanh(inputs @ W1 + b1)
    out = np.tanh(h @ W2 + b2)
    return out


def _theta_dim_nav(m: int) -> int:
    """Number of ANN parameters for navigation problem with m proximity sensors."""
    n_in  = m + 2
    n_hid = 3 * n_in
    n_out = 2
    return n_in * n_hid + n_hid + n_hid * n_out + n_out


def simulate_navigation(theta: np.ndarray, m: int, arena_name: str,
                        rng_sim: np.random.Generator,
                        t_sim: float = T_SIM) -> float:
    """
    Run one simulation episode and return mean distance to target.

    Args:
        theta       : ANN weight vector
        m           : number of proximity sensors
        arena_name  : 'Small' | 'Large' | 'Maze'
        rng_sim     : seeded RNG for stochastic init
        t_sim       : simulation duration in seconds (default 60 s)
    Returns:
        fitness (float) : mean Euclidean distance robot–target (minimize)
    """
    barriers = ARENAS[arena_name]
    n_in     = m + 2
    n_hid    = 3 * n_in
    n_out    = 2

    # Pre-unpack ANN weights once (avoid per-step reshape overhead)
    idx = 0
    W1 = theta[idx: idx + n_in * n_hid].reshape(n_in, n_hid); idx += n_in * n_hid
    b1 = theta[idx: idx + n_hid];                              idx += n_hid
    W2 = theta[idx: idx + n_hid * n_out].reshape(n_hid, n_out); idx += n_hid * n_out
    b2 = theta[idx: idx + n_out]

    # Stochastic initialisation (paper spec)
    x       = rng_sim.uniform(0.45, 0.55)
    y       = rng_sim.uniform(0.80, 0.85)
    heading = rng_sim.uniform(0, 2 * math.pi)

    steps  = int(t_sim / DT)
    total_dist = 0.0

    # Proximity sensor angles: m sensors evenly distributed on frontal half-arc
    sensor_angles_local = np.linspace(-math.pi / 2, math.pi / 2, m)

    inputs = np.empty(n_in)

    for _ in range(steps):
        # proximity sensors
        for j, local_a in enumerate(sensor_angles_local):
            inputs[j] = _proximity_reading(x, y, heading + local_a, barriers, SENSOR_R)

        # Distance/angle to target
        dx_t   = TARGET[0] - x
        dy_t   = TARGET[1] - y
        dist_t = math.hypot(dx_t, dy_t)
        inputs[m]     = min(dist_t, SENSOR_R) / SENSOR_R
        inputs[m + 1] = math.sin(math.atan2(dy_t, dx_t) - heading)

        # ANN forward (inlined)
        h   = np.tanh(inputs @ W1 + b1)
        out = np.tanh(h @ W2 + b2)

        # Differential drive: tanh ∈ [-1,1] → velocity ∈ [0, V_MAX]
        v_l = ((out[0] + 1.0) * 0.5) * V_MAX
        v_r = ((out[1] + 1.0) * 0.5) * V_MAX
        v     = (v_l + v_r) * 0.5
        omega = (v_r - v_l) / (2.0 * R_ROBOT)

        heading += omega * DT
        x = max(R_ROBOT, min(ARENA_W - R_ROBOT, x + v * math.cos(heading) * DT))
        y = max(R_ROBOT, min(ARENA_H - R_ROBOT, y + v * math.sin(heading) * DT))

        total_dist += math.hypot(x - TARGET[0], y - TARGET[1])

    return total_dist / steps


# ============================================================
# PROBLEM SPEC
# ============================================================
@dataclass(frozen=True)
class ProblemSpec:
    arena: str      # 'Small' | 'Large' | 'Maze'
    m: int          # number of proximity sensors: 3, 5, 9
    problem_name: str  # e.g. 'ea.p.n.Small-3'


def _p_for_spec(spec: ProblemSpec) -> int:
    return _theta_dim_nav(spec.m)


def build_problem_specs() -> List[ProblemSpec]:
    specs = []
    for arena in ["Small", "Large", "Maze"]:
        for m in [3, 5, 9]:
            specs.append(ProblemSpec(
                arena=arena,
                m=m,
                problem_name=f"ea.p.n.{arena}-{m}",
            ))
    return specs


# ============================================================
# WORKER
# ============================================================
def _worker(args):
    # n_sim and t_sim are passed explicitly to avoid relying on mutable globals
    # in multiprocessing spawn workers (which re-import the module fresh).
    spec, solver_key, sigma, seed, n_evals, n_sim, t_sim = args

    dim   = _p_for_spec(spec)
    arena = spec.arena
    m_val = spec.m

    sim_rngs_base = seed * 100_000  # deterministic per EA-seed

    def fitness_fn(theta: np.ndarray) -> float:
        total = 0.0
        for k in range(n_sim):
            rng_sim = np.random.default_rng(sim_rngs_base + k)
            total  += simulate_navigation(theta, m_val, arena, rng_sim, t_sim)
        return total / n_sim

    if solver_key == "CMA-ES":
        records = run_cma_es(fitness_fn, dim, seed, n_evals)
    elif solver_key == "DE":
        records = run_de(fitness_fn, dim, seed, n_evals)
    elif solver_key == "PSO":
        records = run_pso(fitness_fn, dim, seed, n_evals)
    elif solver_key.startswith("ES-"):
        records = run_es(fitness_fn, dim, seed, float(sigma), n_evals)
    elif solver_key.startswith("GA-"):
        records = run_ga(fitness_fn, dim, seed, float(sigma), n_evals)
    else:
        records = []

    j_solv = SOLVER_NAME_MAP[solver_key]
    return [
        {
            **r,
            "seed":          seed + 1,
            "problem":       spec.problem_name,
            "solver_sigma":  j_solv,
            "objective":     "minimize",
            "best→fitness":  r["best_fitness"],
            "genotype_size": dim,
        }
        for r in records
    ]


# ============================================================
# MAIN
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scenario 3 — 2D Navigation")
    p.add_argument("--n_evals", type=int, default=N_EVALS_DEFAULT)
    p.add_argument("--n_rep",   type=int, default=N_REP_DEFAULT)
    p.add_argument("--cores",   type=int, default=N_CORES_DEFAULT)
    p.add_argument("--quick",   action="store_true",
                   help="Smoke test: 2 reps, 500 evals, 1 problem.")
    p.add_argument("--problems", nargs="*", default=None,
                   help="Subset of problem names, e.g. ea.p.n.Small-3")
    return p.parse_args()


def main():
    args     = parse_args()
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.n_rep   = min(args.n_rep, 2)
        args.n_evals = min(args.n_evals, 300)
        args.cores   = 1
        n_sim = N_SIM_SEEDS_QUICK
        t_sim = T_SIM_QUICK
    else:
        n_sim = N_SIM_SEEDS_FULL
        t_sim = T_SIM

    specs = build_problem_specs()
    if args.problems:
        wanted = set(args.problems)
        specs  = [s for s in specs if s.problem_name in wanted]
        if not specs:
            raise ValueError("No matching problems. Example: ea.p.n.Small-3")

    # Pass n_sim and t_sim explicitly in each task tuple
    tasks = []
    for spec in specs:
        for sk in SOLVER_NAME_MAP:
            sig = float(sk.split("-")[1]) if sk.startswith("ES-") or sk.startswith("GA-") else None
            for s in range(args.n_rep):
                tasks.append((spec, sk, sig, s, args.n_evals, n_sim, t_sim))

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

    out_csv = results_dir / "Scenario_3_Novel_Final.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, sep=";", index=False)
    print(f"Done! Saved to {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
