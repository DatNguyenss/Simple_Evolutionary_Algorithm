import numpy as np
import pandas as pd
import time
import os
import warnings
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
N_EVALS = 10_000
N_REP   = 30
N_CORES = max(1, cpu_count() - 1)
DIMS    = [20, 100, 200, 500]

SOLVER_NAME_MAP = {
    'CMA-ES':   'cmaEs',
    'DE':       'differentialEvolution',
    'PSO':      'pso',
    'ES-0.02':  'es-0.02',
    'ES-0.25':  'es-0.25',
    'ES-0.5':   'es-0.5',
    'GA-0.02':  'ga-0.02',
    'GA-0.25':  'ga-0.25',
    'GA-0.5':   'ga-0.5',
}

# ============================================================
# BENCHMARKS
# ============================================================
def sphere(x):
    return float(np.sum(x**2))

def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))

def griewank(x):
    return float(np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, len(x)+1)))) + 1)

def ackley(x):
    d = len(x)
    return float(-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/d))
                 - np.exp(np.sum(np.cos(2*np.pi*x))/d)
                 + 20 + np.e)

def rastrigin(x):
    d = len(x)
    return float(10*d + np.sum(x**2 - 10*np.cos(2*np.pi*x)))

def point_aiming(x, target):
    return float(np.linalg.norm(x-target))

def circular_point_aiming(x, targets):
    return float(min(np.linalg.norm(x-t) for t in targets))

# ============================================================
# RECORD
# ============================================================
def _record(iteration, total_evals, births, best_fit, t0, pop, fitness, n_firsts=1, n_lasts=None):
    n_pop = len(pop)
    n_lasts = n_lasts if n_lasts is not None else n_pop

    rounded_pop = np.round(pop, 6)
    geno_uni = len(np.unique(rounded_pop, axis=0)) / n_pop
    fit_uni = len(np.unique(np.round(fitness, 8))) / n_pop

    return {
        'iterations': iteration,
        'evals': total_evals,
        'births': births,
        'elapsed': round(time.time() - t0, 4),
        'all_size': n_pop,
        'firsts_size': n_firsts,
        'lasts_size': n_lasts,
        'geno_uni': geno_uni,
        'sol_uni': geno_uni,
        'fit_uni': fit_uni,
        'best_fitness': best_fit,
    }

# ============================================================
# SOLVERS
# ============================================================
def run_cma_es(fitness_fn, dim, seed):
    import cma
    t0 = time.time()

    es = cma.CMAEvolutionStrategy(
        np.zeros(dim),
        0.33,
        {'seed': int(seed), 'verbose': -9, 'maxfevals': N_EVALS}
    )

    records = []
    total_evals = 0
    births = 0
    iteration = 0

    while not es.stop() and total_evals < N_EVALS:
        solutions = es.ask()
        fits = np.array([fitness_fn(s) for s in solutions])
        es.tell(solutions, fits.tolist())

        n = len(solutions)
        total_evals += n
        births += n
        iteration += 1

        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   float(np.min(fits)), t0,
                                   np.array(solutions), fits,
                                   n_lasts=n))
    return records

def run_de(fitness_fn, dim, seed):
    t0, rng = time.time(), np.random.default_rng(seed)

    NP, F, CR = 20, 0.7, 0.9

    pop = rng.uniform(-1, 1, (NP, dim))
    fitness = np.array([fitness_fn(ind) for ind in pop])

    total_evals, iteration = NP, 0
    records = [_record(0, NP, NP, np.min(fitness), t0, pop, fitness)]

    while total_evals < N_EVALS:
        iteration += 1
        best_idx = np.argmin(fitness)

        for i in range(NP):
            if total_evals >= N_EVALS:
                break

            idxs = rng.choice([j for j in range(NP) if j != i], 2, replace=False)

            mutant = pop[i] + F*(pop[best_idx]-pop[i]) + F*(pop[idxs[0]]-pop[idxs[1]])
            mutant = np.clip(mutant, -10, 10)

            mask = rng.random(dim) < CR
            if not np.any(mask):
                mask[rng.integers(dim)] = True

            trial = np.where(mask, mutant, pop[i])

            f_trial = fitness_fn(trial)
            total_evals += 1

            if f_trial <= fitness[i]:
                pop[i], fitness[i] = trial, f_trial

        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, total_evals,
                                   np.min(fitness), t0, pop, fitness))
    return records

def run_pso(fitness_fn, dim, seed):
    t0, rng = time.time(), np.random.default_rng(seed)

    n_pop = 50
    chi = 0.729
    c1 = c2 = 1.49445

    pos = rng.uniform(-1, 1, (n_pop, dim))
    vel = rng.uniform(-0.5, 0.5, (n_pop, dim))

    fit = np.array([fitness_fn(p) for p in pos])

    pbest_pos = pos.copy()
    pbest_fit = fit.copy()

    gbest_idx = np.argmin(fit)
    gbest_fit = fit[gbest_idx]
    gbest_pos = pos[gbest_idx].copy()

    total_evals, iteration = n_pop, 0
    records = [_record(0, n_pop, n_pop, gbest_fit, t0, pos, fit)]

    while total_evals < N_EVALS:
        iteration += 1

        r1 = rng.random((n_pop, dim))
        r2 = rng.random((n_pop, dim))

        vel = chi * (vel + c1*r1*(pbest_pos-pos) + c2*r2*(gbest_pos-pos))
        pos = np.clip(pos + vel, -10, 10)

        fit = np.array([fitness_fn(p) for p in pos])
        total_evals += n_pop

        improved = fit < pbest_fit
        pbest_pos[improved] = pos[improved]
        pbest_fit[improved] = fit[improved]

        idx = np.argmin(pbest_fit)
        if pbest_fit[idx] < gbest_fit:
            gbest_fit = pbest_fit[idx]
            gbest_pos = pbest_pos[idx].copy()

        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, total_evals,
                                   gbest_fit, t0, pos, fit))
    return records

def run_es(fitness_fn, dim, seed, sigma=0.25):
    t0, rng = time.time(), np.random.default_rng(seed)

    mu, lamb = 9, 30

    pop = rng.uniform(-1, 1, (mu, dim))
    fit = np.array([fitness_fn(ind) for ind in pop])

    sigmas = np.full(mu, sigma)

    total_evals, births, iteration = mu, mu, 0
    records = [_record(0, mu, mu, np.min(fit), t0, pop, fit)]

    tau = 1 / np.sqrt(dim)

    while total_evals < N_EVALS:
        iteration += 1

        p_idx = rng.integers(0, mu, lamb)
        child_sigma = sigmas[p_idx] * np.exp(tau * rng.normal(size=lamb))

        offspring = pop[p_idx] + rng.normal(size=(lamb, dim)) * child_sigma[:, None]
        off_fit = np.array([fitness_fn(o) for o in offspring])

        total_evals += lamb
        births += lamb

        merged_pop = np.vstack([pop, offspring])
        merged_fit = np.concatenate([fit, off_fit])
        merged_sig = np.concatenate([sigmas, child_sigma])

        best_idx = np.argsort(merged_fit)[:mu]

        pop = merged_pop[best_idx]
        fit = merged_fit[best_idx]
        sigmas = merged_sig[best_idx]

        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   np.min(fit), t0, pop, fit, n_lasts=lamb))
    return records

def run_ga(fitness_fn, dim, seed, sigma=0.25):
    t0, rng = time.time(), np.random.default_rng(seed)

    n_pop, n_tour = 100, 5

    pop = rng.uniform(-1, 1, (n_pop, dim))
    fit = np.array([fitness_fn(ind) for ind in pop])

    total_evals, births, iteration = n_pop, n_pop, 0
    records = [_record(0, n_pop, n_pop, np.min(fit), t0, pop, fit)]

    while total_evals < N_EVALS:
        iteration += 1
        offspring = np.empty_like(pop)

        for k in range(n_pop):
            parts1 = rng.choice(n_pop, n_tour, replace=False)
            parts2 = rng.choice(n_pop, n_tour, replace=False)

            p1 = pop[parts1[np.argmin(fit[parts1])]]
            p2 = pop[parts2[np.argmin(fit[parts2])]]

            alpha = rng.random(dim)
            child = alpha*p1 + (1-alpha)*p2
            child += rng.normal(0, sigma/np.sqrt(dim), dim)

            offspring[k] = child

        off_fit = np.array([fitness_fn(o) for o in offspring])

        total_evals += n_pop
        births += n_pop

        merged_pop = np.vstack([pop, offspring])
        merged_fit = np.concatenate([fit, off_fit])

        best_idx = np.argsort(merged_fit)[:n_pop]
        pop = merged_pop[best_idx]
        fit = merged_fit[best_idx]

        if iteration % 5 == 0:
            records.append(_record(iteration, total_evals, births,
                                   np.min(fit), t0, pop, fit))
    return records

# ============================================================
# WORKER
# ============================================================
def _worker(args):
    prob_key, jname, fn_tag, dim, obj, solver_key, sigma, seed = args

    if fn_tag == 'sphere':
        fn = sphere
    elif fn_tag == 'ackley':
        fn = ackley
    elif fn_tag == 'rastrigin':
        fn = rastrigin
    elif fn_tag == 'rosenbrock':
        fn = rosenbrock
    elif fn_tag == 'griewank':
        fn = griewank
    elif fn_tag == 'pa10':
        target = np.full(dim, 10.0)
        fn = lambda x: point_aiming(x, target)
    elif fn_tag == 'pa1':
        target = np.ones(dim)
        fn = lambda x: point_aiming(x, target)
    elif fn_tag == 'cpa':
        rng = np.random.default_rng(42)
        ts = [np.ones(dim)]
        angles = rng.uniform(0, 2*np.pi, 5)
        for a in angles:
            t = np.ones(dim)
            t[0] += 2*np.cos(a)
            if dim > 1:
                t[1] += 2*np.sin(a)
            ts.append(t)
        fn = lambda x: circular_point_aiming(x, ts)
    else:
        raise ValueError(f"Tag error: {fn_tag}")

    if solver_key == 'CMA-ES':
        records = run_cma_es(fn, dim, seed)
    elif solver_key == 'DE':
        records = run_de(fn, dim, seed)
    elif solver_key == 'PSO':
        records = run_pso(fn, dim, seed)
    elif solver_key.startswith('ES-'):
        records = run_es(fn, dim, seed, sigma)
    elif solver_key.startswith('GA-'):
        records = run_ga(fn, dim, seed, sigma)
    else:
        records = []

    j_solv = SOLVER_NAME_MAP[solver_key]

    return [
        {**r,
         'seed': seed+1,
         'problem': jname,
         'solver_sigma': j_solv,
         'objective': obj,
         'best→fitness': r['best_fitness']}
        for r in records
    ]

# ============================================================
# MAIN
# ============================================================
def main():
    tasks = []

    for d in DIMS:
        problems = {
            f'Sphere-{d}': ('sphere', f'ea.p.s.sphere-{d}'),
            f'PA-1-{d}': ('pa1', f'ea.p.s.pointAiming-{d}-1'),
            f'PA-10-{d}': ('pa10', f'ea.p.s.pointAiming-{d}-10'),
            f'CPA-{d}': ('cpa', f'ea.p.s.circularPointAiming-{d}'),
            f'Ackley-{d}': ('ackley', f'ea.p.s.ackley-{d}'),
            f'Rastrigin-{d}': ('rastrigin', f'ea.p.s.rastrigin-{d}'),
            f'Rosenbrock-{d}': ('rosenbrock', f'ea.p.s.rosenbrock-{d}'),
            f'Griewank-{d}': ('griewank', f'ea.p.s.griewank-{d}')
        }

        for pk, (tag, jname) in problems.items():
            for sk in SOLVER_NAME_MAP.keys():
                sig = float(sk.split('-')[1]) if sk.startswith('ES-') or sk.startswith('GA-') else None
                for s in range(N_REP):
                    tasks.append((pk, jname, tag, d, 'minimize', sk, sig, s))

    all_rows = []

    with Pool(N_CORES) as p:
        for res in tqdm(p.imap_unordered(_worker, tasks, chunksize=20), total=len(tasks)):
            all_rows.extend(res)

    os.makedirs('results', exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv('results/Scenario_1_Novel_Final.csv', sep=';', index=False)

    print("Done! Saved to results/Scenario_1_Novel_Final.csv")

if __name__ == '__main__':
    main()