package ea.comparison;

import ea.comparison.problem.SyntheticProblem;
import ea.comparison.solver.SolverFactory;
import io.github.ericmedvet.jgea.core.solver.Individual;
import io.github.ericmedvet.jgea.core.solver.IterativeSolver;
import io.github.ericmedvet.jgea.core.solver.ListPopulationState;
import io.github.ericmedvet.jgea.core.solver.POCPopulationState;
import io.github.ericmedvet.jgea.core.solver.State;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;

/**
 * Scenario 1 — Synthetic Numerical Benchmarks (paper Section 2.2.1)
 *
 * <p>24 problems = 6 functions × 4 dims; 9 EAs; n_rep=30 seeds; 10 000 evals.
 *
 * <p>Chạy:
 * <pre>
 *   java -jar jgea-runner-1.0-jar-with-dependencies.jar s1 [--quick] [--n_rep N] [--n_evals E] [--cores C]
 * </pre>
 */
public class Scenario1Runner {

    private static final Logger LOG = Logger.getLogger(Scenario1Runner.class.getName());

    public static final int N_EVALS_DEFAULT = 10_000;
    public static final int N_REP_DEFAULT = 30;
    public static final int[] DIMS_DEFAULT = {20, 100, 200, 500};

    private static final String[] SOLVER_KEYS = {
        "cmaEs", "differentialEvolution", "pso",
        "es-0.02", "es-0.25", "es-0.5",
        "ga-0.02", "ga-0.25", "ga-0.5"
    };

    record RunTask(String jName, String fnTag, int dim, String solverKey, int seed, int nEvals) {}

    public static void run(String[] args) throws Exception {
        int nEvals = N_EVALS_DEFAULT, nRep = N_REP_DEFAULT;
        int cores = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
        boolean quick = false;
        int[] dims = DIMS_DEFAULT;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--quick"  -> quick = true;
                case "--n_evals"-> nEvals = Integer.parseInt(args[++i]);
                case "--n_rep"  -> nRep = Integer.parseInt(args[++i]);
                case "--cores"  -> cores = Integer.parseInt(args[++i]);
            }
        }
        if (quick) {
            nEvals = Math.min(nEvals, 500); nRep = Math.min(nRep, 2); dims = new int[]{20};
            LOG.info("QUICK MODE: n_rep=" + nRep + " n_evals=" + nEvals + " dims=[20]");
        }
        LOG.info(String.format("Scenario 1 | n_rep=%d | n_evals=%d | cores=%d", nRep, nEvals, cores));

        // Task list
        List<RunTask> tasks = new ArrayList<>();
        String[][] fns = {
            {"sphere",    "ea.p.s.sphere"},
            {"pa1",       "ea.p.s.pointAiming-1"},
            {"pa10",      "ea.p.s.pointAiming-10"},
            {"cpa",       "ea.p.s.circularPointAiming"},
            {"ackley",    "ea.p.s.ackley"},
            {"rastrigin", "ea.p.s.rastrigin"}
        };
        for (int dim : dims) {
            for (String[] fn : fns) {
                for (String sk : SOLVER_KEYS) {
                    for (int s = 0; s < nRep; s++) {
                        tasks.add(new RunTask(fn[1] + "-" + dim, fn[0], dim, sk, s, nEvals));
                    }
                }
            }
        }
        LOG.info("Total tasks: " + tasks.size());

        List<String> lines = Collections.synchronizedList(new ArrayList<>());
        lines.add("iteration;evals;elapsed_sec;best_fitness;seed;problem;solver;dim");

        long t0 = System.currentTimeMillis();
        if (cores <= 1) {
            int done = 0;
            for (RunTask t : tasks) {
                lines.addAll(executeTask(t));
                if (++done % 20 == 0) LOG.info("Progress: " + done + "/" + tasks.size());
            }
        } else {
            ExecutorService pool = Executors.newFixedThreadPool(cores);
            List<Future<List<String>>> futures = tasks.stream()
                    .map(t -> pool.submit(() -> executeTask(t))).toList();
            int done = 0;
            for (Future<List<String>> f : futures) {
                lines.addAll(f.get());
                if (++done % 20 == 0) LOG.info("Progress: " + done + "/" + tasks.size());
            }
            pool.shutdown();
        }
        LOG.info(String.format("Done %.1f s", (System.currentTimeMillis() - t0) / 1000.0));

        Path outDir = Path.of("Scenario_1", "results");
        Files.createDirectories(outDir);
        Path out = outDir.resolve("Scenario_1_JGEA.csv");
        Files.write(out, lines);
        LOG.info("Saved: " + out.toAbsolutePath());
    }

    static List<String> executeTask(RunTask task) {
        List<String> lines = new ArrayList<>();
        try {
            var problem = SyntheticProblem.of(task.fnTag(), task.dim());
            var solver = (IterativeSolver) SolverFactory.buildSolver(task.solverKey(), task.dim(), task.nEvals());
            var rng = new Random(task.seed());
            var exec = Executors.newSingleThreadExecutor();
            long t0 = System.nanoTime();
            final int[] iter = {0};

            solver.solve(problem, rng, exec, state -> {
                iter[0]++;
                // nOfQualityEvaluations() is on POCPopulationState
                long ev = getEvals(state);
                double best = extractBestFitness(state);
                double elapsed = (System.nanoTime() - t0) / 1e9;
                if (iter[0] % 5 == 0 || ev >= task.nEvals()) {
                    lines.add(String.join(";",
                            String.valueOf(iter[0]), String.valueOf(ev),
                            String.format("%.4f", elapsed), String.valueOf(best),
                            String.valueOf(task.seed() + 1), task.jName(),
                            task.solverKey(), String.valueOf(task.dim())));
                }
            });
            exec.shutdownNow();
        } catch (Exception e) {
            LOG.warning("Failed: " + task.jName() + "/" + task.solverKey()
                    + " seed=" + task.seed() + ": " + e.getMessage());
        }
        return lines;
    }

    /** Extract nOfQualityEvaluations from any state type. */
    static long getEvals(Object state) {
        if (state instanceof POCPopulationState<?, ?, ?, ?, ?> ps) return ps.nOfQualityEvaluations();
        if (state instanceof State<?, ?> s) return s.nOfIterations();
        return 0L;
    }

    /** Extract minimum fitness from population state. */
    @SuppressWarnings({"unchecked", "rawtypes"})
    static double extractBestFitness(Object state) {
        try {
            if (state instanceof ListPopulationState lps) {
                return ((List<Individual>) lps.listPopulation()).stream()
                        .mapToDouble(ind -> {
                            Object q = ind.quality();
                            return q instanceof Double d ? d : Double.MAX_VALUE;
                        })
                        .min().orElse(Double.MAX_VALUE);
            }
            if (state instanceof POCPopulationState<?, ?, ?, ?, ?> ps) {
                var firsts = ps.pocPopulation().firsts();
                if (!firsts.isEmpty()) {
                    Object q = firsts.iterator().next().quality();
                    if (q instanceof Double d) return d;
                }
            }
        } catch (Exception ignored) {}
        return Double.MAX_VALUE;
    }
}
