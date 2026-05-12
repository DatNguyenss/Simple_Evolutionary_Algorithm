package ea.comparison;

import ea.comparison.problem.AnnRegressionProblem;
import ea.comparison.solver.SolverFactory;
import io.github.ericmedvet.jgea.core.solver.Individual;
import io.github.ericmedvet.jgea.core.solver.ListPopulationState;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;

/**
 * Scenario 2 — ANN Regression (paper Section 2.2.2).
 *
 * <p>Datasets: Concrete (1030×9), Energy (768×9), Wine (6497×12).
 * ρ_h ∈ {1, 2}: hidden = ρ_h × m features.
 * 9 EAs × 30 seeds × 10 000 evals.
 */
public class Scenario2Runner {

    private static final Logger LOG = Logger.getLogger(Scenario2Runner.class.getName());

    private static final String[] SOLVER_KEYS = {
        "cmaEs", "differentialEvolution", "pso",
        "es-0.02", "es-0.25", "es-0.5",
        "ga-0.02", "ga-0.25", "ga-0.5"
    };
    private static final int[] RHO_H_VALUES = {1, 2};

    record DatasetSpec(String name, String csvPath, int targetCol) {}

    public static void run(String[] args) throws Exception {
        int nEvals = Scenario1Runner.N_EVALS_DEFAULT;
        int nRep   = Scenario1Runner.N_REP_DEFAULT;
        int cores  = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
        boolean quick = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--quick"  -> quick = true;
                case "--n_evals"-> nEvals = Integer.parseInt(args[++i]);
                case "--n_rep"  -> nRep = Integer.parseInt(args[++i]);
                case "--cores"  -> cores = Integer.parseInt(args[++i]);
            }
        }
        if (quick) { nEvals = Math.min(nEvals, 500); nRep = Math.min(nRep, 2); cores = 1; }
        LOG.info(String.format("Scenario 2 | n_rep=%d | n_evals=%d | cores=%d", nRep, nEvals, cores));

        List<DatasetSpec> specs = List.of(
            new DatasetSpec("concrete", "Scenario_2/data/Concrete_Data.csv",   8),
            new DatasetSpec("energy",   "Scenario_2/data/ENB2012_data.csv",    8),
            new DatasetSpec("wine",     "Scenario_2/data/winequality-white.csv", 11)
        );

        // Load datasets
        Map<String, double[][]> Xs = new LinkedHashMap<>();
        Map<String, double[]>   ys = new LinkedHashMap<>();
        for (DatasetSpec ds : specs) {
            Path p = Path.of(ds.csvPath);
            if (!Files.exists(p)) { LOG.warning("Missing: " + p); continue; }
            double[][] raw = loadCsv(p, ds.targetCol);
            int n = raw.length, m = raw[0].length - 1;
            double[][] X = new double[n][m];
            double[] y = new double[n];
            for (int i = 0; i < n; i++) { System.arraycopy(raw[i], 0, X[i], 0, m); y[i] = raw[i][m]; }
            standardize(X);
            rescaleY(y);
            Xs.put(ds.name, X); ys.put(ds.name, y);
            LOG.info(String.format("Loaded %s: n=%d, m=%d", ds.name, n, m));
        }
        if (Xs.isEmpty()) { LOG.severe("No datasets! Aborting."); return; }

        // Build tasks
        record Task(AnnRegressionProblem prob, String ds, int rhoH, String sk, int seed, int nE) {}
        List<Task> tasks = new ArrayList<>();
        for (String dsName : Xs.keySet()) {
            double[][] X = Xs.get(dsName);
            double[] y = ys.get(dsName);
            for (int rhoH : RHO_H_VALUES) {
                AnnRegressionProblem prob = new AnnRegressionProblem(dsName, X, y, rhoH);
                for (String sk : SOLVER_KEYS)
                    for (int s = 0; s < nRep; s++)
                        tasks.add(new Task(prob, dsName, rhoH, sk, s, nEvals));
            }
        }
        LOG.info("Total S2 tasks: " + tasks.size());

        List<String> lines = Collections.synchronizedList(new ArrayList<>());
        lines.add("iteration;evals;elapsed_sec;best_fitness;seed;dataset;rho_h;solver;dim");

        Callable<List<String>> mkCall = () -> null;  // placeholder
        if (cores <= 1) {
            for (Task t : tasks) lines.addAll(runTask(t.prob(), t.ds(), t.rhoH(), t.sk(), t.seed(), t.nE()));
        } else {
            ExecutorService pool = Executors.newFixedThreadPool(cores);
            List<Future<List<String>>> futures = tasks.stream()
                    .map(t -> pool.submit(() -> runTask(t.prob(), t.ds(), t.rhoH(), t.sk(), t.seed(), t.nE())))
                    .toList();
            for (Future<List<String>> f : futures) lines.addAll(f.get());
            pool.shutdown();
        }

        Path outDir = Path.of("Scenario_2", "results");
        Files.createDirectories(outDir);
        Path out = outDir.resolve("Scenario_2_JGEA.csv");
        Files.write(out, lines);
        LOG.info("Saved to: " + out.toAbsolutePath());
    }

    static List<String> runTask(AnnRegressionProblem prob, String ds, int rhoH,
                                 String sk, int seed, int nEvals) {
        List<String> lines = new ArrayList<>();
        try {
            var solver = SolverFactory.buildSolver(sk, prob.getDim(), nEvals);
            var rng = new Random(seed);
            var exec = Executors.newSingleThreadExecutor();
            long t0 = System.nanoTime();
            final int[] iter = {0};

            solver.solve(prob, rng, exec, state -> {
                iter[0]++;
                long ev = Scenario1Runner.getEvals(state);
                double best = Scenario1Runner.extractBestFitness(state);
                double elapsed = (System.nanoTime() - t0) / 1e9;
                if (iter[0] % 5 == 0 || ev >= nEvals) {
                    lines.add(String.join(";",
                            String.valueOf(iter[0]), String.valueOf(ev),
                            String.format("%.4f", elapsed), String.valueOf(best),
                            String.valueOf(seed + 1), ds, String.valueOf(rhoH), sk,
                            String.valueOf(prob.getDim())));
                }
            });
            exec.shutdownNow();
        } catch (Exception e) {
            LOG.warning("S2 task failed " + ds + "/" + rhoH + "/" + sk + "/seed=" + seed + ": " + e.getMessage());
        }
        return lines;
    }

    // ── Data utilities ─────────────────────────────────────────────────────────

    static double[][] loadCsv(Path path, int targetCol) throws IOException {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(path)) {
            String line; boolean header = true;
            while ((line = br.readLine()) != null) {
                line = line.trim(); if (line.isEmpty()) continue;
                String sep = line.contains(";") ? ";" : ",";
                String[] parts = line.split(sep);
                if (header) { header = false; continue; }
                double[] row = new double[parts.length]; boolean ok = true;
                for (int i = 0; i < parts.length; i++) {
                    try { row[i] = Double.parseDouble(parts[i].trim()); }
                    catch (NumberFormatException e) { ok = false; break; }
                }
                if (ok) {
                    double[] r = new double[parts.length]; int pos = 0;
                    for (int i = 0; i < parts.length; i++) if (i != targetCol) r[pos++] = row[i];
                    r[pos] = row[targetCol]; rows.add(r);
                }
            }
        }
        return rows.toArray(new double[0][]);
    }

    static void standardize(double[][] X) {
        int n = X.length, m = X[0].length;
        for (int j = 0; j < m; j++) {
            double mean = 0; for (int i = 0; i < n; i++) mean += X[i][j]; mean /= n;
            double std = 0; for (int i = 0; i < n; i++) { double d = X[i][j] - mean; std += d * d; }
            std = Math.sqrt(std / n); if (std < 1e-10) std = 1;
            for (int i = 0; i < n; i++) X[i][j] = (X[i][j] - mean) / std;
        }
    }

    static void rescaleY(double[] y) {
        double min = Arrays.stream(y).min().orElse(0);
        double max = Arrays.stream(y).max().orElse(1);
        double range = max - min; if (range < 1e-10) range = 1;
        for (int i = 0; i < y.length; i++) y[i] = 2.0 * (y[i] - min) / range - 1.0;
    }
}
