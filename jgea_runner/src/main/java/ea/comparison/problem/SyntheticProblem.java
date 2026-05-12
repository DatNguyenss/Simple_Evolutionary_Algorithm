package ea.comparison.problem;

import io.github.ericmedvet.jgea.core.order.PartialComparator;
import io.github.ericmedvet.jgea.core.problem.TotalOrderQualityBasedProblem;

import java.util.*;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/**
 * Synthetic benchmark problems (Scenario 1) — paper Section 2.2.1
 *
 * <p>Implements {@code TotalOrderQualityBasedProblem<List<Double>, Double>}
 * để tương thích với cả {@code StandardEvolver} (GA/ES) và {@code DifferentialEvolution}.
 *
 * <p>Problems:
 * <ul>
 *   <li>Sphere:    f(x) = sum(x_i^2)
 *   <li>PA-1:      f(x) = ||x - 1||
 *   <li>PA-10:     f(x) = ||x - 10||
 *   <li>CPA:       f(x) = min_i ||x - t_i||  (5 points on circle r=2, centre=1)
 *   <li>Ackley:    standard formula
 *   <li>Rastrigin: standard formula
 * </ul>
 */
public class SyntheticProblem implements TotalOrderQualityBasedProblem<List<Double>, Double> {

    private final int dim;
    private final Function<List<Double>, Double> fitnessFunction;
    private final String name;

    private SyntheticProblem(String name, int dim, Function<List<Double>, Double> fn) {
        this.name = name;
        this.dim = dim;
        this.fitnessFunction = fn;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Factory methods
    // ──────────────────────────────────────────────────────────────────────────

    public static SyntheticProblem of(String fnTag, int dim) {
        return switch (fnTag) {
            case "sphere"    -> sphere(dim);
            case "pa1"       -> pointAiming(dim, 1.0);
            case "pa10"      -> pointAiming(dim, 10.0);
            case "cpa"       -> circularPointAiming(dim);
            case "ackley"    -> ackley(dim);
            case "rastrigin" -> rastrigin(dim);
            default -> throw new IllegalArgumentException("Unknown benchmark: " + fnTag);
        };
    }

    /** f(x) = sum(x_i^2), optimum at origin. */
    public static SyntheticProblem sphere(int dim) {
        return new SyntheticProblem("Sphere-" + dim, dim, x -> {
            double s = 0;
            for (double v : x) s += v * v;
            return s;
        });
    }

    /** f(x) = ||x - target||_2, target = scalar * ones. */
    public static SyntheticProblem pointAiming(int dim, double target) {
        return new SyntheticProblem("PA-" + (int) target + "-" + dim, dim, x -> {
            double s = 0;
            for (double v : x) {
                double diff = v - target;
                s += diff * diff;
            }
            return Math.sqrt(s);
        });
    }

    /**
     * Circular Point Aiming: f(x) = min_i ||x - t_i||
     * 5 target points on a circle (radius=2, centre=1) in the (x1,x2) plane;
     * remaining dims are fixed at 1. Targets are deterministic (seed=42).
     */
    public static SyntheticProblem circularPointAiming(int dim) {
        List<double[]> targets = buildCpaTargets(dim, 42L);
        return new SyntheticProblem("CPA-" + dim, dim, x -> {
            double best = Double.MAX_VALUE;
            for (double[] t : targets) {
                double d = 0;
                for (int i = 0; i < dim; i++) {
                    double diff = x.get(i) - t[i];
                    d += diff * diff;
                }
                d = Math.sqrt(d);
                if (d < best) best = d;
            }
            return best;
        });
    }

    /**
     * Ackley function (standard).
     * f(x) = -20 exp(-0.2 sqrt(mean(x^2))) - exp(mean(cos(2π x_i))) + 20 + e
     */
    public static SyntheticProblem ackley(int dim) {
        return new SyntheticProblem("Ackley-" + dim, dim, x -> {
            double sumSq = 0, sumCos = 0;
            for (double v : x) {
                sumSq += v * v;
                sumCos += Math.cos(2.0 * Math.PI * v);
            }
            return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / dim))
                    - Math.exp(sumCos / dim)
                    + 20.0 + Math.E;
        });
    }

    /**
     * Rastrigin function.
     * f(x) = 10d + sum(x_i^2 - 10 cos(2π x_i))
     */
    public static SyntheticProblem rastrigin(int dim) {
        return new SyntheticProblem("Rastrigin-" + dim, dim, x -> {
            double s = 10.0 * dim;
            for (double v : x) {
                s += v * v - 10.0 * Math.cos(2.0 * Math.PI * v);
            }
            return s;
        });
    }

    // ──────────────────────────────────────────────────────────────────────────
    // TotalOrderQualityBasedProblem interface
    // ──────────────────────────────────────────────────────────────────────────

    @Override
    public Function<List<Double>, Double> qualityFunction() {
        return fitnessFunction;
    }

    /** Minimize: smaller quality = better. */
    @Override
    public Comparator<Double> totalOrderComparator() {
        return Double::compareTo;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helper
    // ──────────────────────────────────────────────────────────────────────────

    private static List<double[]> buildCpaTargets(int dim, long seed) {
        Random rng = new Random(seed);
        List<double[]> targets = new ArrayList<>();
        for (int k = 0; k < 5; k++) {
            double angle = rng.nextDouble() * 2.0 * Math.PI;
            double[] t = new double[dim];
            Arrays.fill(t, 1.0);
            t[0] += 2.0 * Math.cos(angle);
            if (dim > 1) t[1] += 2.0 * Math.sin(angle);
            targets.add(t);
        }
        return targets;
    }

    public int getDim() { return dim; }
    public String getName() { return name; }

    @Override
    public String toString() { return "SyntheticProblem{" + name + "}"; }
}
