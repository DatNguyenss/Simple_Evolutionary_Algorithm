package ea.comparison.solver;

import io.github.ericmedvet.jgea.core.Factory;
import io.github.ericmedvet.jgea.core.operator.Crossover;
import io.github.ericmedvet.jgea.core.operator.GeneticOperator;
import io.github.ericmedvet.jgea.core.operator.Mutation;
import io.github.ericmedvet.jgea.core.problem.QualityBasedProblem;
import io.github.ericmedvet.jgea.core.problem.TotalOrderQualityBasedProblem;
import io.github.ericmedvet.jgea.core.selector.Last;
import io.github.ericmedvet.jgea.core.selector.Tournament;
import io.github.ericmedvet.jgea.core.solver.*;
import io.github.ericmedvet.jgea.core.solver.es.CMAESState;
import io.github.ericmedvet.jgea.core.solver.es.CMAEvolutionaryStrategy;
import io.github.ericmedvet.jgea.core.solver.pso.PSOState;
import io.github.ericmedvet.jgea.core.solver.pso.ParticleSwarmOptimization;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * SolverFactory — creates 9 EA solvers matching paper hyperparameters.
 *
 * <p>JGEA 2.8.1 API:
 * <ul>
 *   <li>{@link StandardEvolver} stopCondition type = Predicate&lt;? super POCPopulationState&lt;Individual&lt;G,S,Q&gt;, G, S, Q, QualityBasedProblem&lt;S,Q&gt;&gt;&gt;
 *   <li>{@link CMAEvolutionaryStrategy} stopCondition = Predicate&lt;? super CMAESState&lt;S,Q&gt;&gt;
 *   <li>{@link ParticleSwarmOptimization} stopCondition = Predicate&lt;? super PSOState&lt;S,Q&gt;&gt;
 *   <li>{@link DifferentialEvolution} stopCondition = Predicate&lt;? super ListPopulationState&lt;...&gt;&gt;
 *   <li>Crossover.recombine(g1, g2, rng), Mutation.mutate(g, rng) (NOT apply/lambda patterns)
 * </ul>
 */
public class SolverFactory {

    private static final int DE_POP = 15;
    private static final double DE_F = 0.5;
    private static final double DE_CR = 0.8;
    private static final int PSO_POP = 100;
    private static final double PSO_W = 0.8;
    private static final double PSO_PHI = 1.5;
    private static final int ES_POP = 30;
    private static final int ES_PARENTS = 10;
    private static final int GA_POP = 100;
    private static final int GA_TOURNAMENT_SIZE = 5;
    private static final double GA_P_XO = 0.8;

    /**
     * Returns an IterativeSolver for the given key.
     * Cast to IterativeSolver is needed because JGEA has complex type hierarchy.
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    public static IterativeSolver buildSolver(String solverKey, int dim, int maxEvals) {
        Factory<List<Double>> factory = uniformFactory(dim, -1.0, 1.0);
        Function<List<Double>, List<Double>> id = g -> g;

        return switch (solverKey) {

            // ── Differential Evolution ──────────────────────────────────────────
            case "differentialEvolution" -> {
                Predicate<ListPopulationState<Individual<List<Double>, List<Double>, Double>,
                        List<Double>, List<Double>, Double,
                        TotalOrderQualityBasedProblem<List<Double>, Double>>> stop =
                        state -> state.nOfQualityEvaluations() >= maxEvals;
                yield new DifferentialEvolution<List<Double>, Double>(id, factory, DE_POP, stop, DE_F, DE_CR, false);
            }

            // ── CMA-ES ─────────────────────────────────────────────────────────
            case "cmaEs" -> {
                Predicate<CMAESState<List<Double>, Double>> stop =
                        state -> state.nOfQualityEvaluations() >= maxEvals;
                yield new CMAEvolutionaryStrategy<List<Double>, Double>(id, factory, stop);
            }

            // ── PSO ────────────────────────────────────────────────────────────
            case "pso" -> {
                Predicate<PSOState<List<Double>, Double>> stop =
                        state -> state.nOfQualityEvaluations() >= maxEvals;
                yield new ParticleSwarmOptimization<List<Double>, Double>(
                        id, factory, stop, PSO_POP, PSO_W, PSO_PHI, PSO_PHI);
            }

            // ── Evolution Strategy ─────────────────────────────────────────────
            case "es-0.02" -> buildEs(0.02, factory, id, maxEvals);
            case "es-0.25" -> buildEs(0.25, factory, id, maxEvals);
            case "es-0.5"  -> buildEs(0.5,  factory, id, maxEvals);

            // ── Genetic Algorithm ──────────────────────────────────────────────
            case "ga-0.02" -> buildGa(0.02, factory, id, maxEvals);
            case "ga-0.25" -> buildGa(0.25, factory, id, maxEvals);
            case "ga-0.5"  -> buildGa(0.5,  factory, id, maxEvals);

            default -> throw new IllegalArgumentException("Unknown solver: " + solverKey);
        };
    }

    // ── ES: Gaussian mutation only ─────────────────────────────────────────────

    private static StandardEvolver<List<Double>, List<Double>, Double> buildEs(
            double sigma, Factory<List<Double>> factory,
            Function<List<Double>, List<Double>> id, int maxEvals) {

        double std = Math.sqrt(sigma);
        Mutation<List<Double>> mut = (g, rng) -> {
            List<Double> child = new ArrayList<>(g.size());
            for (double v : g) child.add(v + rng.nextGaussian() * std);
            return child;
        };
        // StandardEvolver needs Predicate<? super POCPopulationState<Individual<G,S,Q>, G, S, Q, QualityBasedProblem<S,Q>>>
        Predicate<POCPopulationState<Individual<List<Double>, List<Double>, Double>,
                List<Double>, List<Double>, Double,
                QualityBasedProblem<List<Double>, Double>>> stop =
                s -> s.nOfQualityEvaluations() >= maxEvals;

        return new StandardEvolver<>(id, factory, ES_POP, stop,
                Map.of(mut, 1.0),
                new Tournament(ES_PARENTS), new Last(),
                ES_POP, true, 0, false, List.of());
    }

    // ── GA: BLX crossover + Gaussian mutation ─────────────────────────────────

    private static StandardEvolver<List<Double>, List<Double>, Double> buildGa(
            double sigma, Factory<List<Double>> factory,
            Function<List<Double>, List<Double>> id, int maxEvals) {

        double std = Math.sqrt(sigma);
        Crossover<List<Double>> blx = (p1, p2, rng) -> {
            double alpha = rng.nextDouble();
            List<Double> child = new ArrayList<>(p1.size());
            for (int i = 0; i < p1.size(); i++)
                child.add(p1.get(i) + alpha * (p2.get(i) - p1.get(i)) + rng.nextGaussian() * std);
            return child;
        };
        Mutation<List<Double>> mut = (g, rng) -> {
            List<Double> child = new ArrayList<>(g.size());
            for (double v : g) child.add(v + rng.nextGaussian() * std);
            return child;
        };
        Map<GeneticOperator<List<Double>>, Double> ops = Map.of(blx, GA_P_XO, mut, 1.0 - GA_P_XO);
        Predicate<POCPopulationState<Individual<List<Double>, List<Double>, Double>,
                List<Double>, List<Double>, Double,
                QualityBasedProblem<List<Double>, Double>>> stop =
                s -> s.nOfQualityEvaluations() >= maxEvals;

        return new StandardEvolver<>(id, factory, GA_POP, stop, ops,
                new Tournament(GA_TOURNAMENT_SIZE), new Last(),
                GA_POP, true, 0, false, List.of());
    }

    // ── Uniform random factory ─────────────────────────────────────────────────

    public static Factory<List<Double>> uniformFactory(int dim, double lo, double hi) {
        return (n, random) -> {
            List<List<Double>> result = new ArrayList<>(n);
            for (int k = 0; k < n; k++) {
                List<Double> ind = new ArrayList<>(dim);
                for (int i = 0; i < dim; i++) ind.add(lo + random.nextDouble() * (hi - lo));
                result.add(ind);
            }
            return result;
        };
    }
}
