package ea.comparison;

/**
 * Entry point chính cho JGEA Runner.
 *
 * <p>Cú pháp:
 * <pre>
 *   java -jar jgea-runner-1.0-jar-with-dependencies.jar {s1|s2} [options]
 *
 *   Options:
 *     --quick          Quick mode: 2 reps, 500 evals, dim=20 (S1) / 1 dataset (S2)
 *     --n_rep N        Number of repetitions (default: 30)
 *     --n_evals E      Max fitness evaluations per run (default: 10000)
 *     --cores C        Number of parallel threads (default: nCPU-1)
 * </pre>
 */
public class Main {

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            printUsage();
            System.exit(1);
        }

        String mode = args[0].toLowerCase();
        String[] rest = new String[args.length - 1];
        System.arraycopy(args, 1, rest, 0, rest.length);

        switch (mode) {
            case "s1" -> Scenario1Runner.run(rest);
            case "s2" -> Scenario2Runner.run(rest);
            default -> {
                System.err.println("Unknown mode: " + mode);
                printUsage();
                System.exit(1);
            }
        }
    }

    private static void printUsage() {
        System.out.println("""
                EA Comparison — JGEA Runner
                ===========================
                Usage:
                  java -jar jgea-runner-1.0-jar-with-dependencies.jar <mode> [options]
                
                Modes:
                  s1    Scenario 1: Synthetic Numerical Benchmarks
                        (Sphere, PA-1, PA-10, CPA, Ackley, Rastrigin)
                  s2    Scenario 2: ANN Regression on Concrete / Energy / Wine
                
                Options:
                  --quick          Quick smoke-test (2 reps, 500 evals)
                  --n_rep N        Repetitions (default: 30)
                  --n_evals E      Max evaluations per run (default: 10000)
                  --cores C        Parallel threads (default: nCPU-1)
                
                Examples:
                  # Quick test S1 (≈30 seconds)
                  java -jar jgea-runner-1.0-jar-with-dependencies.jar s1 --quick
                
                  # Full S1 with 4 cores
                  java -jar jgea-runner-1.0-jar-with-dependencies.jar s1 --n_rep 30 --cores 4
                
                  # Quick test S2
                  java -jar jgea-runner-1.0-jar-with-dependencies.jar s2 --quick
                """);
    }
}
