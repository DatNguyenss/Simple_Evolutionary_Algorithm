package ea.comparison.problem;

import io.github.ericmedvet.jgea.core.problem.TotalOrderQualityBasedProblem;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

/**
 * ANN-based Regression problem (Scenario 2) — paper Section 2.2.2
 *
 * <p>Fitness = MSE(ANN(theta), dataset)
 * ANN: 1 hidden layer, tanh activation
 * Architecture: m → rho_h*m → 1
 * Dimension: p = (m*h + h) + (h*1 + 1) where h = rho_h * m
 *
 * <p>Dataset phải được chuẩn hoá trước (standardize X, rescale y to [-1,1]).
 */
public class AnnRegressionProblem implements TotalOrderQualityBasedProblem<List<Double>, Double> {

    private final double[][] X;  // shape [n, m] — đã standardize
    private final double[] y;    // shape [n] — đã rescale to [-1,1]
    private final int m;         // số features
    private final int h;         // hidden size = rho_h * m
    private final int dim;       // số tham số ANN
    private final String name;

    public AnnRegressionProblem(String name, double[][] X, double[] y, int rhoH) {
        this.name = name;
        this.X = X;
        this.y = y;
        this.m = X[0].length;
        this.h = rhoH * m;
        // dim = (m*h + h) + (h + 1) = m*h + 2h + 1
        this.dim = (m * h + h) + (h + 1);
    }

    @Override
    public Function<List<Double>, Double> qualityFunction() {
        return theta -> {
            double[] tArr = theta.stream().mapToDouble(Double::doubleValue).toArray();
            return computeMse(tArr);
        };
    }

    @Override
    public Comparator<Double> totalOrderComparator() {
        return Double::compareTo;   // minimize
    }

    /**
     * Forward pass + MSE.
     * Layout of theta: [W1(m×h) | b1(h) | W2(h×1) | b2(1)]
     */
    private double computeMse(double[] theta) {
        int n = X.length;
        // Unpack
        // W1: m*h values
        int idx = 0;
        double[][] W1 = new double[m][h];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < h; j++)
                W1[i][j] = theta[idx++];
        double[] b1 = Arrays.copyOfRange(theta, idx, idx + h);
        idx += h;
        double[] W2 = Arrays.copyOfRange(theta, idx, idx + h);
        idx += h;
        double b2 = theta[idx];

        // Forward + MSE
        double mse = 0;
        double[] hidden = new double[h];
        for (int row = 0; row < n; row++) {
            // Hidden layer: tanh(X[row] @ W1 + b1)
            for (int j = 0; j < h; j++) {
                double z = b1[j];
                for (int i = 0; i < m; i++) z += X[row][i] * W1[i][j];
                hidden[j] = Math.tanh(z);
            }
            // Output: tanh(hidden @ W2 + b2)
            double z2 = b2;
            for (int j = 0; j < h; j++) z2 += hidden[j] * W2[j];
            double pred = Math.tanh(z2);
            double diff = pred - y[row];
            mse += diff * diff;
        }
        return mse / n;
    }

    public int getDim() { return dim; }
    public String getName() { return name; }

    @Override
    public String toString() {
        return String.format("AnnRegressionProblem{%s, m=%d, h=%d, dim=%d}", name, m, h, dim);
    }
}
