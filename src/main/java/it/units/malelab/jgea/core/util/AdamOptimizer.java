package it.units.malelab.jgea.core.util;

// https://towardsdatascience.com/complete-guide-to-adam-optimization-1e5f29532c3d

// https://arxiv.org/abs/1412.6980

import java.util.stream.IntStream;

public class AdamOptimizer {

  private final double alpha;
  private final double beta1;
  private final double beta2;
  private final double eps;

  private int t = 0;
  private double[] m;
  private double[] v;

  public AdamOptimizer(double alpha, double beta1, double beta2, double eps) {
    this.alpha = alpha;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
  }

  public AdamOptimizer() {
    this(0.001, 0.9, 0.999, 1e-8);
  }

  public double[] optimizeStep(double[] gradient) {
    if (t == 0) {
      initialize(gradient.length);
    }
    if (m.length != gradient.length) {
      throw new IllegalArgumentException(String.format(
          "Wrong dimensionality of the gradient: %d expected, %d found",
          m.length,
          gradient.length
      ));
    }
    t = t + 1;
    double[] g = new double[gradient.length];
    IntStream.range(0, gradient.length).forEach(i -> {
      m[i] = beta1 * m[i] + (1 - beta1) * gradient[i];
      v[i] = beta2 * v[i] + (1 - beta2) * gradient[i] * gradient[i];
      double mHat = m[i] / (1 - Math.pow(beta1, t));
      double vHat = v[i] / (1 - Math.pow(beta2, t));
      g[i] = - alpha * mHat / (Math.sqrt(vHat) + eps);
    });
    return g;
  }

  private void initialize(int n) {
    m = new double[n];
    v = new double[n];
  }

  public void reset() {
    t = 0;
  }

  public AdamOptimizer copy() {
    AdamOptimizer adamOptimizer = new AdamOptimizer(alpha, beta1, beta2, eps);
    adamOptimizer.t = t;
    System.arraycopy(m, 0, adamOptimizer.m, 0, m.length);
    System.arraycopy(v, 0, adamOptimizer.v, 0, v.length);
    return adamOptimizer;
  }

}
