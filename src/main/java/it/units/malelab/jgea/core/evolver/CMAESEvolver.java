/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.malelab.jgea.core.evolver;

import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.order.PartiallyOrderedCollection;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

// source -> https://arxiv.org/pdf/1604.00772.pdf
public class CMAESEvolver<S, F> extends AbstractIterativeEvolver<List<Double>, S, F> {

  private static final Logger L = Logger.getLogger(CMAESEvolver.class.getName());

  // specified
  /**
   * Population size, sample size, number of offspring, λ.
   */
  private final int lambda;
  /**
   * Parent number, number of (positively) selected search points in the population, number of strictly
   * positive recombination weights, µ.
   */
  private final int mu;
  /**
   * Recombination weights.
   */
  private final double[] weights;
  /**
   * Learning rate for the cumulation for the step-size control.
   */
  private final double cSigma;
  /**
   * Damping parameter for step-size update.
   */
  private final double dSigma;
  /**
   * Learning rate for cumulation for the rank-one update of the covariance matrix.
   */
  private final double cc;
  /**
   * Learning rate for the rank-one update of the covariance matrix update.
   */
  private final double c1;
  /**
   * Learning rate for the rank-mu update of the covariance matrix update.
   */
  private final double cMu;
  /**
   * The variance effective selection mass for the mean.
   */
  private final double muEff;

  /**
   * Problem dimensionality.
   */
  private final int n;
  /**
   * Expectation of ||N(0,I)|| == norm(randn(N,1))
   */
  private final double chiN;


  /**
   * Constructs a new CMA-ES instance using default parameters.
   */
  public CMAESEvolver(
      Function<? super List<Double>, ? extends S> solutionMapper,
      Factory<? extends List<Double>> genotypeFactory,
      PartialComparator<? super Individual<List<Double>, S, F>> individualComparator
  ) {
    super(solutionMapper, genotypeFactory, individualComparator);
    n = genotypeFactory.build(1, new Random(0)).get(0).size();
    chiN = Math.sqrt(n) * (1d - 1d / (4d * n) + 1d / (21d * Math.pow(n, 2)));

    // see Table 1 for details on parameters

    // selection and recombination
    lambda = 4 + (int) Math.floor(3 * Math.log(n));
    mu = (int) Math.floor(lambda / 2d);
    weights = new double[mu];
    double sumOfWeights = 0d;
    double sumOfSquaredWeights = 0d;
    for (int i = 0; i < mu; i++) {
      weights[i] = Math.log((lambda + 1) / 2d) - Math.log(i + 1);
      sumOfWeights += weights[i];
      sumOfSquaredWeights += Math.pow(weights[i], 2);
    }
    for (int i = 0; i < mu; i++) {
      weights[i] /= sumOfWeights;
    }
    muEff = Math.pow(sumOfWeights, 2) / sumOfSquaredWeights;

    // step size control
    cSigma = (muEff + 2) / (n + muEff + 5);
    dSigma = 1 + 2 * Math.max(0, Math.sqrt((muEff - 1) / (n + 1d)) - 1) + cSigma;

    // initialize covariance matrix adaptation
    cc = (4d + muEff / n) / (n + 4d + 2 * muEff / n);
    c1 = 2 / (Math.pow((n + 1.3), 2) + muEff);
    cMu = Math.min(1 - c1, 2 * (muEff - 2 + 1 / muEff) / (Math.pow((n + 2), 2) + 2 * muEff / 2d));

  }

  private class CMAESState extends State {
    // Step-size
    private double stepSize = 0.5;
    // Mean value of the search distribution
    private double[] distributionMean = new double[n];
    // Evolution path for step-size
    private double[] sEvolutionPath = new double[n];
    // Evolution path for covariance matrix, a sequence of successive (normalized) steps, the strategy
    // takes over a number of generations
    private double[] cEvolutionPath = new double[n];
    // Orthogonal matrix. Columns of B are eigenvectors of C with unit length and correspond to the diagonal
    // element of D
    // Covariance matrix at the current generation
    private RealMatrix C = MatrixUtils.createRealIdentityMatrix(n);
    private RealMatrix B = MatrixUtils.createRealIdentityMatrix(n);
    // Diagonal matrix. The diagonal elements of D are square roots of eigenvalues of C and correspond to the
    // respective columns of B
    private RealMatrix D = MatrixUtils.createRealIdentityMatrix(n);

    private final double[][] zK = new double[lambda][n];
    private final double[][] yK = new double[lambda][n];
    private final double[][] xK = new double[lambda][n];

    // Last generation when the eigen decomposition was calculated
    private int lastEigenUpdateGeneration = 0;

    public CMAESState() {
    }

    public CMAESState(
        int iterations,
        int births,
        int fitnessEvaluations,
        long elapsedMillis,
        double stepSize,
        double[] distributionMean,
        double[] sEvolutionPath,
        double[] cEvolutionPath,
        RealMatrix B,
        RealMatrix D,
        RealMatrix C,
        int lastEigenUpdateGeneration
    ) {
      super(iterations, births, fitnessEvaluations, elapsedMillis);
      this.stepSize = stepSize;
      this.distributionMean = distributionMean;
      this.sEvolutionPath = sEvolutionPath;
      this.cEvolutionPath = cEvolutionPath;
      this.B = B;
      this.D = D;
      this.C = C;
      this.lastEigenUpdateGeneration = lastEigenUpdateGeneration;
    }

    @Override
    public State copy() {
      CMAESState cmaesState = new CMAESState(
          getIterations(),
          getBirths(),
          getFitnessEvaluations(),
          getElapsedMillis(),
          stepSize,
          Arrays.copyOf(distributionMean, distributionMean.length),
          Arrays.copyOf(sEvolutionPath, sEvolutionPath.length),
          Arrays.copyOf(cEvolutionPath, cEvolutionPath.length),
          B.copy(),
          D.copy(),
          C.copy(),
          lastEigenUpdateGeneration
      );
      for (int i = 0; i < lambda; i++) {
        System.arraycopy(zK[i], 0, cmaesState.zK[i], 0, zK[i].length);
        System.arraycopy(yK[i], 0, cmaesState.yK[i], 0, yK[i].length);
        System.arraycopy(xK[i], 0, cmaesState.xK[i], 0, xK[i].length);
      }
      return cmaesState;
    }

  }

  private void eigenDecomposition(CMAESState state) {
    L.fine(String.format("Eigen decomposition of covariance matrix (i=%d)", state.getIterations()));
    EigenDecomposition eig = new EigenDecomposition(state.C);
    RealMatrix B = eig.getV();
    RealMatrix D = eig.getD();
    for (int i = 0; i < n; i++) {
      if (D.getEntry(i, i) < 0) {
        L.warning("An eigenvalue has become negative");
        D.setEntry(i, i, 0d);
      }
      D.setEntry(i, i, Math.sqrt(D.getEntry(i, i)));
    }
    state.B = B;
    state.D = D;
    state.lastEigenUpdateGeneration = state.getIterations();
  }

  @Override
  protected Collection<Individual<List<Double>, S, F>> initPopulation(
      Function<S, F> fitnessFunction,
      RandomGenerator random,
      ExecutorService executor,
      State state
  ) throws ExecutionException, InterruptedException {
    CMAESState cmaesState = (CMAESState) state;
    List<Double> randomGenotype = genotypeFactory.build(1, random).get(0);
    cmaesState.distributionMean = randomGenotype.stream().mapToDouble(d -> d).toArray();
    return samplePopulation(fitnessFunction, random, executor, cmaesState);
  }

  @Override
  protected Collection<Individual<List<Double>, S, F>> updatePopulation(
      PartiallyOrderedCollection<Individual<List<Double>, S, F>> orderedPopulation,
      Function<S, F> fitnessFunction,
      RandomGenerator random,
      ExecutorService executor,
      State state
  ) throws ExecutionException, InterruptedException {
    CMAESState cmaesState = (CMAESState) state;
    updateDistribution(orderedPopulation, cmaesState);
    // update B and D from C
    if ((state.getIterations() - cmaesState.lastEigenUpdateGeneration) > (1d / (c1 + cMu) / n / 10d)) {
      eigenDecomposition(cmaesState);
    }
    // flat fitness case
    if (orderedPopulation.firsts().size() >= Math.ceil(0.7 * lambda)) {
      (cmaesState).stepSize *= Math.exp(0.2 + cSigma / dSigma);
      L.warning("Flat fitness, consider reformulating the objective");
    }
    return samplePopulation(fitnessFunction, random, executor, cmaesState);
  }

  @Override
  protected State initState() {
    return new CMAESState();
  }

  private List<Individual<List<Double>, S, F>> samplePopulation(
      Function<S, F> fitnessFunction,
      RandomGenerator random,
      ExecutorService executor,
      CMAESState state
  ) throws ExecutionException, InterruptedException {
    List<List<Double>> genotypes = IntStream.range(0, lambda).mapToObj(k -> {
      state.zK[lambda] = IntStream.range(0, n).mapToDouble(i -> random.nextGaussian()).toArray();
      state.yK[lambda] = state.B.preMultiply(state.D.preMultiply(state.zK[lambda]));
      state.xK[lambda] = IntStream.range(0, n).mapToDouble(i -> state.distributionMean[i] + state.stepSize * state.yK[lambda][i]).toArray();
      return Arrays.stream(state.xK[lambda]).boxed().toList();
    }).toList();
    return AbstractIterativeEvolver.map(genotypes, List.of(), solutionMapper, fitnessFunction, executor, state);
  }

  private void updateDistribution(final PartiallyOrderedCollection<Individual<List<Double>, S, F>> population, final CMAESState state) {
    // best mu ranked points
    List<List<Double>> bestMuGenotypes = population.all().stream()
        .sorted(individualComparator.comparator()).limit(mu)
        .map(Individual::genotype)
        .toList();
    // TODO find more efficient way of implementing this, maybe through annotated individuals
    int[] bestMuIndexes = bestMuGenotypes.stream().mapToInt(l -> findInArrayOfArrays(l, state.xK)).toArray();
    double[][] xMu = new double[mu][n];
    double[][] yMu = new double[mu][n];
    double[][] zMu = new double[mu][n];
    for (int i = 0; i < mu; i++) {
      xMu[i] = state.xK[bestMuIndexes[i]];
      yMu[i] = state.yK[bestMuIndexes[i]];
      zMu[i] = state.zK[bestMuIndexes[i]];
    }

    // selection and recombination
    double[] updatedDistributionMean = IntStream.range(0, n).mapToDouble(j -> IntStream.range(0, mu).mapToDouble(i -> weights[i] * xMu[i][j]).sum()).toArray();
    double[] yW = IntStream.range(0, n).mapToDouble(j -> (updatedDistributionMean[j] - state.distributionMean[j]) / state.stepSize).toArray();
    state.distributionMean = updatedDistributionMean;

    // step size control
    double[] zM = IntStream.range(0, n).mapToDouble(j -> IntStream.range(0, mu).mapToDouble(i -> weights[i] * zMu[i][j]).sum()).toArray();
    double[] bzM = state.B.preMultiply(zM);
    state.sEvolutionPath = IntStream.range(0, n).mapToDouble(i -> (1d - cSigma) * state.sEvolutionPath[i] + (Math.sqrt(cSigma * (2d - cSigma) * muEff)) * bzM[i]).toArray();
    double psNorm = Math.sqrt(Arrays.stream(state.sEvolutionPath).map(d -> d * d).sum());
    state.stepSize *= Math.exp((cSigma / dSigma) * ((psNorm / chiN) - 1));

    // covariance matrix adaptation
    int hSigma = psNorm / Math.sqrt(1 - Math.pow((1d - cSigma), 2 * state.getIterations())) / chiN < (1.4 + 2d / (n + 1)) ? 1 : 0;
    state.cEvolutionPath = IntStream.range(0, n).mapToDouble(i -> (1 - cc) * state.cEvolutionPath[i] + hSigma * Math.sqrt(cc * (2 - cc) * muEff) * yW[i]).toArray();
    double deltaH = (1 - hSigma) * cc * (2 - cc);
    IntStream.range(0, n).forEach(i -> IntStream.range(0, i + 1).forEach(j -> {
      double cij = (1 + c1 * deltaH - c1 - cMu) * state.C.getEntry(i, j) +
          c1 * state.cEvolutionPath[i] * state.cEvolutionPath[j] +
          cMu * IntStream.range(0, mu).mapToDouble(k -> weights[k] * yMu[k][i] * yMu[k][j]).sum();
      state.C.setEntry(i, j, cij);
      state.C.setEntry(j, i, cij);
    }));

  }

  private static int findInArrayOfArrays(List<Double> l, double[][] arrays) {
    for (int i = 0; i < arrays.length; i++) {
      if (genotypeEquals(l, arrays[i])) {
        return i;
      }
    }
    return -1;
  }

  private static boolean genotypeEquals(List<Double> l, double[] d) {
    for (int i = 0; i < l.size(); i++) {
      if (!doubleEquals(l.get(i), d[i])) {
        return false;
      }
    }
    return true;
  }

  private static boolean doubleEquals(double d1, double d2) {
    return Math.abs(d1 - d2) < 0.0001;
  }

}
