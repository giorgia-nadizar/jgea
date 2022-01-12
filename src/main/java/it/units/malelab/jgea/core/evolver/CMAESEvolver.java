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

import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.random.RandomGenerator;

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

  protected class CMAESState extends State {
    // Step-size
    private double stepSize = 0.5;
    // Mean value of the search distribution
    private double[] distributionMean = new double[n];
    // Evolution path for step-size
    private double[] sigmaEvolutionPath = new double[n];
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

    // Last generation when the eigendecomposition was calculated
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
        double[] sigmaEvolutionPath,
        double[] cEvolutionPath,
        RealMatrix B,
        RealMatrix D,
        RealMatrix C,
        int lastEigenUpdateGeneration
    ) {
      super(iterations, births, fitnessEvaluations, elapsedMillis);
      this.stepSize = stepSize;
      this.distributionMean = distributionMean;
      this.sigmaEvolutionPath = sigmaEvolutionPath;
      this.cEvolutionPath = cEvolutionPath;
      this.B = B;
      this.D = D;
      this.C = C;
      this.lastEigenUpdateGeneration = lastEigenUpdateGeneration;
    }

    @Override
    public State copy() {
      return new CMAESState(
          getIterations(),
          getBirths(),
          getFitnessEvaluations(),
          getElapsedMillis(),
          stepSize,
          Arrays.copyOf(distributionMean, distributionMean.length),
          Arrays.copyOf(sigmaEvolutionPath, sigmaEvolutionPath.length),
          Arrays.copyOf(cEvolutionPath, cEvolutionPath.length),
          B.copy(),
          D.copy(),
          C.copy(),
          lastEigenUpdateGeneration
      );
    }

  }

  private void eigenDecomposition(CMAESState state) {
    L.fine(String.format("Eigen decomposition of covariance matrix (i=%d)", state.getIterations()));
    state.lastEigenUpdateGeneration = state.getIterations();
    EigenDecomposition eig = new EigenDecomposition(state.C);
    // normalized eigenvectors
    RealMatrix B = eig.getV();
    RealMatrix D = eig.getD();
    for (int i = 0; i < n; i++) {
      // numerical problem?
      if (D.getEntry(i, i) < 0) {
        L.warning("An eigenvalue has become negative");
        D.setEntry(i, i, 0d);
      }
      // D contains standard deviations now
      D.setEntry(i, i, Math.sqrt(D.getEntry(i, i)));
    }
    state.B = B;
    state.D = D;
  }

  @Override
  protected Collection<Individual<List<Double>, S, F>> initPopulation(
      Function<S, F> fitnessFunction,
      RandomGenerator random,
      ExecutorService executor,
      State state
  ) throws ExecutionException, InterruptedException {
    // objective variables initial point
    List<Double> point = genotypeFactory.build(1, random).get(0);
    ((CMAESState) state).distributionMean = point.stream().mapToDouble(d -> d).toArray();
    return samplePopulation(fitnessFunction, random, executor, (CMAESState) state);
  }

  @Override
  protected Collection<Individual<List<Double>, S, F>> updatePopulation(
      PartiallyOrderedCollection<Individual<List<Double>, S, F>> orderedPopulation,
      Function<S, F> fitnessFunction,
      RandomGenerator random,
      ExecutorService executor,
      State state
  ) throws ExecutionException, InterruptedException {
    updateDistribution(orderedPopulation, (CMAESState) state);
    // update B and D from C
    if ((state.getIterations() - ((CMAESState) state).lastEigenUpdateGeneration) > (1d / (c1 + cMu) / n / 10d)) {
      eigenDecomposition((CMAESState) state);
    }
    // escape flat fitness, or better terminate?
    if (orderedPopulation.firsts().size() >= Math.ceil(0.7 * lambda)) {
      double stepSize = ((CMAESState) state).stepSize;
      stepSize *= Math.exp(0.2 + cSigma / dSigma);
      ((CMAESState) state).stepSize = stepSize;
      L.warning("Flat fitness, consider reformulating the objective");
    }
    return samplePopulation(fitnessFunction, random, executor, (CMAESState) state);
  }

  @Override
  protected State initState() {
    return new CMAESState();
  }

  protected List<Individual<List<Double>, S, F>> samplePopulation(
      Function<S, F> fitnessFunction,
      RandomGenerator random,
      ExecutorService executor,
      CMAESState state
  ) throws ExecutionException, InterruptedException {
    List<List<Double>> genotypes = new ArrayList<>();
    while (genotypes.size() < lambda) {
      // new point
      List<Double> genotype = new ArrayList<>(Collections.nCopies(n, 0d));
      // z ∼ N (0, I) normally distributed vector
      double[] arz = new double[n];
      for (int i = 0; i < n; i++) {
        arz[i] = random.nextGaussian();
      }
      // y ∼ N (0, C) y = (B*(D*z))
      double[] ary = state.B.preMultiply(state.D.preMultiply(arz));
      for (int i = 0; i < n; i++) {
        // add mutation (sigma*B*(D*z))
        genotype.set(i, state.distributionMean[i] + state.stepSize * ary[i]);
      }
      genotypes.add(genotype);
    }
    return AbstractIterativeEvolver.map(genotypes, List.of(), solutionMapper, fitnessFunction, executor, state);
  }

  protected void updateDistribution(
      final PartiallyOrderedCollection<Individual<List<Double>, S, F>> population,
      final CMAESState state
  ) {
    // best mu ranked points
    List<Individual<List<Double>, S, F>> bestMuPoints = population
        .all()
        .stream()
        .sorted(individualComparator.comparator())
        .limit(mu)
        .toList();
    double[] distrMean = state.distributionMean;
    double[] oldDistrMean = Arrays.copyOf(distrMean, distrMean.length);
    double[] artmp = new double[n];
    // recombination
    for (int i = 0; i < n; i++) {
      distrMean[i] = 0;
      for (int j = 0; j < mu; j++) {
        distrMean[i] += weights[j] * bestMuPoints.get(j).genotype().get(i);
      }
      artmp[i] = (distrMean[i] - oldDistrMean[i]) / state.stepSize;
    }
    state.distributionMean = distrMean;

    // (D^-1*B'*(xmean-xold)/sigma)
    double[] zmean = MatrixUtils.inverse(state.D).preMultiply(state.B.transpose().preMultiply(artmp));

    // cumulation: update evolution paths
    double[] Bzmean = state.B.preMultiply(zmean);
    double[] sEvolutionPath = state.sigmaEvolutionPath;
    for (int i = 0; i < n; i++) {
      sEvolutionPath[i] = (1d - cSigma) * sEvolutionPath[i] + (Math.sqrt(cSigma * (2d - cSigma) * muEff)) * Bzmean[i];
    }
    state.sigmaEvolutionPath = sEvolutionPath;

    // calculate step-size evolution path norm
    double psNorm = 0.0;
    for (int i = 0; i < n; i++) {
      psNorm += sEvolutionPath[i] * sEvolutionPath[i];
    }
    psNorm = Math.sqrt(psNorm);

    // Heaviside function
    int hsig = 0;
    if (psNorm / Math.sqrt(1 - Math.pow((1d - cSigma), 2 * state.getIterations())) / chiN < (1.4 + 2d / (n + 1))) {
      hsig = 1;
    }

    double[] CEvolutionPath = state.cEvolutionPath;
    for (int i = 0; i < n; i++) {
      CEvolutionPath[i] = (1 - cc) * CEvolutionPath[i] + hsig * Math.sqrt(cc * (2 - cc) * muEff) * artmp[i];
    }
    state.cEvolutionPath = CEvolutionPath;

    RealMatrix C = state.C;
    // adapt covariance matrix C
    for (int i = 0; i < n; i++) {
      for (int j = 0; j <= i; j++) {
        double rankOneUpdate = CEvolutionPath[i] * CEvolutionPath[j] + (1 - hsig) * cc * (2 - cc) * C.getEntry(i, j);
        double rankMuUpdate = 0d;
        for (int k = 0; k < mu; k++) {
          rankMuUpdate += weights[k] * ((bestMuPoints.get(k)
              .genotype()
              .get(i) - oldDistrMean[i]) / state.stepSize) * ((bestMuPoints.get(k)
              .genotype()
              .get(j) - oldDistrMean[j]) / state.stepSize);
        }
        C.setEntry(i, j, (1 - c1 - cMu) * C.getEntry(i, j) + c1 * rankOneUpdate + cMu * rankMuUpdate);
        if (i != j) {
          // force symmetric matrix
          C.setEntry(j, i, C.getEntry(i, j));
        }
      }
    }
    state.C = C;

    // adapt step size sigma
    double stepSize = state.stepSize;
    stepSize *= Math.exp((cSigma / dSigma) * ((psNorm / chiN) - 1));
    state.stepSize = stepSize;
  }
}
