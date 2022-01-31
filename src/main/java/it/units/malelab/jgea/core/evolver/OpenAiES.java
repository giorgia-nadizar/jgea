package it.units.malelab.jgea.core.evolver;

import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.order.PartiallyOrderedCollection;
import it.units.malelab.jgea.core.util.AdamOptimizer;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.function.Function;
import java.util.random.RandomGenerator;
import java.util.stream.IntStream;

// https://github.com/snolfi/evorobotpy2/blob/master/bin/openaies.py
// https://bacrobotics.com/Chapter6.html

public class OpenAiES<S,F> extends AbstractIterativeEvolver<List<Double>, S, F> {

  private final int lambda;
  private final double sigma;
  private final Mutation<List<Double>> mutation;

  public OpenAiES(Function<? super List<Double>, ? extends S> solutionMapper, Factory<? extends List<Double>> genotypeFactory, PartialComparator<? super Individual<List<Double>, S, F>> individualComparator, int lambda, double sigma) {
    super(solutionMapper, genotypeFactory, individualComparator);
    this.sigma = sigma;
    this.lambda = lambda;
    mutation = new GaussianMutation(1);
  }

  private class OpenAiEsState extends State {

    List<Double> centralGenotype;
    final double[][] noise = new double[lambda][];
    private List<Individual<List<Double>, S, F>> individuals;

    final AdamOptimizer adam;

    public OpenAiEsState() {
      adam = new AdamOptimizer();
    }

    public OpenAiEsState(int iterations, int births, int fitnessEvaluations, long elapsedMillis, List<Double> centralGenotype, AdamOptimizer adam) {
      super(iterations, births, fitnessEvaluations, elapsedMillis);
      this.centralGenotype = centralGenotype;
      this.adam = adam;
    }

    private double[] estimateGradient() {
      Function<Integer, Individual<List<Double>, S, F>> mapper = i -> individuals.get(i);
      Comparator<Integer> comparator = individualComparator.comparing(mapper).comparator();
      List<Double> normalizedRanks = IntStream.range(0, 2 * lambda).boxed().sorted(comparator)
          .map(d -> (double) d / (2 * lambda - 1) - 0.5).toList();
      List<Double> u = IntStream.range(0, lambda)
          .mapToObj(i -> normalizedRanks.get(i) - normalizedRanks.get(lambda + i)).toList();
      return IntStream.range(0, centralGenotype.size()).mapToDouble(i ->
          IntStream.range(0, lambda).mapToDouble(d -> u.get(d) * noise[d][i]).sum() / lambda
      ).toArray();
    }

    @Override
    public State copy() {
      OpenAiEsState openAiEsState = new OpenAiEsState(
          getIterations(),
          getBirths(),
          getFitnessEvaluations(),
          getElapsedMillis(),
          centralGenotype.stream().toList(),
          adam.copy()
      );
      for (int i = 0; i < lambda; i++) {
        System.arraycopy(noise[i], 0, openAiEsState.noise[i], 0, noise[i].length);
      }
      openAiEsState.individuals = new ArrayList<>(individuals);
      return openAiEsState;
    }
  }

  @Override
  protected State initState() {
    return new OpenAiEsState();
  }

  @Override
  protected Collection<Individual<List<Double>, S, F>> initPopulation(Function<S, F> fitnessFunction, RandomGenerator random, ExecutorService executor, State state) throws ExecutionException, InterruptedException {
    OpenAiEsState openAiEsState = (OpenAiEsState) state;
    openAiEsState.centralGenotype = genotypeFactory.build(1, random).get(0);
    return updateDistribution(fitnessFunction, random, executor, openAiEsState);
  }

  private Collection<Individual<List<Double>, S, F>> updateDistribution(
      Function<S, F> fitnessFunction,
      RandomGenerator random,
      ExecutorService executor, OpenAiEsState state) throws ExecutionException, InterruptedException {
    List<Double> zeroList = IntStream.range(0, state.centralGenotype.size()).mapToObj(i -> 0d).toList();
    IntStream.range(0, lambda).forEach(i ->
        state.noise[i] = mutation.mutate(zeroList, random).stream().mapToDouble(d -> d).toArray()
    );
    List<List<Double>> mutatedGenotypes = new ArrayList<>(IntStream.range(0, lambda).mapToObj(i ->
        IntStream.range(0, state.centralGenotype.size()).mapToObj(d -> state.centralGenotype.get(d) + sigma * state.noise[i][d]).toList()
    ).toList());
    mutatedGenotypes.addAll(IntStream.range(0, lambda).mapToObj(i ->
        IntStream.range(0, state.centralGenotype.size()).mapToObj(d -> state.centralGenotype.get(d) - sigma * state.noise[i][d]).toList()
    ).toList());
    mutatedGenotypes.add(state.centralGenotype);
    List<Individual<List<Double>, S, F>> individuals = AbstractIterativeEvolver.map(mutatedGenotypes, List.of(), solutionMapper, fitnessFunction, executor, state);
    state.individuals = individuals.stream().limit(2L * lambda).toList();
    return individuals;
  }

  @Override
  protected Collection<Individual<List<Double>, S, F>> updatePopulation(PartiallyOrderedCollection<Individual<List<Double>, S, F>> orderedPopulation, Function<S, F> fitnessFunction, RandomGenerator random, ExecutorService executor, State state) throws ExecutionException, InterruptedException {
    OpenAiEsState openAiEsState = (OpenAiEsState) state;
    double[] optimizedGradient = openAiEsState.adam.optimizeStep(openAiEsState.estimateGradient());
    openAiEsState.centralGenotype = IntStream.range(0, openAiEsState.centralGenotype.size()).mapToObj(i ->
        openAiEsState.centralGenotype.get(i) + optimizedGradient[i]
    ).toList();
    return updateDistribution(fitnessFunction, random, executor, openAiEsState);
  }

}
