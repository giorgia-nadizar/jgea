package it.units.malelab.jgea.core.evolver;

import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.order.MapElites;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.order.PartiallyOrderedCollection;

import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.function.Function;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MapElitesEvolver<G, S, F> extends AbstractIterativeEvolver<G, S, F> {

  private final int populationSize;
  private final int batchSize;
  private final Mutation<G> mutation;

  private final Function<Individual<G, S, F>, List<Double>> featuresExtractor;
  private final List<MapElites.Feature> features;

  public MapElitesEvolver(
      Function<? super G, ? extends S> solutionMapper,
      Factory<? extends G> genotypeFactory,
      PartialComparator<? super Individual<G, S, F>> individualComparator,
      int populationSize,
      int batchSize,
      Mutation<G> mutation,
      Function<Individual<G, S, F>, List<Double>> featuresExtractor,
      List<Integer> featuresSizes,
      List<Double> featuresMinValues,
      List<Double> featuresMaxValues
  ) {
    super(solutionMapper, genotypeFactory, individualComparator);
    this.populationSize = populationSize;
    this.batchSize = batchSize;
    this.mutation = mutation;
    this.featuresExtractor = featuresExtractor;
    features = MapElites.buildFeatures(featuresSizes, featuresMinValues, featuresMaxValues);
  }

  private class MapElitesState extends State {

    private final MapElites<Individual<G, S, F>> mapElites;

    public MapElitesState() {
      mapElites = new MapElites<>(features, true, featuresExtractor, individualComparator);
    }

    public MapElitesState(int iterations, int births, int fitnessEvaluations, long elapsedMillis, MapElites<Individual<G, S, F>> mapElites) {
      super(iterations, births, fitnessEvaluations, elapsedMillis);
      this.mapElites = mapElites;
    }

    @Override
    public State copy() {
      return new MapElitesState(
          getIterations(),
          getBirths(),
          getFitnessEvaluations(),
          getElapsedMillis(),
          mapElites.copy()
      );
    }
  }

  @Override
  protected State initState() {
    return new MapElitesState();
  }

  @Override
  protected Collection<Individual<G, S, F>> initPopulation(Function<S, F> fitnessFunction, RandomGenerator random, ExecutorService executor, State state) throws ExecutionException, InterruptedException {
    Collection<Individual<G, S, F>> population = initPopulation(populationSize, fitnessFunction, random, executor, state);
    ((MapElitesState) state).mapElites.addAll(population);
    return population;
  }

  @Override
  protected Collection<Individual<G, S, F>> updatePopulation(PartiallyOrderedCollection<Individual<G, S, F>> orderedPopulation, Function<S, F> fitnessFunction, RandomGenerator random, ExecutorService executor, State state) throws ExecutionException, InterruptedException {
    List<Individual<G, S, F>> allGenotypes = orderedPopulation.all().stream().filter(Objects::nonNull).toList();
    Collection<G> offspringGenotypes = IntStream.range(0, batchSize)
        .mapToObj(i -> mutation.mutate(allGenotypes.get(random.nextInt(allGenotypes.size())).genotype(), random))
        .collect(Collectors.toList());
    Collection<Individual<G, S, F>> offspringIndividuals = map(offspringGenotypes, List.of(), solutionMapper, fitnessFunction, executor, state);
    ((MapElitesState) state).mapElites.addAll(offspringIndividuals);
    return ((MapElitesState) state).mapElites.all();
  }

}
