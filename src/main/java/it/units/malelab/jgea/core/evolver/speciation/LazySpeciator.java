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

package it.units.malelab.jgea.core.evolver.speciation;

import it.units.malelab.jgea.core.evolver.Evolver;
import it.units.malelab.jgea.core.order.PartiallyOrderedCollection;
import it.units.malelab.jgea.distance.Distance;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * @author federico
 */
public class LazySpeciator<G, S, F> implements SpeciatedEvolver.Speciator<Evolver.Individual<G, S, F>> {
  private final Distance<Evolver.Individual<G, S, F>> distance;
  private final double distanceThreshold;

  public LazySpeciator(Distance<Evolver.Individual<G, S, F>> distance, double distanceThreshold) {
    this.distance = distance;
    this.distanceThreshold = distanceThreshold;
  }

  @Override
  public Collection<SpeciatedEvolver.Species<Evolver.Individual<G, S, F>>> speciate(PartiallyOrderedCollection<Evolver.Individual<G, S, F>> population) {
    List<List<Evolver.Individual<G, S, F>>> clusters = new ArrayList<>();
    for (Evolver.Individual<G, S, F> individual : population.all()) {
      List<Double> distances = clusters.stream()
          .map(c -> distance.apply(individual, c.get(0))).toList();
      if (distances.isEmpty()) {
        List<Evolver.Individual<G, S, F>> cluster = new ArrayList<>();
        cluster.add(individual);
        clusters.add(cluster);
      } else {
        int closestIndex = 0;
        for (int i = 1; i < distances.size(); i++) {
          if (distances.get(i) < distances.get(closestIndex)) {
            closestIndex = i;
          }
        }
        if (distances.get(closestIndex) < distanceThreshold) {
          clusters.get(closestIndex).add(individual);
        } else {
          List<Evolver.Individual<G, S, F>> cluster = new ArrayList<>();
          cluster.add(individual);
          clusters.add(cluster);
        }
      }
    }
    return clusters.stream()
        .map(c -> new SpeciatedEvolver.Species<>(c, c.get(0)))
        .toList();
  }

}
