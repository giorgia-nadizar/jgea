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

package it.units.malelab.jgea.core.distance;

import com.google.common.collect.Sets;

import java.util.Set;

/**
 * @author eric
 */
public class Jaccard implements Distance<Set<?>> {
  @Override
  public Double apply(Set<?> s1, Set<?> s2) {
    if (s1.isEmpty() && s2.isEmpty()) {
      return 0d;
    }
    return 1d - (double) Sets.intersection(s1, s2).size() / (double) Sets.union(s1, s2).size();
  }
}
