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

package it.units.malelab.jgea.core.order;

import it.units.malelab.jgea.core.util.Sized;

import java.util.Collection;

/**
 * @author eric
 */
public interface PartiallyOrderedCollection<T> extends Sized {
  void add(T t);

  Collection<T> all();

  Collection<T> firsts();

  Collection<T> lasts();

  boolean remove(T t);

  @Override
  default int size() {
    return all().size();
  }
}
