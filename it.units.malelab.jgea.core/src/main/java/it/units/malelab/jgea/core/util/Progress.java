/*
 * Copyright 2022 eric
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

package it.units.malelab.jgea.core.util;

public record Progress(
    Number start,
    Number end,
    Number current
) {
  public static Progress NA = new Progress(0, 0, 0);

  public double rate() {
    return Math.min(
        1d,
        Math.max(0d, current.doubleValue() - start.doubleValue()) / (end.doubleValue() - start.doubleValue())
    );
  }

}
