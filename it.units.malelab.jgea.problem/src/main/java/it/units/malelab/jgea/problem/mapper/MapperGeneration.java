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

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package it.units.malelab.jgea.problem.mapper;

import it.units.malelab.jgea.core.ProblemWithValidation;
import it.units.malelab.jgea.core.order.ParetoDominance;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.representation.grammar.Grammar;
import it.units.malelab.jgea.core.representation.grammar.GrammarBasedProblem;
import it.units.malelab.jgea.core.representation.tree.Tree;
import it.units.malelab.jgea.core.util.Pair;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.function.Function;

/**
 * @author eric
 */
@SuppressWarnings("rawtypes")
public class MapperGeneration implements GrammarBasedProblem<String, Pair<Tree<Element>, Tree<Element>>>,
    ProblemWithValidation<Pair<Tree<Element>, Tree<Element>>, List<Double>> {

  private final Grammar<String> grammar;
  private final FitnessFunction learningFitnessFunction;
  private final FitnessFunction validationFitnessFunction;
  private final int dimensionality;

  public MapperGeneration(
      List<EnhancedProblem> learningProblems,
      int learningGenotypeSize,
      int learningN,
      int learningMaxMappingDepth,
      List<FitnessFunction.Property> learningProperties,
      List<EnhancedProblem> validationProblems,
      int validationN,
      int validationGenotypeSize,
      int validationMaxMappingDepth,
      List<FitnessFunction.Property> validationProperties,
      long seed
  ) throws IOException {
    this.grammar = Grammar.fromFile(new File("grammars/mapper.bnf"));
    learningFitnessFunction = new FitnessFunction(
        learningProblems,
        learningGenotypeSize,
        learningN,
        learningMaxMappingDepth,
        learningProperties,
        seed
    );
    validationFitnessFunction = new FitnessFunction(
        validationProblems,
        validationGenotypeSize,
        validationN,
        validationMaxMappingDepth,
        validationProperties,
        seed
    );
    dimensionality = learningProperties.size();
  }

  @Override
  public Grammar<String> getGrammar() {
    return grammar;
  }

  @Override
  public Function<Tree<String>, Pair<Tree<Element>, Tree<Element>>> getSolutionMapper() {
    return (Tree<String> rawMappingTree) -> {
      Tree<Element> optionChooser = MapperUtils.transform(rawMappingTree.child(0));
      Tree<Element> genoAssigner = MapperUtils.transform(rawMappingTree.child(1));
      return Pair.of(optionChooser, genoAssigner);
    };
  }

  @Override
  public PartialComparator<List<Double>> qualityComparator() {
    return ParetoDominance.build(Double.class, dimensionality);
  }

  @Override
  public Function<Pair<Tree<Element>, Tree<Element>>, List<Double>> qualityFunction() {
    return learningFitnessFunction;
  }

  @Override
  public Function<Pair<Tree<Element>, Tree<Element>>, List<Double>> validationQualityFunction() {
    return validationFitnessFunction;
  }

}
