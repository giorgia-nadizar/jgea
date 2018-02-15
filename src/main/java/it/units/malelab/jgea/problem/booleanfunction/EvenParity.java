/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package it.units.malelab.jgea.problem.booleanfunction;

import it.units.malelab.jgea.core.Node;
import it.units.malelab.jgea.core.fitness.BooleanFunctionFitness;
import it.units.malelab.jgea.core.mapper.BoundMapper;
import it.units.malelab.jgea.core.mapper.DeterministicMapper;
import it.units.malelab.jgea.grammarbased.Grammar;
import it.units.malelab.jgea.grammarbased.GrammarBasedProblem;
import it.units.malelab.jgea.problem.booleanfunction.element.Element;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 *
 * @author eric
 */
public class EvenParity implements GrammarBasedProblem<String, List<Node<Element>>, Double> {

  private static class TargetFunction implements BooleanFunctionFitness.TargetFunction {

    private final String[] varNames;

    public TargetFunction(int size) {
      varNames = new String[size];
      for (int i = 0; i < size; i++) {
        varNames[i] = "b" + i;
      }
    }

    @Override
    public boolean[] compute(boolean... arguments) {
      int count = 0;
      for (boolean argument : arguments) {
        count = count + (argument ? 1 : 0);
      }
      return new boolean[]{(count % 2) == 1};
    }

    @Override
    public String[] varNames() {
      return varNames;
    }

  }

  private final Grammar<String> grammar;
  private final DeterministicMapper<Node<String>, List<Node<Element>>> solutionMapper;
  private final BoundMapper<List<Node<Element>>, Double> fitnessMapper;

  public EvenParity(final int size) throws IOException {
    grammar = Grammar.fromFile(new File("grammars/boolean-parity-var.bnf"));
    List<List<String>> vars = new ArrayList<>();
    for (int i = 0; i < size; i++) {
      vars.add(Collections.singletonList("b" + i));
    }
    grammar.getRules().put("<v>", vars);
    solutionMapper = new FormulaMapper();
    TargetFunction targetFunction = new TargetFunction(size);
    fitnessMapper = new BooleanFunctionFitness(
            targetFunction,
            BooleanUtils.buildCompleteObservations(targetFunction.varNames)
    );
  }

  @Override
  public Grammar<String> getGrammar() {
    return grammar;
  }

  @Override
  public DeterministicMapper<Node<String>, List<Node<Element>>> getSolutionMapper() {
    return solutionMapper;
  }

  @Override
  public BoundMapper<List<Node<Element>>, Double> getFitnessMapper() {
    return fitnessMapper;
  }

}
