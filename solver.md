---
layout: default
title: Solver
parent: Structure and components
nav_order: 2
---

# Solver
A problem can be solved by an implementation of the ``Solver`` interface, which is responsible for providing the caller with a collection of solutions upon the invocation of its method ``solve()``.
```java
public interface Solver<P extends Problem<S>, S> {
  Collection<S> solve(
      P problem,
      RandomGenerator random,
      ExecutorService executor
  ) throws SolverException;
}
```

We highlight that, in general, a ``Solver`` might not be suitable for solving all possible problems.
Therefore, we introduce the generic parameter ``P`` to indicate the subset of problems a ``Solver`` can tackle.

We also remark that ``solve()`` takes two additional elements besides the ``P problem``: a ``RandomGenerator`` and an ``ExecutorService``, since a ``Solver`` can be non-deterministic and capable of exploiting concurrency.
The contract for the ``solve()`` method is that the passed ``RandomGenerator`` instance will be used for all the random choices, hence allowing for _repeatability_ of the experimentation (_reproducibility_, instead, might not always be guaranteed due to concurrency).
Similarly, the contract states that the ``ExecutorService`` instance will be used for distributing computation (usually, of the fitness of candidate solutions) across different workers of the executor.
In fact, population-based optimization methods are naturally suited for exploiting parallel computation (see, e.g., the large taxonomy of parallel methods already developed more than 20 years ago), and, even though we design JGEA aiming at clarity and ease of use, we also take into consideration efficiency.

We designed the ``solve()`` method of the ``Solver`` interface in order to model the stateless nature of the solver with respect to its capability, i.e., to solve problems.
Namely, since both the ``RandomGenerator`` and the ``ExecutorService`` are provided (besides the problem itself) when ``solve()`` is invoked, different problems may in principle be solved at the same time by the same instance of the solver.
