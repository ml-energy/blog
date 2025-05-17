---
date: 2023-08-01
authors:
  - jaywonchung
categories:
  - energy
  - research
links:
  - Zeus NSDI'23 paper: https://www.usenix.org/conference/nsdi23/presentation/you
---

# ML Energy, Performance, and Accuracy

Zeus's [batch size optimizer](https://ml.energy/zeus/optimize/batch_size_optimizer) changes the model's training batch size to optimize time and energy consumption, but does that hurt the model's final quality?
Short answer: No.
Let's look into how in today's post.

<!-- more -->

[Zeus](https://ml.energy/zeus) is the first energy measurement and optimization framework for Deep Learning.
In our [NSDI paper](https://www.usenix.org/conference/nsdi23/presentation/you), we introduce an algorithm to automatically optimize a training job's time and energy consumption by finding the best the batch size and GPU power limit.

## Two Categories of MLSys

Generally speaking, there are two big categories of systems that support machine learning:

1. Training or inference quality (as measured by an appropriate ML metric like accuracy or perplexity) is the same compared to not using the system.
2. It can be different (Usually worse quality, but the system is faster in some way or uses less memory).

## Ensuring Model Quality

When you build systems for machine learning, one way to ensure that your system is of type 1 (no quality change) is to make sure it does the exact same computations that would have happened without the system.
Data Parallel training on multiple GPUs is a good example.
Even if you split the training batch across GPUs and compute the gradient separately, due to the arithmetic property of gradient computation, doing an AllReduce at the end recovers the gradient that would have been computed if you ran the entire batch on a hypothetical gigantic GPU.[^1]

However, ensuring perfect computational equivalence is sometimes difficult and limits the scope of optimization.
Thus, another emerging direction, is to allow system to do whatever in the end, but just make sure they **reach the final target metric** in the end.
Time-To-Accuracy[^2] is a very good example of this, which measures the wall clock time it took for a training system (regardless of whatever it does to the model) to reach the target validation accuracy of 93% on ImageNet.
They have a similar metric for inference speed, too.

## Which Category is Zeus?

Zeus is a type 1 MLSys because it minimizes a linear combination of Energy-To-Accuracy (ETA) and Time-To-Accuracy (TTA), where the user selects the target training validation metric.
If the model does not reach the target metric, ETA and TTA are both infinity -- definitely not optimal.

How is this optimization feasible?
Changing the GPU's power limit does not change what is computed by the GPU, and hence there cannot be any model quality change.
Changing the training batch size does change the computation done by the GPU (more importantly model convergence itself), but the room for optimization comes from the fact that there isn't just one batch size that can reach the target metric; there can be a couple, and Zeus will automatically find the best batch size among them.[^3]

All in all, Zeus does not hurt model quality by design.

## What about Performance?

Performance, or more specifically training throughput, depends on the model and what you compare against.
So there's a Pareto frontier of DNN training time and energy on GPUs.
If the initial batch size and GPU power limit you've been using was *not* on the Pareto frontier, Zeus will reduce *both* training time and energy consumption by finding the optimal batch size and GPU power limit.
However, if you happened to be training with the time-optimal configuration, but you still wanted to reduce energy consumption, Zeus will have to trade performance to reduce energy consumption.
That is just how Pareto frontiers work -- On the frontier, there is no way to reduce one dimension without sacrificing the other dimensions.

## A Multi-Objective Optimization View

When you want to optimize more than one competing objectives simultaneously, you have a [Multi-Objective Optimization problem](https://en.wikipedia.org/wiki/Multi-objective_optimization).
As you can see, in general terms, we're facing a three-way multi-objective optimization problem, involving training time, model quality, and energy consumption.

Zeus's approach was to (1) fix model quality (to what the user specified), and (2) perform [linear scalarization](https://en.wikipedia.org/wiki/Multi-objective_optimization#Scalarizing) for time and energy to make the optimization problem solvable.
Speaking of which, for (2), why can't we use the $\epsilon$-constraint method?
Well, we can, and thus [`GlobalPowerLimitOptimizer`](https://ml.energy/zeus/getting_started/#globalpowerlimitoptimizer) supports both linear scalarization ([`ZeusCost`](https://ml.energy/zeus/reference/optimizer/power_limit/#zeus.optimizer.power_limit.ZeusCost)) and $\epsilon$-constraint ([`MaxSlowdownConstraint`](https://ml.energy/zeus/reference/optimizer/power_limit/#zeus.optimizer.power_limit.MaxSlowdownConstraint)).


[^1]: Well, to be precise, it won't be the *exact* same because on computers, the *order* of floating point operations [can make a small difference in the final result](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems) even if the two order of operations are mathematically equivalent. Still, it's close enough.
[^2]: [DAWNBench](https://dawn.cs.stanford.edu/benchmark/)
[^3]: Even in cases where there is really just one batch size that can reach the target metric, the user can disable batch size exploration by passing in a single set for the list of batch sizes of explore. Zeus still finds the optimal GPU power limit, leading to energy gains.
