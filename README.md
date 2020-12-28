# Introduction of Tr-SAEA
This repository contains code necessary to reproduce the experiments presented in Transfer Learning Based Surrogate Assisted Evolutionary Bi-objective Optimization for Objectives with Different Evaluation Times.
Various multiobjective optimization algorithms have been proposed with a common assumption that the evaluation of each objective function takes the same period of time. 
Little attention has been paid to more general and realistic optimization scenarios where different objectives are evaluated by different computer simulations or physical
experiments with different time complexities and only a very limited number of function evaluations is allowed for the slow objective. In this work, we adopt the same 
environment as proposed by Allmendinger and co-authors, and investigate benchmark scenarios with two objectives. We propose a transfer learning scheme within a surrogate-assisted
evolutionary algorithm framework to augment the training data for the surrogate for the slow objective function by transferring knowledge from the fast one. Specifically,
a hybrid domain adaptation method aligning the second-order statistics and marginal distributions across domains is introduced to generate promising samples in the decision space
according to the search experience of the fast one. A Gaussian process model based co-training method is adopted to predict the value of the slow objective and those having 
a high confidence level are selected as the augmented synthetic training data, thereby enhancing the approximation quality of the surrogate of the slow objective. Our experimental
results on three test suites demonstrate that the proposed algorithm is competitive for solving bi-objective optimization problems where objectives have different evaluation times,
compared with existing surrogate and non-surrogate-assisted delay-handling methods.
If you found DNN-AR-MOEA useful, we would be grateful if you cite the following reference:
....
