In preference learning problems, our goal is to model the behavior of an expert agent, which given an exogenous signal, returns a response action. An underlying assumption is that to compute its response, the expert agent solves an optimization problem parametric in an exogenous signal. We assume to know the constraints imposed on the expert, but not its cost function. Therefore, using examples of exogenous signals and corresponding expert response actions, our goal is to model the cost function being optimized by the expert.


More concretely, given a dataset $\mathcal{D} = \\{(\hat{x}_ i, \hat{y}_ i)\\}_ {i=1}^N$ of exogenous signals $\hat{x}_ i$ and the respective expert's response $\hat{y}_ i$, feature mapping $\phi$, our goal is to find a cost vector $w \in \mathbb{R}^p$ such that a minimizer $y_ i$ of the **Forward Optimization Problem (FOP)**

$$
y_i \in \arg\min_ {y \in \mathbb{X}(\hat{x}_ i)} \ \langle w,\phi(\hat{x}_ i,y) \rangle
$$

reproduces (or in some sense approximates) the expert's action $\hat{y}_ i$.
