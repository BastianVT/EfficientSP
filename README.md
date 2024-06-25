In preference learning problems, our goal is to model the behavior of an expert agent, which given an exogenous signal, returns a response action. An underlying assumption is that to compute its response, the expert agent solves an optimization problem parametric in an exogenous signal. We assume to know the constraints imposed on the expert, but not its cost function. Therefore, using examples of exogenous signals and corresponding expert response actions, our goal is to model the cost function being optimized by the expert.


More concretely, given a dataset $\mathcal{D} = \\{(\hat{x}_ i, \hat{y}_ i)\\}_ {i=1}^N$ of exogenous signals $\hat{x}_ i$ and the respective expert's response $\hat{y}_ i$, feature mapping $\phi$, our goal is to find a cost vector $w \in \mathbb{R}^p$ such that a minimizer $y_ i$ of the **Forward Optimization Problem (FOP)**

$$
y_i \in \arg\min_ {y \in \mathbb{Y}(\hat{x}_ i)} \ \langle w,\phi(\hat{x}_ i,y) \rangle
$$

reproduces (or in some sense approximates) the expert's action $\hat{y}_ i$.

-------------------------
DATA GENERATION:

We generate data for two types of problems using Python: the Knapsack (KP) and the Price-Collecting Travelling Salesman Problem -modifying size parameters generated multiple sizes of both problems.

To generate the KP data we use the scriptS named: "\main\kp\instance_generator.py" and "\main\kp\dataset_generator_1.py" and for the PCTSP data we use the following script to generate the instances: "\main\pctsp\instances_generator.py", which are consolidated in the file: "\main\pctsp\dataset_generator.py". the parameter 'n_materials' defines the number of total clients served in this problem while the 'n_datasets' parameter defines the number of instances.

The instances are stored in txt files in folders, with the following nomenclature: 'size_nxx_yy' with xx the amount of noise introduced and yy the size of the problem (number of clients).

--------------------------------

If you are using the Efficient SP, please cite our accompanying paper:

```bibtex
@article{vejarefficient,
  title={An Efficient Structured Perceptron for NP-hard Combinatorial Optimization Problems},
  author={V{\'e}jar, Basti{\'a}n and Aglin, Ga{\"e}l and Mahmuto{\u{g}}ullar{\i}, Ali {\.I}rfan and Nijssen, Siegfried and Schaus, Pierre and Guns, Tias}
}
```


