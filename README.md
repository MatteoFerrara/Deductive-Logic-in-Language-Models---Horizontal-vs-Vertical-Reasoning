# Deductive Logic in Language Models: Horizontal vs. Vertical Reasoning

This is the official implementation of the experiments conducted in **[[1]](https://doi.org/10.3390/make8070214)**.

# Overview

The repository includes all necessary files to replicate the experiments conducted using:
- the **nanoGPT** architecture introduced in **[[2]](https://github.com/karpathy/nanoGPT)**,
- and the code available on [Github](https://github.com/abhay-sheshadri/backward-chaining-circuits) and presented in **[[3]](https://arxiv.org/abs/2402.11917)**.

# How to run
Clone the repository
```bash
git clone https://github.com/MatteoFerrara/Deductive-Logic-in-Language-Models---Horizontal-vs-Vertical-Reasoning.git
 ```   

Execute the Task 1 and Task 2 Jupyter notebooks.

# Requirements

The Jupyter notebooks `Task 1 - Training with CoT.ipynb` and `Task 1 - Training without CoT.ipynb` have been tested using the following libraries:

- Python 3.10
- Matplotlib 3.9
- NumPy 1.26
- Torch 2.1

The Jupyter notebook `Task 1 - Evaluation.ipynb` has been tested using the following libraries:

- Python 3.11
- ipywidgets 8.1
- Matplotlib 3.10
- NumPy 2.2
- Torch 2.7
- TransformerLens 2.15

The Jupyter notebooks `Task 2 - Unbiased dataset creation.ipynb`, `Task 2 - Root to leaf distance computation` and `Task 2 - Dataset without CoT creation` have been tested using the following libraries:

- Python 3.11

The Jupyter notebooks `Task 2 - Training with CoT.ipynb`, `Task 2 - Training without CoT.ipynb` and `Task 2 - CoT vs. Curriculum Learning.ipynb` have been tested using the following libraries:

- Python 3.11
- Matplotlib 3.10
- NumPy 2.2
- Torch 2.7
- TransformerLens 2.15

Other versions may work but are not guaranteed.

# Citation
Please cite [1], [2] and [3] in all publications and works that use this code.

# Bibliography
[1] D. Maltoni, and M. Ferrara, "Deductive Logic in Language Models: Horizontal vs. Vertical Reasoning", Machine Learning and Knowledge Extraction, 2026.

[2] A. Karpathy, "nanoGPT: A lightweight implementation of medium-sized GPTs", https://github.com/karpathy/nanoGPT, 2022.

[3] J. Brinkmann, A. Sheshadri, V. Levoso, P. Swoboda, and Christian Bartelt "A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task", arXiv:2402.11917, 2024.
