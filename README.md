<h1 align="center"> Wasserstein Convergence of Critically Damped Langevin Diffusions </h1>


This repository contains training and sampling code for Critically Damped Langevin Diffusions (CLDs) and scripts to reproduce the experiments from our paper, “Wasserstein Convergence of Critically Damped Langevin Diffusions”. CLDs extend standard Score-based Generative Models to an augmented position–velocity space, inspired by Hamiltonian dynamics. The paper establishes the first Wasserstein convergence result for the sampling error of CLD-based methods and analyzes a generalized dynamic that introduces a hyperparameter controlling noise on the data coordinates, exploiting the extended space more effectively.
The implementation supports both Euler–Maruyama and Leapfrog discretizations, includes a regularized CLD variant with a tunable smoothing hyperparameter $\varepsilon$, and offers reproducible evaluation of Wasserstein error on synthetic datasets such as Funnel, MG-25, and Diamonds.


### 🛠 Environment Setup
To set up the environment, execute the following commands:

```shell
conda create -n cld python=3.9
conda activate cld
pip install -r requirements.txt
```

### 🚀 Usage Instructions

To reproduce the results, run:

```shell
python src/run.py
```

To run the code for a specific dataset with drift coefficient $a$ and regularization parameter $\varepsilon$, use:
```shell
python src/run.py --dataset "funnel" --a 1 --epsilon 0.5
```


### 🙏 Acknowledgments
We gratefully acknowledge the repository that served as a baseline and inspiration for our implementation: [CLD-SGM by NV-TLABS](https://github.com/nv-tlabs/CLD-SGM).











