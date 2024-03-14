# Bayesian Machine Learning Project

## Description

This repository contains our work for the validation project of the course Bayesian Machine Learning, 2023/2024, ENS Paris-Saclay, Master MVA.

It consists in a thorough study of the paper "Challenges in Markov chain Monte Carlo
for Bayesian neural networks" by Theodore Papamarkou et al.

## Features

- Implementation of several MCMC algorithms for Bayesian inference : Random Walk Metropolis-Hastings (`RWHM.py`), Hamiltonian Monte Carlo (`HMC.py`), Metropolis-adjusted Langevin algorithm (`MALA.py`);
- Implementation of a BNN model, an MLP (`model.py`) fit for Bayesian inference via MCMC;
- BNN implementation with PyTorch (MLPs);
- Application to synthetic and real datasets both from the paper and our own.

## Installation

To set up a virtual environment and install the necessary dependencies, follow these steps:

### Prerequisites

Ensure you have Python 3.6 or newer installed on your system. You can download Python from [the official website](https://www.python.org/downloads/).

### Setting Up a Virtual Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/hugo-qea/BayesianML-Project.git
   cd BayesianML-Project
    ```
2. Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```
3. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
After installing the project, you can run the examples or your own scripts within the virtual environment. Make sure to activate the virtual environment whenever you're working on the project.

### Running Examples
To run our examples, dedicated notebooks is provided in the `notebooks` directory. You can open them using Jupyter Notebook or Jupyter Lab and execute the cells to see the results.
    
    ```bash
    jupyter notebook
    ```

    ```bash
    jupyter lab
    ```
- A notebook that carries out the predictions on the datasets of the paper `mcmc.ipynb` is provided.
- A notebook that computes several diagnostics on this Markov Chains `mcmc_diagnostic.ipynb` is provided.
- A notebook that carries out our own experiments on the Iris dataset `mcmc_iris.ipynb` is provided.
- A notebook that determines the accuracy through SGD `sgd.ipynb` is provided.

For testing, the folder `tests` contains several notebooks with tests for the different samplers and models.


### Creating Your Own Scripts
You can create your own Python scripts and run them within the virtual environment. Make sure to activate the virtual environment before running your scripts:
```bash
# Activate the virtual environment
source venv/bin/activate  # Unix/MacOS
.\venv\Scripts\activate  # Windows

# Run your script
python your_script.py
```

## Authors
- Ben Kabongo, Ecole Normale Supérieure Paris-Saclay - [BenKabongo25](https://github.com/BenKabongo25)
- Hugo Queniat, Télécom Paris - [hugo-qea](https://github.com/hugo-qea)
- Thibault Robine, Télécom SudParis - [ThibaultRobine](https://github.com/ThibaultRobine)
