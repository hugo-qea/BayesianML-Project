Repository for the project of the course Bayesian Machine Learning, 2023/2024, ENS Paris-Saclay, Master MVA.

It is a thorough study of the paper "Challenges in Markov chain Monte Carlo
for Bayesian neural networks" by Theodore Papamarkou et al.

# Bayesian Machine Learning Project

## Description

This project is dedicated to exploring and implementing Bayesian Machine Learning techniques. It aims to provide a comprehensive understanding and application of Bayesian methods in machine learning, showcasing how these approaches can be used to model uncertainty, improve predictions, and make informed decisions based on probabilistic reasoning.

## Features

- Implementation of various Bayesian models
- Examples and tutorials on Bayesian inference
- Comparison of Bayesian methods with traditional machine learning approaches
- Case studies demonstrating the application of Bayesian Machine Learning in real-world scenarios

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
To run our examples, a dedicated notebook is provided in the `notebooks` directory. You can open it using Jupyter Notebook or Jupyter Lab and execute the cells to see the results.

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
- Ben Kabongo, Ecole Normale Supérieure Paris-Saclay - BenKabongo25
- Hugo Queniat, Télécom Paris - hugo-qea
- Thibault Robine, Télécom SudParis - ThibaultRobine
