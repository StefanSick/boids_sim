# Boids Simulation

## Setup and Installation

To ensure reproducibility and correct dependency management, please use the provided `environment.yml` file to set up your local environment.

### 1. Create the Conda Environment
Open your terminal or Anaconda Prompt in the project root directory and run:
```bash
conda env create -f environment.yml
```
### 2. Activate the Environment
```bash
conda activate boids
```
### 3. Register the Kernel for Jupyter
To ensure that Jupyter Notebook can "see" this specific environment, you must install it as a kernel:
```bash
python -m ipykernel install --user --name boids --display-name "Python (boids)"
```

