# Airfoil Optimization Project

## Overview

This project focuses on optimizing airfoil parameters using two different optimization algorithms:

1. **Genetic Algorithm (GA)**: Utilizes evolutionary principles to evolve airfoil designs for optimal aerodynamic performance.
2. **Stochastic Gradient Descent (SGD)**: Employs gradient-based optimization to fine-tune airfoil parameters for enhanced performance.

Also we provide a **Streamlit** web application to visualize and compare the optimized airfoil shapes.

## Project Structure

```
├── PANELS
│   ├── STREAMLINE_SPM.py
│   └── XFOIL.py
├── GA.py
├── SGD.py
├── streamlit_app.py
├── run_ga.py
├── requirements.txt
└── README.md
```

- **PANELS/**: Contains modules for aerodynamic calculations using XFOIL.
  - `STREAMLINE_SPM.py`
  - `XFOIL.py`
- **GA.py**: Implements the Genetic Algorithm for airfoil optimization.
- **SGD.py**: Implements the Stochastic Gradient Descent optimizer for airfoil optimization.
- **streamlit_app.py**: Streamlit web application for visualizing airfoil shapes.
- **run_ga.py**: Automates running the GA optimization across multiple angles of attack.
- **requirements.txt**: Lists all Python dependencies.

## Prerequisites

- **Python**: Version 3.7 or higher.
- **XFOIL**: Ensure XFOIL is installed and accessible. You can download it from [XFOIL Official Page](http://web.mit.edu/drela/Public/web/xfoil/). 

**Note:** XFOIL compiled with Windows is included in the repository, but if you want to run the program on another OS, you need to follow the documentation to replace the current executable with the one compiled for your OS.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/airfoil-optimization.git
cd airfoil-optimization
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Python Dependencies

Ensure you have `pip` installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

*If `requirements.txt` is not present, you can install dependencies manually:*

```bash
pip install numpy matplotlib argparse streamlit
```

## Usage

### 1. Genetic Algorithm (`GA.py`)

**Description**: Optimizes airfoil parameters (m, p, t) using a Genetic Algorithm to maximize the lift-to-drag ratio (Cl/Cd) based on XFOIL simulations.

**Execution Command**:

```bash
python GA.py [OPTIONS]
```

**Options**:

- `--alpha`: Angle of attack in degrees. *(Default: 5)*
- `--Re`: Reynolds number. *(Default: 1e6)*
- `--mu`: Population size. *(Default: 20)*
- `--lambda_`: Number of offspring to generate. *(Default: 40)*
- `--generations`: Maximum number of generations. *(Default: 50)*
- `--crossover_prob`: Probability of crossover. *(Default: 0.8)*
- `--mutation_prob`: Probability of mutation. *(Default: 0.1)*
- `--elitism_ratio`: Fraction of top individuals to carry over. *(Default: 0.1)*
- `--tournament_size`: Number of individuals in tournament selection. *(Default: 3)*
- `--convergence_threshold`: Threshold to detect convergence. *(Default: 0.1)*
- `--max_no_improvement`: Max generations with no improvement to consider convergence. *(Default: 10)*
- `--crossover_type`: Type of crossover (`one_point`, `two_point`, `uniform`). *(Default: `one_point`)*

**Example**:

```bash
python GA.py --alpha 5 --Re 1e6 --mu 30 --lambda_ 60 --generations 100 --crossover_type two_point
```

**Output**:

- **Best Individual**: Displays the best airfoil parameters found.
- **Fitness Improvement Plot**: Saves `fitness_improvement.png` showing the evolution of fitness over generations.

### 2. Stochastic Gradient Descent (`SGD.py`)

**Description**: Optimizes airfoil parameters (m, p, t) using Stochastic Gradient Descent to maximize the lift-to-drag ratio (Cl/Cd) based on XFOIL simulations.

**Execution Command**:

```bash
python SGD.py [OPTIONS]
```

**Options**:

- `--alpha`: Angle of attack in degrees. *(Default: 5)*
- `--Re`: Reynolds number. *(Default: 1e6)*
- `--learning_rate`: Learning rate for SGD updates. *(Default: 1)*
- `--epochs`: Number of optimization steps. *(Default: 1000)*
- `--tol`: Tolerance for convergence. *(Default: 1e-6)*
- `--verbose`: Enable verbose output. *(Flag)*
- `--initial_m`: Initial value for m parameter. *(Default: 2.0)*
- `--initial_p`: Initial value for p parameter. *(Default: 4.0)*
- `--initial_t`: Initial value for t parameter. *(Default: 15.0)*

**Example**:

```bash
python SGD.py --alpha 5 --Re 1e6 --learning_rate 0.01 --epochs 500 --initial_m 3.0 --initial_p 5.0 --initial_t 16.0 --verbose
```

**Output**:

- **Best Parameters**: Displays the optimized airfoil parameters.
- **Objective Improvement Plot**: Saves `objective_improvement.png` showing the optimization progress over epochs.

### 3. Streamlit Application (`plotAirfoil.py`)

**Description**: This Streamlit application generates and visualizes airfoil shapes based on NACA 4-digit airfoil parameters, allowing comparison between initial, SGD optimized, and GA optimized airfoil shapes.

**Execution Command**:

To start the Streamlit application, use the following command:

```bash
streamlit run streamlit_app.py
```

**Options in the Application**:

- **Initial Airfoil**: Input the NACA code for the initial airfoil (e.g., "2412").
- **SGD Optimized Airfoil**: Input the NACA code for the airfoil optimized with Stochastic Gradient Descent (e.g., "9406").
- **GA Optimized Airfoil**: Input the NACA code for the airfoil optimized with Genetic Algorithm (e.g., "9306").

**Example Usage**:

1. Run the Streamlit application:

   ```bash
   streamlit run streamlit_app.py
   ```

2. In the browser, enter the NACA codes in the provided sidebar fields and click the **Plot Airfoils** button.

**Output**:

- **Airfoil Shape Comparison**: A plot comparing the shapes of the initial, SGD optimized, and GA optimized airfoils.


