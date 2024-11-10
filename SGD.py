import numpy as np
from abc import ABC, abstractmethod
from PANELS.XFOIL import XFOIL  # Ensure this module is correctly imported
import argparse
import time
import logging
import matplotlib.pyplot as plt  # Added import for plotting

class SGDOptimizer(ABC):
    """Abstract base class for a Stochastic Gradient Descent Optimizer."""

    def __init__(self):
        """Initializes the optimizer with an empty history."""
        self.history = []
        self.best_objective = -np.inf
        self.best_params = None

    @abstractmethod
    def objective(self, params):
        """Calculates the objective value for the given parameters.

        Args:
            params (np.array): The parameters to evaluate.

        Returns:
            float: The objective value.
        """
        pass

    def optimize(self, initial_params, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs=100,
                 tol=1e-6, verbose=True):
        """Runs the Adam optimization process.

        Args:
            initial_params (np.array): Initial values for the parameters.
            learning_rate (float): The learning rate for Adam updates.
            beta1 (float): Exponential decay rate for the first moment estimate (momentum).
            beta2 (float): Exponential decay rate for the second moment estimate (adaptive learning rate).
            epsilon (float): Small constant for numerical stability.
            epochs (int): The number of optimization steps.
            tol (float): Tolerance for convergence.
            verbose (bool): Whether to print progress.

        Returns:
            np.array: The best parameters found.
        """
        # Configure logging
        logging_level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(level=logging_level, format='%(message)s')
        self.logger = logging.getLogger()

        params = np.array(initial_params, dtype=float)
        m = np.zeros_like(params)  # Initialize first moment vector
        v = np.zeros_like(params)  # Initialize second moment vector
        self.best_objective = self.objective(params)
        self.best_params = params.copy()
        self.history.append(self.best_objective)

        self.logger.info(f"Initial Params: {params}, Objective: {self.best_objective:.6f}")

        for epoch in range(1, epochs + 1):
            grad = self.compute_gradient(params)

            # Update biased first and second moment estimates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            # Correct bias in moment estimates
            m_hat = m / (1 - beta1 ** (epoch + 1))
            v_hat = v / (1 - beta2 ** (epoch + 1))

            # Update parameters
            params += learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            # Ensure parameters remain within their valid ranges
            params = self.enforce_bounds(params)

            current_objective = self.objective(params)
            self.history.append(current_objective)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Objective = {current_objective:.6f}, Params = {params}")

            # Check for convergence
            if np.abs(current_objective - self.best_objective) < tol:
                if verbose:
                    self.logger.info(f"Convergence detected at epoch {epoch}.")
                break

            if current_objective > self.best_objective:
                self.best_objective = current_objective
                self.best_params = params.copy()

        return self.best_params

    def compute_gradient(self, params, epsilon=1):
        """Computes the numerical gradient of the objective function.

        Args:
            params (np.array): Current parameter values.
            epsilon (float): Small perturbation for finite differences.

        Returns:
            np.array: The gradient vector.
        """
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_eps = params.copy()
            params_eps[i] += epsilon
            f1 = self.objective(params_eps)
            f0 = self.objective(params)
            grad[i] = (f1 - f0) / epsilon
        return grad

    @abstractmethod
    def enforce_bounds(self, params):
        """Ensures that the parameters remain within their allowed ranges.

        Args:
            params (np.array): Current parameter values.

        Returns:
            np.array: The adjusted parameters.
        """
        pass

class AirfoilSGDOptimization(SGDOptimizer):
    """SGD Optimizer for optimizing airfoil parameters."""

    def __init__(self, alpha, Re, verbose=False):
        """Initializes the airfoil optimization with given conditions.

        Args:
            alpha (float): Angle of attack in degrees.
            Re (float): Reynolds number.
            verbose (bool): Whether to enable verbose output.
        """
        super().__init__()
        self.alpha = alpha
        self.Re = Re
        self.verbose = verbose
        # Define parameter bounds as tuples: (min, max)
        self.bounds = {
            'm': (0, 9),     # m between 0 and 9
            'p': (1, 9),     # p between 1 and 9
            't': (6, 24)     # t between 6 and 24
        }

    def objective(self, params):
        """Calculates the objective based on airfoil performance.

        Args:
            params (np.array): The parameters to evaluate (m, p, t).

        Returns:
            float: The objective value (Cl/Cd ratio).
        """
        try:
            m, p, t = params
            m_int = int(round(m))
            p_int = int(round(p))
            t_int = int(round(t))

            # Validate parameters
            if not (self.bounds['m'][0] <= m_int <= self.bounds['m'][1]):
                if self.verbose:
                    print(f"Invalid m value: {m_int}. Assigning low objective value.")
                return -1  # Assign a smaller negative objective to penalize
            if not (self.bounds['p'][0] <= p_int <= self.bounds['p'][1]):
                if self.verbose:
                    print(f"Invalid p value: {p_int}. Assigning low objective value.")
                return -1  # Assign a smaller negative objective to penalize
            if not (self.bounds['t'][0] <= t_int <= self.bounds['t'][1]):
                if self.verbose:
                    print(f"Invalid t value: {t_int}. Assigning low objective value.")
                return -1  # Assign a smaller negative objective to penalize

            naca_code = f"{m_int}{p_int}{t_int:02d}"
            AoAR = self.alpha   # Angle of attack in degrees

            # PPAR menu options for XFOIL (adjust as needed)
            PPAR = ['170', '4', '1', '1', '1 1', '1 1']

            # Get XFOIL results for the prescribed airfoil
            xFoilResults = XFOIL(naca_code, PPAR, AoAR, 'circle', useNACA=True, Re=self.Re)

            # Check if XFOIL returned valid results
            if xFoilResults is None or len(xFoilResults) < 9:
                if self.verbose:
                    print(f"XFOIL failed for NACA {naca_code}. Assigning low objective value.")
                return -1  # Assign a smaller negative objective to penalize

            # Extract results
            afName = xFoilResults[0]        # Airfoil name
            xFoilX = xFoilResults[1]        # X-coordinates for Cp
            xFoilY = xFoilResults[2]        # Y-coordinates for Cp
            xFoilCP = xFoilResults[3]       # Pressure coefficient Cp
            XB = xFoilResults[4]            # Leading edge X-coordinates
            YB = xFoilResults[5]            # Leading edge Y-coordinates
            xFoilCL = xFoilResults[6]       # Lift coefficient CL
            xFoilCD = xFoilResults[7]       # Drag coefficient CD

            # Print the coefficients for debugging if verbose
            if self.verbose:
                print(f"Airfoil NACA {naca_code}: CL={xFoilCL}, CD={xFoilCD}")

            # Check if CD is valid to avoid division by zero
            if xFoilCD <= 0 or not np.isfinite(xFoilCL) or not np.isfinite(xFoilCD):
                if self.verbose:
                    print(f"Invalid values for NACA {naca_code}: CL={xFoilCL}, CD={xFoilCD}. Assigning low objective value.")
                return -1  # Assign a smaller negative objective to penalize

            # Calculate objective as the CL/CD ratio
            objective_value = xFoilCL / xFoilCD
            return objective_value
        except Exception as e:
            # Handle any other exception
            if self.verbose:
                print(f"Error in the objective function for NACA {naca_code}: {e}")
            return -1e6  # Assign a very low objective to penalize

    def enforce_bounds(self, params):
        """Ensures that the parameters remain within their valid ranges.

        Args:
            params (np.array): Current parameter values.

        Returns:
            np.array: The adjusted parameters.
        """
        m, p, t = params
        m = np.clip(m, self.bounds['m'][0], self.bounds['m'][1])
        p = np.clip(p, self.bounds['p'][0], self.bounds['p'][1])
        t = np.clip(t, self.bounds['t'][0], self.bounds['t'][1])
        return np.array([m, p, t])

def main():
    parser = argparse.ArgumentParser(description="Stochastic Gradient Descent Optimizer for Airfoil Parameters using XFOIL.")

    # Airfoil parameters
    parser.add_argument('--alpha', type=float, default=5, help='Angle of attack in degrees.')
    parser.add_argument('--Re', type=float, default=1e6, help='Reynolds number.')

    # Optimization hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1, help='Learning rate for SGD updates.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of optimization steps.')
    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for convergence.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')

    # Initial parameters
    parser.add_argument('--initial_m', type=float, default=2.0, help='Initial value for m parameter.')
    parser.add_argument('--initial_p', type=float, default=4.0, help='Initial value for p parameter.')
    parser.add_argument('--initial_t', type=float, default=15.0, help='Initial value for t parameter.')

    args = parser.parse_args()

    # Create an instance of the airfoil optimization class with provided parameters
    sgd_optimizer = AirfoilSGDOptimization(alpha=args.alpha, Re=args.Re, verbose=args.verbose)

    # Initialize parameters (m, p, t) within their ranges
    initial_params = np.array([args.initial_m, args.initial_p, args.initial_t])

    # Run the optimization with provided hyperparameters
    start_time = time.time()

    best_params = sgd_optimizer.optimize(
        initial_params=initial_params,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        tol=args.tol,
        verbose=args.verbose
    )

    end_time = time.time()

    total_time = end_time - start_time

    # Extract the best parameters
    m, p, t = best_params

    # Print the best solution
    print("\nOptimization Completed.")
    print("Best parameters: m = {:.2f}, p = {:.2f}, t = {:.2f}".format(m, p, t))
    print(f"Execution time: {total_time:.2f} seconds.")

    # Plot the objective improvement over epochs
    epochs_range = range(1, len(sgd_optimizer.history) + 1)
    objective_values = sgd_optimizer.history

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, objective_values, label='Objective (Cl/Cd)', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Objective Value (Cl/Cd)')
    plt.title('Objective Improvement Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Highlight the best objective
    plt.legend()

    # Show the plot

    plt.savefig('objective_improvement.png')

    plt.close()

if __name__ == "__main__":
    main()
