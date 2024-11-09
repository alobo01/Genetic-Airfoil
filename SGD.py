import numpy as np
import random
from abc import ABC, abstractmethod
from PANELS.XFOIL import XFOIL  # Ensure this module is correctly imported

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

    def optimize(self, initial_params, learning_rate=0.01, epochs=100,
                tol=1e-6, verbose=True):
        """Runs the SGD optimization process.

        Args:
            initial_params (np.array): Initial values for the parameters.
            learning_rate (float): The learning rate for SGD updates.
            epochs (int): The number of optimization steps.
            tol (float): Tolerance for convergence.
            verbose (bool): Whether to print progress.

        Returns:
            np.array: The best parameters found.
        """
        params = np.array(initial_params, dtype=float)
        self.best_objective = self.objective(params)
        self.best_params = params.copy()
        self.history.append(self.best_objective)

        for epoch in range(epochs):
            grad = self.compute_gradient(params)
            params += learning_rate * grad  # Ascending gradient since we maximize objective

            # Ensure parameters remain within their valid ranges
            params = self.enforce_bounds(params)

            current_objective = self.objective(params)
            self.history.append(current_objective)

            if current_objective > self.best_objective:
                self.best_objective = current_objective
                self.best_params = params.copy()

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Objective = {current_objective:.6f}, Params = {params}")

            # Check for convergence
            if np.abs(current_objective - self.best_objective) < tol:
                if verbose:
                    print(f"Convergence detected at epoch {epoch}.")
                break

        return self.best_params

    def compute_gradient(self, params, epsilon=1e-5):
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

    def __init__(self, alpha, Re, M):
        """Initializes the airfoil optimization with given conditions.

        Args:
            alpha (float): Angle of attack in degrees.
            Re (float): Reynolds number.
            M (float): Mach number.
        """
        super().__init__()
        self.alpha = alpha
        self.Re = Re
        self.M = M

    def objective(self, params):
        """Calculates the objective based on airfoil performance.

        Args:
            params (np.array): The parameters to evaluate (m, p, t).

        Returns:
            float: The objective value (Cl/Cd ratio).
        """
        m, p, t = params
        naca_code = f"{int(round(m))}{int(round(p))}{int(round(t)):02d}"
        AoAR = self.alpha * (np.pi / 180)  # Convert angle of attack to radians

        # PPAR menu options for XFOIL (adjust as needed)
        PPAR = ['170', '4', '1', '1', '1 1', '1 1']

        try:
            # Get XFOIL results for the prescribed airfoil
            xFoilResults = XFOIL(naca_code, PPAR, AoAR, 'circle', useNACA=True, Re=self.Re)

            # Check if XFOIL returned valid results
            if xFoilResults is None or len(xFoilResults) < 9:
                print(f"XFOIL failed for NACA {naca_code}. Assigning low objective value.")
                return -1e6  # Assign a very low objective to penalize

            # Extract results
            afName = xFoilResults[0]        # Airfoil name
            xFoilX = xFoilResults[1]        # X-coordinates for Cp
            xFoilY = xFoilResults[2]        # Y-coordinates for Cp
            xFoilCP = xFoilResults[3]       # Pressure coefficient Cp
            XB = xFoilResults[4]            # Leading edge X-coordinates
            YB = xFoilResults[5]            # Leading edge Y-coordinates
            xFoilCL = xFoilResults[6]       # Lift coefficient CL
            xFoilCD = xFoilResults[7]       # Drag coefficient CD

            # Print the coefficients for debugging
            print(f"Airfoil NACA {naca_code}: CL={xFoilCL}, CD={xFoilCD}")

            # Check if CD is valid to avoid division by zero
            if xFoilCD <= 0 or not np.isfinite(xFoilCL) or not np.isfinite(xFoilCD):
                print(f"Invalid values for NACA {naca_code}: CL={xFoilCL}, CD={xFoilCD}. Assigning low objective value.")
                return -1e6  # Assign a very low objective to penalize

            # Calculate objective as the CL/CD ratio
            objective_value = xFoilCL / xFoilCD
            return objective_value

        except Exception as e:
            # Handle any other exception
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
        m = np.clip(m, 0, 9)    # m between 0 and 9
        p = np.clip(p, 1, 10)   # p between 1 and 10
        t = np.clip(t, 6, 24)   # t between 6 and 24
        return np.array([m, p, t])

if __name__ == "__main__":
    # Define optimization conditions
    alpha = 5    # Angle of attack in degrees
    Re = 1e6     # Reynolds number
    M = 0.2      # Mach number

    # Create an instance of the airfoil optimization class
    sgd_optimizer = AirfoilSGDOptimization(alpha, Re, M)

    # Initialize parameters (m, p, t) within their ranges
    initial_params = np.array([2.0, 4.0, 15.0])  # Example initial guess

    # Define optimization hyperparameters
    learning_rate = 0.01
    epochs = 1000
    tolerance = 1e-6
    verbose = True

    # Run the optimization
    best_params = sgd_optimizer.optimize(
        initial_params=initial_params,
        learning_rate=learning_rate,
        epochs=epochs,
        tol=tolerance,
        verbose=verbose
    )

    # Extract the best parameters
    m, p, t = best_params

    # Print the best solution
    print("\nOptimization Completed.")
    print("Best parameters: m = {:.2f}, p = {:.2f}, t = {:.2f}".format(m, p, t))
    print("Best objective (Cl/Cd ratio): {:.6f}".format(sgd_optimizer.best_objective))
