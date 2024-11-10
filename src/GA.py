import random
from abc import ABC, abstractmethod
from PANELS.XFOIL import XFOIL
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

class GeneticAlgorithm(ABC):
    """Abstract base class for a Genetic Algorithm."""

    def __init__(self):
        """Initializes the genetic algorithm with an empty fitness history."""
        self.fitness_history = []

    @abstractmethod
    def fitness(self, individual):
        """Calculates the fitness of an individual.

        Args:
            individual (list): The individual to evaluate.

        Returns:
            float: The fitness value of the individual.
        """
        pass

    @abstractmethod
    def create_individual(self):
        """Creates a new random individual.

        Returns:
            list: A new individual represented as a list of genes.
        """
        pass

    def create_population(self, size):
        """Creates an initial population.

        Args:
            size (int): The size of the population to create.

        Returns:
            list: A list of individuals representing the population.
        """
        return [self.create_individual() for _ in range(size)]

    def one_point_crossover(self, parent1, parent2):
        """Performs one-point crossover between two parents to produce two offspring.

        Args:
            parent1 (list): The first parent individual.
            parent2 (list): The second parent individual.

        Returns:
            tuple: Two offspring individuals resulting from the crossover.
        """
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def two_point_crossover(self, parent1, parent2):
        """Performs two-point crossover between two parents to produce two offspring.

        Args:
            parent1 (list): The first parent individual.
            parent2 (list): The second parent individual.

        Returns:
            tuple: Two offspring individuals resulting from the crossover.
        """
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)
        
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        
        return child1, child2

    def uniform_crossover(self, parent1, parent2, crossover_prob=0.5):
        """Performs uniform crossover between two parents to produce two offspring.

        Args:
            parent1 (list): The first parent individual.
            parent2 (list): The second parent individual.
            crossover_prob (float): The probability of selecting genes from parent1.

        Returns:
            tuple: Two offspring individuals resulting from the crossover.
        """
        child1 = []
        child2 = []
        
        for p1_gene, p2_gene in zip(parent1, parent2):
            if random.random() < crossover_prob:
                child1.append(p1_gene)
                child2.append(p2_gene)
            else:
                child1.append(p2_gene)
                child2.append(p1_gene)
        
        return child1, child2

    def perform_crossover(self, parent1, parent2, crossover_type):
        """Perform crossover.

        Args:
            parent1 (list): The first parent individual.
            parent2 (list): The second parent individual.
            crossover_type (string): The type of crossover to apply.

        Returns:
            tuple: Two offspring individuals resulting from the crossover.
        """
        if crossover_type == "one_point":
            return self.one_point_crossover(parent1, parent2)
        elif crossover_type == "two_point":
            return self.two_point_crossover(parent1, parent2)
        elif crossover_type == "uniform":
            return self.uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover type: {crossover_type}")

    def binomial_mutation(self, individual, mutpb):
        """Applies binomial mutation to an individual.

        Args:
            individual (list): The individual to mutate.
            mutpb (float): The mutation probability for each gene.

        Returns:
            list: The mutated individual.
        """
        return [1 - bit if random.random() < mutpb else bit for bit in individual]

    def tournament_selection(self, population, tournament_size):
        """Selects an individual from the population using tournament selection.

        Args:
            population (list): The population to select from.
            tournament_size (int): The number of individuals competing in the tournament.

        Returns:
            list: The selected individual.
        """
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda ind: self.fitness(ind))
        return winner

    def detect_convergence(self, threshold=0.001, max_no_improvement=10):
        """Detects if the algorithm has converged based on fitness history.

        Args:
            threshold (float): The minimum relative improvement to consider.
            max_no_improvement (int): The number of generations with no improvement to check.

        Returns:
            bool: True if convergence is detected, False otherwise.
        """
        if len(self.fitness_history) < max_no_improvement:
            return False

        mean_fitnesses = [mean for mean, _ in self.fitness_history[-max_no_improvement:]]
        max_fitnesses = [max_val for _, max_val in self.fitness_history[-max_no_improvement:]]

        # Check relative improvement in fitness
        mean_change = abs(mean_fitnesses[-1] - mean_fitnesses[0]) / (abs(mean_fitnesses[0]) + 1e-9)
        max_change = abs(max_fitnesses[-1] - max_fitnesses[0]) / (abs(max_fitnesses[0]) + 1e-9)

        return mean_change < threshold and max_change < threshold

    def optimize(self, mu=20, lambda_=40, generations=50, crossover_prob=0.8, mutation_prob=0.1,
                 elitism_ratio=0.1, tournament_size=3, convergence_threshold=0.1, max_no_improvement=10, crossover_type='one_point'):
        """Runs the genetic algorithm optimization process.

        Args:
            mu (int): The size of the population.
            lambda_ (int): The number of offspring to generate.
            generations (int): The maximum number of generations to run.
            crossover_prob (float): The probability of crossover occurring.
            mutation_prob (float): The probability of a gene mutating.
            elitism_ratio (float): The fraction of top individuals to carry over to the next generation.
            tournament_size (int): The size of the tournament for selection.
            convergence_threshold (float): The threshold to detect convergence.
            max_no_improvement (int): The number of generations with no improvement to consider convergence.

        Returns:
            list: The best individual found during optimization.
        """
        # Create the initial population (μ)
        population = self.create_population(mu)

        actual_generation = 0
        # We add caching to speed up the process
        cache = {}
        while actual_generation < generations:
            # Evaluate fitness of current population with caching
            fitness_values = []
            for ind in population:
                # Check if individual fitness is already in cache
                ind_key = str(ind)
                if ind_key in cache:
                    fitness = cache[ind_key]
                else:
                    # Calculate fitness and store it in cache
                    fitness = self.fitness(ind)
                    cache[ind_key] = fitness
                fitness_values.append(fitness)
            
            mean_fitness = sum(fitness_values) / len(fitness_values)
            max_fitness = max(fitness_values)
            self.fitness_history.append((mean_fitness, max_fitness))

            # Logging for debugging
            print(f"Generation {actual_generation + 1}: Mean Fitness = {mean_fitness:.4f}, Max Fitness = {max_fitness:.4f}")

            # Convergence check
            if self.detect_convergence(convergence_threshold, max_no_improvement):
                print(f"Convergence detected at generation {actual_generation + 1}.")
                break

            # Elitism: keep the best individuals
            num_elite = max(1, int(mu * elitism_ratio))  # Ensure at least one elite
            population = sorted(population, key=lambda ind: cache[str(ind)], reverse=True)
            next_gen = population[:num_elite]

            # Generate offspring
            offspring = []
            while len(offspring) < lambda_:
                # Parent selection
                parent1 = self.tournament_selection(population, tournament_size)
                parent2 = self.tournament_selection(population, tournament_size)

                # Crossover
                if random.random() < crossover_prob:
                    child1, child2 = self.perform_crossover(parent1, parent2, crossover_type)
                else:
                    child1, child2 = parent1[:], parent2[:]

                # Mutation
                child1 = self.binomial_mutation(child1, mutation_prob)
                child2 = self.binomial_mutation(child2, mutation_prob)

                offspring.extend([child1, child2])

            next_gen.extend(offspring[:mu - num_elite])  # Fill up to μ individuals
            population = next_gen[:mu]  # New generation
            actual_generation += 1

        # After optimization
        best_individual = max(population, key=lambda ind: cache[str(ind)])
        num_xfoil_calls = len(cache)
        print(f"Optimization completed at generation {actual_generation}.")
        print(f"Total XFOIL calculations: {num_xfoil_calls}")

        return best_individual


# Gray coding functions
def binary_to_gray(n):
    """Converts a binary number to its Gray code equivalent.

    Args:
        n (int): The binary number to convert.

    Returns:
        int: The Gray code equivalent of the binary number.
    """
    return n ^ (n >> 1)

def gray_to_binary(n):
    """Converts a Gray code number to its binary equivalent.

    Args:
        n (int): The Gray code number to convert.

    Returns:
        int: The binary equivalent of the Gray code number.
    """
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return n

def int_to_gray_bits(n, num_bits):
    """Converts an integer to a list of Gray-coded bits.

    Args:
        n (int): The integer to convert.
        num_bits (int): The number of bits to represent the number.

    Returns:
        list: A list of bits representing the Gray-coded integer.
    """
    gray = binary_to_gray(n)
    return [int(bit) for bit in f"{gray:0{num_bits}b}"]

def gray_bits_to_int(bits):
    """Converts a list of Gray-coded bits to an integer.

    Args:
        bits (list): The list of Gray-coded bits.

    Returns:
        int: The integer value of the Gray-coded bits.
    """
    gray = int("".join(map(str, bits)), 2)
    return gray_to_binary(gray)


class AirfoilGAOptimization(GeneticAlgorithm):
    """Genetic Algorithm for optimizing airfoil parameters."""

    def __init__(self, alpha, Re):
        """Initializes the airfoil optimization with given conditions.

        Args:
            alpha (float): Angle of attack in degrees.
            Re (float): Reynolds number.
        """
        super().__init__()
        self.alpha = alpha
        self.Re = Re
        self.m_bits = 4  # Number of bits for m parameter
        self.p_bits = 4  # Number of bits for p parameter
        self.t_bits = 7  # Number of bits for t parameter
        self.n_bits = self.m_bits + self.p_bits + self.t_bits  # Total number of bits per individual
    

    def fitness(self, individual):
        """Calculates the fitness of an individual based on airfoil performance.

        Args:
            individual (list): The individual to evaluate.

        Returns:
            float: The fitness value (Cl/Cd) of the individual.
        """
        try:
            # Decode the individual into parameters m, p, t
            m, p, t = self.decode(individual)
            naca_code = f"{m}{p}{t:02d}"
            AoAR = self.alpha  # Use the alpha provided during initialization
            
            # PPAR menu options for XFOIL
            PPAR = ['170', '4', '1', '1', '1 1', '1 1']
            
            # Get XFOIL results for the prescribed airfoil
            xFoilResults = XFOIL(naca_code, PPAR, AoAR, 'circle', useNACA=True, Re=self.Re)
            
            # Check if XFOIL returned valid results
            if xFoilResults is None or len(xFoilResults) < 9:
                print(f"XFOIL failed for NACA {naca_code}. Assigning negative fitness.")
                return -1  # Assign a smaller negative fitness to penalize
            
            # Extract results
            afName = xFoilResults[0]        # Airfoil name
            xFoilX = xFoilResults[1]        # X-coordinates for Cp
            xFoilY = xFoilResults[2]        # Y-coordinates for Cp
            xFoilCP = xFoilResults[3]       # Pressure coefficient Cp
            XB = xFoilResults[4]            # Leading edge X-coordinates
            YB = xFoilResults[5]            # Leading edge Y-coordinates
            xFoilCL = xFoilResults[6]       # Lift coefficient CL
            xFoilCD = xFoilResults[7]       # Drag coefficient CD
            
            # Print the drag coefficient for debugging
            print(f"Airfoil NACA {naca_code}: CL={xFoilCL}, CD={xFoilCD}")
            
            # Check if CD is valid to avoid division by zero
            if xFoilCD <= 0 or not np.isfinite(xFoilCL) or not np.isfinite(xFoilCD):
                print(f"Invalid values for NACA {naca_code}: CL={xFoilCL}, CD={xFoilCD}. Assigning negative fitness.")
                return -1  # Assign a smaller negative fitness to penalize
            
            # Calculate fitness as the CL/CD ratio
            fitness_value = xFoilCL / xFoilCD
            return fitness_value
        
        except Exception as e:
            # Handle any other exception
            print(f"Error in the fitness function for NACA {naca_code}: {e}")
            return -1  # Assign a smaller negative fitness to penalize

    def create_individual(self):
        """Creates a new random individual.

        Returns:
            list: A new individual represented as a list of bits.
        """
        return [random.randint(0, 1) for _ in range(self.n_bits)]

    def decode(self, individual):
        """Decodes an individual into airfoil parameters m, p, and t.

        Args:
            individual (list): The individual to decode.

        Returns:
            tuple: The decoded parameters (m, p, t).
        """
        m_bits = individual[:self.m_bits]
        p_bits = individual[self.m_bits:self.m_bits + self.p_bits]
        t_bits = individual[self.m_bits + self.p_bits:]

        # Convert Gray-coded bits to integer and scale to appropriate ranges
        m = round(gray_bits_to_int(m_bits) / (2**self.m_bits - 1) * 9)
        p = round(gray_bits_to_int(p_bits) / (2**self.p_bits - 1) * 9)  # Changed to 0-9
        t = round(gray_bits_to_int(t_bits) / (2**self.t_bits - 1) * 18 + 6)

        return m, p, t

    def encode(self, m, p, t):
        """Encodes airfoil parameters m, p, and t into an individual.

        Args:
            m (int): The maximum camber as a percentage of the chord.
            p (int): The position of the maximum camber along the chord.
            t (int): The maximum thickness as a percentage of the chord.

        Returns:
            list: The encoded individual represented as a list of bits.
        """
        m_bits = int_to_gray_bits(int(m / 9 * (2**self.m_bits - 1)), self.m_bits)
        p_bits = int_to_gray_bits(int(p / 9 * (2**self.p_bits - 1)), self.p_bits)  # Changed to 0-9
        t_bits = int_to_gray_bits(int((t - 6) / 18 * (2**self.t_bits - 1)), self.t_bits)
        return m_bits + p_bits + t_bits

def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Airfoil Optimization using XFOIL.")
    
    # Airfoil parameters
    parser.add_argument('--alpha', type=float, default=5, help='Angle of attack in degrees.')
    parser.add_argument('--Re', type=float, default=1e6, help='Reynolds number.')
    
    # GA parameters
    parser.add_argument('--mu', type=int, default=20, help='Population size.')
    parser.add_argument('--lambda_', type=int, default=40, help='Number of offspring to generate.')
    parser.add_argument('--generations', type=int, default=50, help='Maximum number of generations.')
    parser.add_argument('--crossover_prob', type=float, default=0.8, help='Crossover probability.')
    parser.add_argument('--mutation_prob', type=float, default=0.1, help='Mutation probability.')
    parser.add_argument('--elitism_ratio', type=float, default=0.1, help='Elitism ratio.')
    parser.add_argument('--tournament_size', type=int, default=3, help='Tournament selection size.')
    parser.add_argument('--convergence_threshold', type=float, default=0.1, help='Convergence threshold.')
    parser.add_argument('--max_no_improvement', type=int, default=10, help='Max generations with no improvement.')
    parser.add_argument('--crossover_type', type=str, default='one_point', choices=['one_point', 'two_point', 'uniform'], help='Type of crossover.')

    args = parser.parse_args()

    # Create an instance of the airfoil optimization class with provided parameters
    ga_optimizer = AirfoilGAOptimization(alpha=args.alpha, Re=args.Re)
    
    start_time = time.time()

    # Run the optimization with provided GA parameters
    best_individual = ga_optimizer.optimize(
        mu=args.mu,
        lambda_=args.lambda_,
        generations=args.generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        elitism_ratio=args.elitism_ratio,
        tournament_size=args.tournament_size,
        convergence_threshold=args.convergence_threshold,
        max_no_improvement=args.max_no_improvement,
        crossover_type=args.crossover_type
    )

    end_time = time.time()

    total_time = end_time - start_time
    
    # Decode the best individual to get the airfoil parameters
    m, p, t = ga_optimizer.decode(best_individual)
    
    # Print the best solution
    print("\nBest individual (Gray-coded):", best_individual)
    print(f"Best airfoil parameters: m={m}, p={p}, t={t}")
    print("Best fitness (Cl/Cd ratio):", ga_optimizer.fitness(best_individual))
    print(f"Execution time: {total_time:.2f} seconds.")
    
    # Plot the fitness improvement over generations
    generations_range = range(1, len(ga_optimizer.fitness_history) + 1)
    mean_fitness = [mean for mean, _ in ga_optimizer.fitness_history]
    max_fitness = [max_val for _, max_val in ga_optimizer.fitness_history]

    plt.figure(figsize=(10, 6))
    plt.plot(generations_range, mean_fitness, label='Mean Fitness', color='blue')
    plt.plot(generations_range, max_fitness, label='Max Fitness', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Cl/Cd)')
    plt.title('Fitness Improvement Over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Adjust y-axis limits to better visualize negative and positive fitness values
    combined_fitness = mean_fitness + max_fitness
    min_fitness = min(combined_fitness)
    max_fitness_plot = max(combined_fitness)
    plt.ylim(min_fitness - 0.1 * abs(min_fitness), max_fitness_plot + 0.1 * abs(max_fitness_plot))

    plt.savefig('fitness_improvement.png')

    plt.close()

if __name__ == "__main__":
    main()