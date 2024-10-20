import random
from abc import ABC, abstractmethod
import utils

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

    def crossover(self, parent1, parent2):
        """Performs crossover between two parents to produce two offspring.

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
        mean_change = abs(mean_fitnesses[-1] - mean_fitnesses[0]) / mean_fitnesses[0]
        max_change = abs(max_fitnesses[-1] - max_fitnesses[0]) / max_fitnesses[0]

        return mean_change < threshold and max_change < threshold

    def optimize(self, mu=20, lambda_=40, generations=50, crossover_prob=0.8, mutation_prob=0.1,
                 elitism_ratio=0.1, tournament_size=3, convergence_threshold=0.001, max_no_improvement=10):
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

        while actual_generation < generations:
            # Evaluate fitness of current population
            fitness_values = [self.fitness(ind) for ind in population]
            mean_fitness = sum(fitness_values) / len(fitness_values)
            max_fitness = max(fitness_values)
            self.fitness_history.append((mean_fitness, max_fitness))

            # Convergence check
            if self.detect_convergence(convergence_threshold, max_no_improvement):
                print(f"Convergence detected at generation {actual_generation}.")
                break

            # Elitism: keep the best individuals
            num_elite = int(mu * elitism_ratio)
            population = sorted(population, key=lambda ind: self.fitness(ind), reverse=True)
            next_gen = population[:num_elite]

            # Generate offspring
            offspring = []
            while len(offspring) < lambda_:
                # Parent selection
                parent1 = self.tournament_selection(population, tournament_size)
                parent2 = self.tournament_selection(population, tournament_size)

                # Crossover
                if random.random() < crossover_prob:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]

                # Mutation
                child1 = self.binomial_mutation(child1, mutation_prob)
                child2 = self.binomial_mutation(child2, mutation_prob)

                offspring.extend([child1, child2])

            next_gen.extend(offspring[:mu - num_elite])  # Fill up to μ individuals
            population = next_gen[:mu]  # New generation
            actual_generation += 1

        best_individual = max(population, key=lambda ind: self.fitness(ind))
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
        m, p, t = self.decode(individual)
        Cl, Cd, _ = utils.calculate_coefficients(self.alpha, m, p, t, self.Re, self.M)
        return Cl / Cd

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
        p = round(gray_bits_to_int(p_bits) / (2**self.p_bits - 1) * 9 + 1)
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
        p_bits = int_to_gray_bits(int((p - 1) / 9 * (2**self.p_bits - 1)), self.p_bits)
        t_bits = int_to_gray_bits(int((t - 6) / 18 * (2**self.t_bits - 1)), self.t_bits)
        return m_bits + p_bits + t_bits
