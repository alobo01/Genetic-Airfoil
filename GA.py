import random
from abc import ABC, abstractmethod
import utils

class GeneticAlgorithm(ABC):
    def __init__(self):
        self.fitness_history = []

    @abstractmethod
    def fitness(self, individual):
        pass

    @abstractmethod
    def create_individual(self):
        pass

    def create_population(self, size):
        return [self.create_individual() for _ in range(size)]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def binomial_mutation(self, individual, mutpb):
        return [1 - bit if random.random() < mutpb else bit for bit in individual]

    def tournament_selection(self, population, tournament_size):
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda ind: self.fitness(ind))
        return winner

    def detect_convergence(self, threshold=0.001, max_no_improvement=10):
        if len(self.fitness_history) < max_no_improvement:
            return False

        mean_fitnesses = [mean for mean, _ in self.fitness_history[-max_no_improvement:]]
        max_fitnesses = [max_val for _, max_val in self.fitness_history[-max_no_improvement:]]

        # Check relative improvement in fitness
        mean_change = abs(mean_fitnesses[-1] - mean_fitnesses[0]) / mean_fitnesses[0]
        max_change = abs(max_fitnesses[-1] - max_fitnesses[0]) / max_fitnesses[0]

        return mean_change < threshold and max_change < threshold

    def optimize(self, mu=20, lambda_=40, generations=50, crossover_prob=0.8, mutation_prob=0.1, elitism_ratio=0.1, tournament_size=3, convergence_threshold=0.001, max_no_improvement=10):
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

# Gray coding
def binary_to_gray(n):
    return n ^ (n >> 1)

def gray_to_binary(n):
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return n

def int_to_gray_bits(n, num_bits):
    gray = binary_to_gray(n)
    return [int(bit) for bit in f"{gray:0{num_bits}b}"]

def gray_bits_to_int(bits):
    gray = int("".join(map(str, bits)), 2)
    return gray_to_binary(gray)

class AirfoilGAOptimization(GeneticAlgorithm):
    def __init__(self, alpha, Re, M):
        super().__init__()
        self.alpha = alpha
        self.Re = Re
        self.M = M
        self.m_bits = 4 
        self.p_bits = 4  
        self.t_bits = 7 
        self.n_bits = self.m_bits + self.p_bits + self.t_bits

    def fitness(self, individual):
        m, p, t = self.decode(individual)
        Cl, Cd, _ = utils.calculate_coefficients(self.alpha, m, p, t, self.Re, self.M)
        return Cl / Cd

    def create_individual(self):
        return [random.randint(0, 1) for _ in range(self.n_bits)]

    def decode(self, individual):
        m_bits = individual[:self.m_bits]
        p_bits = individual[self.m_bits:self.m_bits + self.p_bits]
        t_bits = individual[self.m_bits + self.p_bits:]

        # Convert Gray-coded bits to integer and scale to appropriate ranges
        m = round(gray_bits_to_int(m_bits) / (2**self.m_bits - 1) * 9)
        p = round(gray_bits_to_int(p_bits) / (2**self.p_bits - 1) * 9 + 1)
        t = round(gray_bits_to_int(t_bits) / (2**self.t_bits - 1) * 18 + 6)
        
        return m, p, t

    def encode(self, m, p, t):
        m_bits = int_to_gray_bits(int(m / 9 * (2**self.m_bits - 1)), self.m_bits)
        p_bits = int_to_gray_bits(int((p - 1) / 9 * (2**self.p_bits - 1)), self.p_bits)
        t_bits = int_to_gray_bits(int((t - 6) / 18 * (2**self.t_bits - 1)), self.t_bits)
        return m_bits + p_bits + t_bits