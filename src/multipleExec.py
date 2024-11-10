import subprocess
import matplotlib.pyplot as plt

# Parameters for GA method
angles_of_attack = list(range(0, 16))  # Angles from 0 to 15
ga_results = []
naca_series = []

# Function to run GA method for a given angle and store the best fitness and NACA series
def run_ga(alpha):
    command = [
        "python", "GA.py",
        "--alpha", str(alpha),
        "--Re", "1e6",
        "--lambda_", "40",
        "--mu", "20",
        "--generations", "50",
        "--crossover_type", "two_point"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    output = result.stdout
    print(f"GA Output for alpha={alpha}:\n{output}")  # Debug print
    
    fitness_value = None
    m, p, t = None, None, None  # Initialize m, p, t to capture the airfoil parameters
    
    # Extract best fitness and NACA parameters from output
    for line in output.splitlines():
        if line.startswith("Best fitness (Cl/Cd ratio):"):
            try:
                fitness_value = float(line.split(":")[-1].strip())
            except ValueError:
                print(f"Warning: Could not parse fitness value for alpha = {alpha}. Output line: {line}")
        elif line.startswith("Best airfoil parameters:"):
            # Extract m, p, t values
            parts = line.split("m=")[1].split(", ")
            m = int(parts[0])
            p = int(parts[1].split("=")[-1])
            t = int(parts[2].split("=")[-1])
    
    # Construct the NACA code if parameters are found
    if m is not None and p is not None and t is not None:
        naca_code = f"{m}{p}{t:02d}"
        naca_series.append(naca_code)
    else:
        naca_series.append("N/A")  # Placeholder if parsing failed
    
    return fitness_value

# Run GA method for each angle of attack and collect results
for alpha in angles_of_attack:
    ga_fitness = run_ga(alpha)
    ga_results.append(ga_fitness if ga_fitness is not None else float('-inf'))

print("GA Results:", ga_results)       # Debug print for fitness values
print("NACA Series:", naca_series)     # Print the list of NACA series

# Plotting the GA best fitness values for each angle of attack
plt.figure(figsize=(10, 6))
plt.plot(angles_of_attack, ga_results, label="GA (Fitness)", marker='x')
plt.xlabel("Angle of Attack (Degrees)")
plt.ylabel("Best Fitness Value (Cl/Cd)")
plt.title("Best Fitness Values at Different Angles of Attack (GA Only)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
