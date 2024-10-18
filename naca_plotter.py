import numpy as np
import matplotlib.pyplot as plt


def plot_airfoil_from_dat(file_name):
    # Read the coordinates from the .dat file
    coordinates = []
    with open(file_name, 'r') as file:
        # Skip the header lines
        for _ in range(2):
            next(file)
        # Read the coordinates
        for line in file:
            x, y = map(float, line.split())
            coordinates.append((x, y))

    # Separate the coordinates into x and y lists
    x_coords, y_coords = zip(*coordinates)

    # Plot the airfoil
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, marker='o', markersize=3, linestyle='-', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title(f'Airfoil Plot: {file_name}', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def generate_naca_airfoil_coordinates(max_camber=8, max_camber_position=20, thickness=30,
                                      num_points=120, cosine_spacing=True, close_trailing_edge=False,
                                      file_name="airfoil.dat"):
    # Validate input parameters
    if not (0 <= max_camber <= 9.5):
        raise ValueError("Max Camber must be between 0% and 9.5%.")
    
    if not (0 <= max_camber_position <= 90):
        raise ValueError("Max Camber Position must be between 0% and 90%.")
    
    if not (1 <= thickness <= 40):
        raise ValueError("Thickness must be between 1% and 40%.")
    
    if not (20 <= num_points <= 200):
        raise ValueError("Number of points must be between 20 and 200.")

    # Normalize the input values for use in calculations
    m = max_camber / 100.0  # Maximum camber as a fraction
    p = max_camber_position / 100.0  # Maximum camber position as a fraction
    t = thickness / 100.0  # Maximum thickness as a fraction

    # Constants for the thickness equation
    a0 = 0.2969
    a1 = -0.126
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1036 if close_trailing_edge else -0.1015  # Adjust for closed or open trailing edge

    # Generate x-coordinates using either cosine or linear spacing
    if cosine_spacing:
        beta = np.linspace(0, np.pi, num_points)
        x = (1 - np.cos(beta)) / 2  # Cosine spacing for better resolution near the leading edge
    else:
        x = np.linspace(0, 1, num_points)  # Linear spacing

    # Initialize arrays for upper and lower surfaces
    upper_surface = np.zeros((num_points, 2))
    lower_surface = np.zeros((num_points, 2))

    # Calculate thickness distribution yt
    yt = (t / 0.2) * (a0 * np.sqrt(x) + a1 * x + a2 * x**2 + a3 * x**3 + a4 * x**4)

    # Calculate camber line yc and its slope dyc/dx
    yc = np.where(x < p, 
                  (m / p**2) * (2 * p * x - x**2), 
                  (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc_dx = np.where(x < p, 
                      (2 * m / p**2) * (p - x), 
                      (2 * m / (1 - p)**2) * (p - x))

    # Calculate the angle theta of the camber line slope
    theta = np.arctan(dyc_dx)

    # Compute the upper and lower surface coordinates
    upper_surface[:, 0] = x - yt * np.sin(theta)
    upper_surface[:, 1] = yc + yt * np.cos(theta)
    lower_surface[:, 0] = x + yt * np.sin(theta)
    lower_surface[:, 1] = yc - yt * np.cos(theta)

    # Combine the upper and lower surfaces into one array, starting from the trailing edge at the top
    airfoil_coordinates = np.vstack([upper_surface[::-1], lower_surface[1:]])

    # Save the coordinates to a .dat file
    with open(file_name, 'w') as file:
        file.write(f"NACA Airfoil Coordinates (Max Camber: {max_camber}%, Position: {max_camber_position}%, Thickness: {thickness}%)\n")
        file.write("   X         Y\n")
        for coord in airfoil_coordinates:
            file.write(f"{coord[0]:.6f}   {coord[1]:.6f}\n")

    print(f"Airfoil coordinates saved to {file_name}")

if __name__ == "__main__":
    # Example usage
    generate_naca_airfoil_coordinates(max_camber=2, max_camber_position=40, thickness=12,
                                  num_points=81, cosine_spacing=True, close_trailing_edge=True,
                                  file_name="airfoil.dat")

    plot_airfoil_from_dat("airfoil.dat")