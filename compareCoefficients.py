import numpy as np
from utils import calculate_coefficients, calculate_coefficients_from_xy, naca_4digit

# Define the range of NACA airfoil parameters to test
# m: maximum camber as percentage of the chord (0 to 10%)
# p: location of maximum camber as percentage of the chord (0 to 40%)
# t: maximum thickness as percentage of the chord (6% to 18%)
m_values = [2, 4, 6, 8, 10, 12]   # Maximum camber percentages
p_values = [1, 10, 20, 30, 40]    # Maximum camber positions
t_values = [6, 9, 12, 15, 18]     # Maximum thickness percentages

# Angle of attack in degrees
alpha = 5

# Reynolds number and Mach number (assuming subsonic, incompressible flow)
Re = 1e6
M = 0.1

# Initialize lists to store the squared errors
errors_Cl = []
errors_Cd = []
errors_Cm = []

# Loop over different combinations of m, p, and t to test various NACA airfoils
for m in m_values:
    for p in p_values:
        for t in t_values:
            # Skip invalid combinations (e.g., p=0 when m>0)
            if m > 0 and p == 0:
                continue

            # Calculate coefficients using the analytical method
            # Theory: Based on thin airfoil theory and empirical correlations
            Cl_analytical, Cd_analytical, Cm_analytical = calculate_coefficients(
                alpha, m, p, t, Re, M)

            # Generate airfoil coordinates using the NACA 4-digit equation
            # Theory: Standard geometric definition of NACA 4-digit airfoils
            x = np.linspace(0, 1, 200)  # Chordwise positions
            xu, yu, xl, yl, yc = naca_4digit(m/100, p/10, t/100, x)

            # Combine upper and lower surfaces for panel method
            # Theory: Need a closed loop of coordinates for the panel method
            x_coords = np.concatenate((xu, xl[::-1]))
            y_coords = np.concatenate((yu, yl[::-1]))

            # Calculate coefficients using the numerical method
            # Theory: Panel method and potential flow theory
            Cl_numerical, Cd_numerical, Cm_numerical = calculate_coefficients_from_xy(
                alpha, x_coords, y_coords, Re, M)

            # Calculate squared errors for each coefficient
            error_Cl = (Cl_analytical - Cl_numerical)**2
            error_Cd = (Cd_analytical - Cd_numerical)**2
            error_Cm = (Cm_analytical - Cm_numerical)**2

            # Append errors to the lists
            errors_Cl.append(error_Cl)
            errors_Cd.append(error_Cd)
            errors_Cm.append(error_Cm)

            # Print the errors
            print(f"Error in Cl: {error_Cl}")
            print(f"Error in Cd: {error_Cd}")
            print(f"Error in Cm: {error_Cm}")
            

# Calculate mean square errors for each coefficient
mse_Cl = np.mean(errors_Cl)
mse_Cd = np.mean(errors_Cd)
mse_Cm = np.mean(errors_Cm)

# Print the mean square errors
print(f"Mean Square Error in Cl: {mse_Cl}")
print(f"Mean Square Error in Cd: {mse_Cd}")
print(f"Mean Square Error in Cm: {mse_Cm}")