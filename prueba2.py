import numpy as np
import matplotlib.pyplot as plt


def naca4(m, p, t, c=1.0, n_points=100):
    x = np.linspace(0, c, n_points)
    yt = 5 * t * (0.2969 * np.sqrt(x / c) - 0.1260 * (x / c) - 0.3516 * (x / c)**2 + 0.2843 * (x / c)**3 - 0.1015 * (x / c)**4)
    yc = np.where(x <= p * c, m / p**2 * (2 * p * (x / c) - (x / c)**2), m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * (x / c) - (x / c)**2))
    dyc_dx = np.where(x <= p * c, 2 * m / p**2 * (p - x / c), 2 * m / (1 - p)**2 * (p - x / c))
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return np.concatenate([xu, xl[::-1]]), np.concatenate([yu, yl[::-1]])


def compute_panel_method(x_coords, y_coords, U_inf=1.0):
    # Number of panels
    num_panels = len(x_coords) - 1
    
    # Panel angles and lengths
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    panel_length = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    
    # Compute control points (midpoints of each panel)
    x_mid = 0.5 * (x_coords[:-1] + x_coords[1:])
    y_mid = 0.5 * (y_coords[:-1] + y_coords[1:])
    
    # Solve for circulation and velocity distribution (simple linear system)
    # Matrix construction for influence coefficients should go here (omitted for brevity)
    # Solve for vortex strengths and then compute velocity field
    
    # Placeholder values for pressure coefficient Cp computation
    Cp = 1 - (U_inf / U_inf)**2  # Modify with velocity field calculation

    return Cp, x_mid, y_mid

# NACA 4415 Airfoil parameters
x_coords, y_coords = naca4(m=0.04, p=0.4, t=0.15, n_points=150)

# Compute pressure distribution using panel method
Cp, x_mid, y_mid = compute_panel_method(x_coords, y_coords)

# Plot the pressure distribution
plt.figure(figsize=(8, 6))
plt.plot(x_mid, Cp, label="Pressure Coefficient", marker='o')
plt.gca().invert_yaxis()
plt.xlabel('x/c (Chordwise Position)')
plt.ylabel('Cp (Pressure Coefficient)')
plt.title('Pressure Distribution on NACA Airfoil')
plt.grid(True)
plt.legend()
plt.show()
