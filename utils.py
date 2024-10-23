from matplotlib import pyplot as plt
import numpy as np

# Define a function to rotate the coordinates by a given angle
def rotate_airfoil(x, y, alpha):
    alpha_rad = np.radians(alpha)  # Convert angle of attack to radians
    x_new = x * np.cos(alpha_rad) - y * np.sin(alpha_rad)
    y_new = x * np.sin(alpha_rad) + y * np.cos(alpha_rad)
    return x_new, y_new

def naca_4digit(m, p, t, x):
    # Calculate camber line
    yc = np.where(x < p,
                  m * (2*p*x - x**2) / p**2,
                  m * ((1-2*p) + 2*p*x - x**2) / (1-p)**2)
    
    # Calculate gradient of camber line
    dyc_dx = np.where(x < p,
                      2*m * (p-x) / p**2,
                      2*m * (p-x) / (1-p)**2)
    
    theta = np.arctan(dyc_dx)
    
    # Calculate thickness distribution
    yt = 5*t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    
    # Calculate upper and lower surface coordinates
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    
    return xu, yu, xl, yl, yc

def calculate_coefficients(alpha, m, p, t, Re, M):
    """
    Calculate lift, drag, and moment coefficients using empirical formulas,
    adjusted to match the assumptions of a 2D potential flow analysis.

    Parameters:
    alpha (float): Angle of attack in degrees.
    m (float): Maximum camber as a percentage of the chord (e.g., 2 for 2%).
    p (float): Location of maximum camber along the chord in tenths (e.g., 4 for 0.4c).
    t (float): Maximum thickness as a percentage of the chord (e.g., 12 for 12%).
    Re (float): Reynolds number.
    M (float): Mach number.

    Returns:
    Cl (float): Lift coefficient.
    Cd (float): Drag coefficient.
    Cm (float): Moment coefficient about the quarter chord.
    """
    # Convert percentages to decimals
    m, p, t = m / 100, p / 10, t / 100

    alpha_rad = np.radians(alpha)

    # Prandtl-Glauert compressibility correction for subsonic flows
    if M < 1:
        beta = np.sqrt(1 - M**2)
    else:
        beta = 1  # Avoid imaginary values for supersonic speeds

    # Estimate effective camber and angle of zero lift
    eps = m / 0.9  # Approximation for camber line effects
    alpha_0 = -2 * np.pi * eps  # Effective zero-lift angle

    # Lift coefficient without stall correction
    Cl = 2 * np.pi * (alpha_rad - alpha_0) / beta

    # Remove nonlinear stall correction for consistency with panel method
    # Cl = Cl_max * np.tanh(Cl / Cl_max)  # Removed

    # Drag estimation: Only skin friction drag is considered
    # Adjust skin friction coefficient using Blasius solution for laminar flow
    Cf = 1.328 / np.sqrt(Re)

    # Form factor to account for thickness effects
    FF = (1 + 2.7 * t + 100 * t**4)

    # Exclude pressure drag due to camber
    Cd_p = 0  # Ignored for consistency

    # Exclude induced drag for 2D analysis
    Cd_i = 0  # For 2D flow

    # Exclude wave drag for incompressible flow
    Cd_w = 0  # For M < 0.3

    # Total drag coefficient: Only skin friction drag included
    Cd = Cf * FF

    # Moment coefficient about the quarter chord
    Cm = -0.1 * Cl - 0.25 * eps  # Retained as per original function

    return Cl, Cd, Cm

def calculate_coefficients_from_xy(alpha, x_coords, y_coords, Re, M):
    """
    Calculate lift, drag, and moment coefficients from airfoil geometry using the vortex panel method,
    adjusted to match the assumptions of a 2D potential flow analysis consistent with the first function.

    Parameters:
    alpha (float): Angle of attack in degrees (-5 to 15 degrees).
    x_coords (array): x-coordinates of the airfoil surface.
    y_coords (array): y-coordinates of the airfoil surface.
    Re (float): Reynolds number.
    M (float): Mach number.

    Returns:
    Cl (float): Lift coefficient.
    Cd (float): Drag coefficient.
    Cm (float): Moment coefficient about the quarter chord.
    """
    # Ensure alpha is within the specified range
    if not (-5 <= alpha <= 15):
        raise ValueError("Alpha must be between -5 and 15 degrees.")

    # Convert angle of attack to radians
    alpha_rad = np.radians(alpha)

    # Number of panels
    N = len(x_coords) - 1

    # Initialize arrays
    S = np.zeros(N)         # Length of each panel
    delta = np.zeros(N)     # Angle of each panel
    xc = np.zeros(N)        # x-coordinate of control point
    yc = np.zeros(N)        # y-coordinate of control point

    for i in range(N):
        dx = x_coords[i+1] - x_coords[i]
        dy = y_coords[i+1] - y_coords[i]
        S[i] = np.sqrt(dx**2 + dy**2)
        delta[i] = np.arctan2(dy, dx)
        xc[i] = 0.5 * (x_coords[i] + x_coords[i+1])
        yc[i] = 0.5 * (y_coords[i] + y_coords[i+1])

    # Initialize influence coefficient matrices
    CN1 = np.zeros((N, N))
    CN2 = np.zeros((N, N))
    CT1 = np.zeros((N, N))
    CT2 = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                CN1[i, j] = -1.0
                CN2[i, j] = 1.0
                CT1[i, j] = 0.5 * np.pi
                CT2[i, j] = 0.5 * np.pi
            else:
                A = -(xc[i] - x_coords[j]) * np.cos(delta[j]) - (yc[i] - y_coords[j]) * np.sin(delta[j])
                B = (xc[i] - x_coords[j])**2 + (yc[i] - y_coords[j])**2
                C = np.sin(delta[i] - delta[j])
                D = np.cos(delta[i] - delta[j])
                E = (xc[i] - x_coords[j]) * np.sin(delta[j]) - (yc[i] - y_coords[j]) * np.cos(delta[j])
                F = np.log(1.0 + S[j] * (S[j] + 2.0 * A) / B)
                G = np.arctan2(E * S[j], B + A * S[j])

                P = (xc[i] - x_coords[j]) * np.sin(delta[i] - 2.0 * delta[j]) + (yc[i] - y_coords[j]) * np.cos(delta[i] - 2.0 * delta[j])
                Q = (xc[i] - x_coords[j]) * np.cos(delta[i] - 2.0 * delta[j]) - (yc[i] - y_coords[j]) * np.sin(delta[i] - 2.0 * delta[j])

                CN2[i, j] = D + 0.5 * F * C - G * S[j] * C / B
                CN1[i, j] = D - 0.5 * F * C + G * S[j] * C / B
                CT2[i, j] = C - 0.5 * F * D + G * S[j] * D / B
                CT1[i, j] = C + 0.5 * F * D - G * S[j] * D / B

    # Assemble the linear system
    A = np.zeros((N+1, N+1))
    B = np.zeros(N+1)

    # No-penetration boundary condition
    for i in range(N):
        for j in range(N):
            A[i, j] = CN1[i, j] + CN2[i, j]
        # Enforce the Kutta condition (vorticity conservation)
        A[i, N] = 0.0
        B[i] = np.sin(delta[i] - alpha_rad)

    # Kutta condition
    for j in range(N):
        A[N, j] = CT1[0, j] + CT2[0, j] + CT1[-1, j] + CT2[-1, j]
    A[N, N] = 0.0
    B[N] = -np.cos(delta[0] - alpha_rad) - np.cos(delta[-1] - alpha_rad)

    # Solve for vortex strengths
    gamma = np.linalg.solve(A, B)

    # Tangential velocity on each panel
    Vt = np.zeros(N)
    for i in range(N):
        Vt[i] = np.cos(delta[i] - alpha_rad)
        for j in range(N):
            Vt[i] += (CT1[i, j] + CT2[i, j]) * gamma[j]
        Vt[i] += gamma[N] * (CT1[i, N-1] + CT2[i, N-1])

    # Pressure coefficient
    Cp = 1 - Vt**2

    # Calculate lift and moment coefficients
    Cl = 0.0
    Cm = 0.0
    c = max(x_coords) - min(x_coords)  # Chord length
    x_ref = min(x_coords) + 0.25 * c   # Quarter chord point

    for i in range(N):
        dCp = Cp[i] * S[i]

        # Lift coefficient
        Cl += dCp * np.sin(delta[i])

        # Moment coefficient about quarter chord
        xi = xc[i] - x_ref
        Cm += -Cp[i] * xi * S[i] * np.cos(delta[i])

    # Normalize coefficients
    Cl /= c
    Cm /= c**2

    # Compressibility correction using Prandtl-Glauert rule
    if M < 1:
        beta = np.sqrt(1 - M**2)
        Cl /= beta
        Cm /= beta

    # Adjust skin friction drag calculation
    Cf = 1.328 / np.sqrt(Re)

    # Form factor FF (assuming thin airfoil)
    FF = 1.0

    # Wetted area
    wetted_area = sum(S)

    # Total drag coefficient (only skin friction drag included)
    Cd = Cf * FF * (wetted_area / c)

    return Cl, Cd, Cm

def objective_function(params, alpha, Re, M):
    m, p, t = params
    Cl, Cd, _ = calculate_coefficients(alpha, m, p, t, Re, M)
    return -Cl / Cd  # Negative because we want to maximize L/D

def compute_coordinates(magnitudes):
    """
    Computes the x, y coordinates for a list of magnitudes with a constant angle step.

    Args:
    - magnitudes (list or numpy array): A vector of real numbers representing magnitudes.

    Returns:
    - coordinates (list of tuples): List of (x, y) coordinates corresponding to the magnitudes.
    """
    coordinates_x, coordinates_y = [],[]
    current_angle = 0  # Start at 0 radians
    angle_step = 2*np.pi / len(magnitudes)

    for magnitude in magnitudes:
        x = magnitude * np.cos(current_angle)
        y = magnitude * np.sin(current_angle)
        coordinates_x.append(x)
        coordinates_y.append(y)
        current_angle += angle_step  # Increment the angle by the angle step

    return coordinates_x, coordinates_y

def save_to_xfoil_dat(filename, coordinates_x, coordinates_y):
    """
    Saves the x, y coordinates to a .dat file in a format readable by XFOIL.

    Args:
    - filename (str): The name of the output file (e.g., "output.dat").
    - coordinates_x (list): List of x coordinates.
    - coordinates_y (list): List of y coordinates.
    """
    if len(coordinates_x) != len(coordinates_y):
        raise ValueError("The lengths of the x and y coordinates must be the same.")

    with open(filename, 'w') as file:
        for x, y in zip(coordinates_x, coordinates_y):
            file.write(f"{x:.6f} {y:.6f}\n")  # XFOIL expects coordinates with high precision

    print(f"Coordinates saved to {filename} successfully.")


if __name__ == "__main__":
    magnitudes = np.full(100,1)
    x,y = compute_coordinates(magnitudes)
    save_to_xfoil_dat("circle.dat",x,y)
    plt.plot(x,y)
    plt.tight_layout()
    plt.show(block=True)

