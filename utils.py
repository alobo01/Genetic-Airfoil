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

    # Stall and max lift assumptions (improvement opportunity)
    Cl_max = 2 * np.pi * np.radians(15)  # Simplified assumption of max Cl
    Cl = 2 * np.pi * (alpha_rad - alpha_0) / beta
    
    # Nonlinear stall correction
    Cl = Cl_max * np.tanh(Cl / Cl_max)

    # Drag estimation: skin friction, pressure drag, induced drag
    Cf = (0.074 * Re**-0.2) / (1 + (Re / 3.7e5)**0.62)
    FF = (1 + 2.7*t + 100*t**4)
    
    # Quadratic drag increase due to effective camber
    Cd_p = 2 * eps**2 * ((alpha - np.degrees(alpha_0)) / 0.15)**2

    # Induced drag, with aspect ratio and Oswald efficiency factor
    AR = 8
    e = 0.9
    Cd_i = Cl**2 / (np.pi * AR * e)
    
    # Wave drag based on Mach number if beyond drag divergence Mach
    Mdd = 0.87 - 0.108 * t - 0.1 * Cl
    Cd_w = 0 if M < Mdd else 20 * (M - Mdd)**4

    # Total drag coefficient
    Cd = Cf * FF + Cd_p + Cd_i + Cd_w

    # Moment coefficient
    Cm = -0.1 * Cl - 0.25 * eps

    return Cl, Cd, Cm

def objective_function(params, alpha, Re, M):
    m, p, t = params
    Cl, Cd, _ = calculate_coefficients(alpha, m, p, t, Re, M)
    return -Cl / Cd  # Negative because we want to maximize L/D