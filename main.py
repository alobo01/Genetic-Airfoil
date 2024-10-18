import numpy as np
import matplotlib.pyplot as plt
from pyxfoil import Xfoil, set_workdir, set_xfoilexe 

# Set the working directory for pyXFOIL
set_workdir('XFOIL')
set_xfoilexe('XFOIL\\xfoil.exe')

# Function to generate NACA airfoil points and save them to a DAT file
def generate_naca_airfoil(naca_code):
    # Extract parameters from the NACA code
    if len(naca_code) == 4:
        m = int(naca_code[0]) / 100.0  # Maximum camber
        p = int(naca_code[1]) / 10.0    # Position of maximum camber
        t = int(naca_code[2:]) / 100.0  # Thickness
        
        # Number of points on the airfoil
        num_points = 180
        x = np.linspace(0, 1, num_points)
        
        # Calculate camber line
        yc = np.where(x < p, m / (p**2) * (2 * p * x - x**2), 
                      m / ((1 - p)**2) * ((1 - 2 * p) + 2 * p * x - x**2))
        
        # Calculate thickness distribution
        yt = (t / 0.2) * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                          0.2843 * x**3 - 0.1015 * x**4)
        
        # Upper and lower surface coordinates
        xu = x - yt * np.sin(np.arctan(np.gradient(yc)))
        yu = yc + yt * np.cos(np.arctan(np.gradient(yc)))
        xl = x + yt * np.sin(np.arctan(np.gradient(yc)))
        yl = yc - yt * np.cos(np.arctan(np.gradient(yc)))
        
        # Combine upper and lower surface points
        points = np.vstack((xu, yu)).T.tolist() + np.vstack((xl[::-1], yl[::-1])).T.tolist()
        
        # Save points to a DAT file
        with open(f'./XFOIL/NACA_{naca_code}.dat', 'w') as f:
            f.write(f'{len(points)}\n')
            for point in points:
                f.write(f'{point[0]} {point[1]}\n')
        
        return f'./XFOIL/NACA_{naca_code}.dat'
    else:
        raise ValueError("NACA code must be 4 digits long.")

# Function to compute the lift/drag ratio and pressure distribution using pyXFOIL
def compute_airfoil_performance(naca_code, angle_of_attack, Re=1e6, mach=0.1):
    # Generate airfoil points and get DAT file path
    dat_file_path = generate_naca_airfoil(naca_code)

    # Initialize XFoil with NACA airfoil code
    xfoil = Xfoil(f'NACA {naca_code}')
    
    # Load points from DAT file
    xfoil.points_from_dat(dat_file_path)
    xfoil.set_ppar(180)  # Set the number of points

    # Run XFOIL for the given angle of attack, Mach number, and Reynolds number
    result = xfoil.run_result(angle_of_attack, Re)

    # Extract lift and drag coefficients from the result
    cl = result.cl
    cd = result.cd
    ld_ratio = cl / cd if cd != 0 else np.inf  # prevent division by zero
    
    # Extract airfoil coordinates and pressure coefficient (Cp) distribution
    x, y = xfoil.get_coords()
    cp_upper, cp_lower = result.get_cp_distribution()

    return ld_ratio, x, y, cp_upper, cp_lower

# Plotting function for pressure field
def plot_airfoil_pressure_field(x, y, cp_upper, cp_lower, naca_code, angle_of_attack, ld_ratio):
    # Create a finer grid for the flow field (adjusted for better visualization)
    X, Y = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 0.5, 200))
    
    # Simplified pressure field calculation (visualization purposes)
    Z = np.zeros_like(X)
    
    # Plot the pressure coefficient contours
    plt.figure(figsize=(10, 6))
    cp_plot = plt.contourf(X, Y, Z, levels=np.linspace(-1.5, 1.2, 50), cmap='coolwarm', alpha=0.7)
    plt.colorbar(cp_plot, label='Pressure Coefficient (Cp)')

    # Plot the airfoil shape
    plt.plot(x, y, color='black', lw=2, label='Airfoil Upper Surface')
    plt.plot(x, -y, color='black', lw=2, label='Airfoil Lower Surface')

    # Annotate the upper and lower pressure coefficients
    plt.scatter(x, y, c=cp_upper, cmap='coolwarm', edgecolors='k', label='Upper Cp')
    plt.scatter(x, -y, c=cp_lower, cmap='coolwarm', edgecolors='k', label='Lower Cp')

    plt.title(f'NACA {naca_code} - AoA: {angle_of_attack}° - L/D Ratio: {ld_ratio:.2f}')
    plt.xlabel('X-axis (Chord)')
    plt.ylabel('Y-axis (Thickness)')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

# Main function to compute and plot
def main():
    naca_code = "2412"  # NACA airfoil code
    angle_of_attack = 5.0  # Angle of attack in degrees
    Re = 1e6  # Reynolds number
    mach = 0.1  # Mach number
    
    # Compute performance using pyXFOIL
    ld_ratio, x, y, cp_upper, cp_lower = compute_airfoil_performance(naca_code, angle_of_attack, Re, mach)
    
    # Plot the airfoil with pressure field and Cp distribution
    plot_airfoil_pressure_field(x, y, cp_upper, cp_lower, naca_code, angle_of_attack, ld_ratio)

# Run the main function
if __name__ == "__main__":
    main()
