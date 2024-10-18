import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import minimize
import time


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
    m = m / 100
    p = p / 10
    t = t / 100
    
    # Convert angle of attack to radians
    alpha_rad = np.deg2rad(alpha)
    
    # Prandtl-Glauert compressibility correction
    beta = np.sqrt(1 - M**2)
    
    # Estimate effective camber and angle of zero lift
    eps = m / 0.9  # Approximation of effective camber
    alpha_0 = -2 * np.pi * eps  # Angle of zero lift
    
    # Lift coefficient calculation (including stall effects)
    Cl_max = 2 * np.pi * (15 * np.pi/180 - alpha_0)  # Approximate max Cl
    Cl_linear = 2 * np.pi * (alpha_rad - alpha_0) / beta
    Cl = Cl_max * np.tanh(Cl_linear / Cl_max)
    
    # Drag coefficient calculation
    # Skin friction coefficient (Schiller-Naumann correlation)
    Cf = (0.074 * (Re**-0.2)) / (1 + (Re / 3.7e5)**0.62)
    
    # Form factor for drag due to thickness
    FF = (1 + 2.7*t + 100*t**4)
    
    # Pressure drag coefficient
    Cd_p = 2 * eps**2 * ((alpha - np.rad2deg(alpha_0))/0.15)**2  # Quadratic in effective angle of attack
    
    # Induced drag coefficient
    e = 0.9  # Oswald efficiency factor
    AR = 8  # Assuming an aspect ratio of 8
    Cd_i = Cl**2 / (np.pi * e * AR)
    
    # Wave drag (rudimentary approximation)
    Mdd = 0.87 - 0.108 * t - 0.1 * Cl  # Drag divergence Mach number
    Cd_w = 0 if M < Mdd else 20 * (M - Mdd)**4
    
    # Total drag coefficient
    Cd = Cf * FF + Cd_p + Cd_i + Cd_w
    
    # Moment coefficient (approximation)
    Cm = -0.1 * Cl - 0.25 * eps  # Assuming the aerodynamic center is at 25% chord
    
    return Cl, Cd, Cm

def objective_function(params, alpha, Re, M):
    m, p, t = params
    Cl, Cd, _ = calculate_coefficients(alpha, m, p, t, Re, M)
    return -Cl / Cd  # Negative because we want to maximize L/D

def optimize_airfoil(alpha, Re, M, initial_m, initial_p, initial_t):
    initial_params = [initial_m, initial_p, initial_t]
    
    bounds = [(0, 9), (1, 9), (6, 24)]  # Bounds for m, p, t
    
    result = minimize(
        objective_function,
        initial_params,
        args=(alpha, Re, M),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100}
    )
    
    return result.x

def main():
    st.title("Comprehensive NACA 4-Digit Airfoil Analysis and Optimization")
    
    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        m = st.slider("Maximum Camber (% chord)", 0, 9, 2, 1)
        p = st.slider("Position of Max Camber (tenths of chord)", 1, 9, 4, 1)
        t = st.slider("Maximum Thickness (% chord)", 6, 24, 12, 1)
    with col2:
        alpha = st.slider("Design Angle of Attack (degrees)", -5.0, 20.0, 5.0, 0.1)
        Re = st.slider("Reynolds Number", 1e5, 1e8, 1e6, 1e5)
        M = st.slider("Mach Number", 0.0, 0.9, 0.2, 0.05)
    
    # Calculate coefficients for the original airfoil
    Cl, Cd, Cm = calculate_coefficients(alpha, m, p, t, Re, M)
    
    # Display results for the original airfoil
    st.subheader("Current Airfoil Results:")
    st.write(f"NACA {m}{p}{t:02d}")
    st.write(f"Lift Coefficient (Cl): {Cl:.4f}")
    st.write(f"Drag Coefficient (Cd): {Cd:.6f}")
    st.write(f"Moment Coefficient (Cm): {Cm:.4f}")
    st.write(f"Lift-to-Drag Ratio (L/D): {Cl/Cd:.2f}")
    
    # Optimization section
    st.subheader("Airfoil Optimization")
    if st.button("Optimize Airfoil"):
        with st.spinner("Optimizing airfoil... This may take a moment."):
            start_time = time.time()
            opt_m, opt_p, opt_t = optimize_airfoil(alpha, Re, M, m, p, t)
            end_time = time.time()
            
            opt_Cl, opt_Cd, opt_Cm = calculate_coefficients(alpha, opt_m, opt_p, opt_t, Re, M)
            
            st.success(f"Optimization completed in {end_time - start_time:.2f} seconds!")
            st.subheader("Optimized Airfoil Results:")
            st.write(f"Optimized NACA {opt_m:.0f}{opt_p:.0f}{opt_t:.0f}")
            st.write(f"Lift Coefficient (Cl): {opt_Cl:.4f}")
            st.write(f"Drag Coefficient (Cd): {opt_Cd:.6f}")
            st.write(f"Moment Coefficient (Cm): {opt_Cm:.4f}")
            st.write(f"Lift-to-Drag Ratio (L/D): {opt_Cl/opt_Cd:.2f}")
            
            # Comprehensive plotting
            fig = plt.figure(figsize=(20, 15))
            
            # 1. Airfoil Shape Comparison
            ax1 = fig.add_subplot(2, 3, 1)
            x = np.linspace(0, 1, 100)
            xu, yu, xl, yl, yc = naca_4digit(m/100, p/10, t/100, x)
            xu_opt, yu_opt, xl_opt, yl_opt, yc_opt = naca_4digit(opt_m/100, opt_p/10, opt_t/100, x)
            
            xu_rotated, yu_rotated = rotate_airfoil(xu, yu, -alpha)
            xl_rotated, yl_rotated = rotate_airfoil(xl, yl, -alpha)
            yc_rotated = rotate_airfoil(x, yc, -alpha)[1]  # Only y-coordinates are needed for yc

            # Rotate optimized airfoil coordinates
            xu_opt_rotated, yu_opt_rotated = rotate_airfoil(xu_opt, yu_opt, -alpha)
            xl_opt_rotated, yl_opt_rotated = rotate_airfoil(xl_opt, yl_opt, -alpha)
            yc_opt_rotated = rotate_airfoil(x, yc_opt, -alpha)[1]  # Only y-coordinates are needed for yc_opt

            # Plot original airfoil
            ax1.plot(xu_rotated, yu_rotated, 'b', label='Original')
            ax1.plot(xl_rotated, yl_rotated, 'b')
            ax1.plot(x, yc_rotated, 'b--')

            # Plot optimized airfoil
            ax1.plot(xu_opt_rotated, yu_opt_rotated, 'r', label='Optimized')
            ax1.plot(xl_opt_rotated, yl_opt_rotated, 'r')
            ax1.plot(x, yc_opt_rotated, 'r--')

            # Set labels and title
            ax1.set_xlabel("x/c")
            ax1.set_ylabel("y/c")
            ax1.set_title("Airfoil Shape Comparison")
            ax1.legend()
            ax1.axis('equal')
            ax1.grid(True)
            
            # 2. Cl vs Alpha
            ax2 = fig.add_subplot(2, 3, 2)
            alphas = np.linspace(-5, 20, 100)
            Cls_orig = [calculate_coefficients(a, m, p, t, Re, M)[0] for a in alphas]
            Cls_opt = [calculate_coefficients(a, opt_m, opt_p, opt_t, Re, M)[0] for a in alphas]
            ax2.plot(alphas, Cls_orig, 'b', label='Original')
            ax2.plot(alphas, Cls_opt, 'r', label='Optimized')
            ax2.set_xlabel("Angle of Attack (degrees)")
            ax2.set_ylabel("Lift Coefficient (Cl)")
            ax2.set_title("Lift Curve Comparison")
            ax2.legend()
            ax2.grid(True)
            
            # 3. Drag Polar
            ax3 = fig.add_subplot(2, 3, 3)
            Cds_orig = [calculate_coefficients(a, m, p, t, Re, M)[1] for a in alphas]
            Cds_opt = [calculate_coefficients(a, opt_m, opt_p, opt_t, Re, M)[1] for a in alphas]
            ax3.plot(Cds_orig, Cls_orig, 'b', label='Original')
            ax3.plot(Cds_opt, Cls_opt, 'r', label='Optimized')
            ax3.set_xlabel("Drag Coefficient (Cd)")
            ax3.set_ylabel("Lift Coefficient (Cl)")
            ax3.set_title("Drag Polar Comparison")
            ax3.legend()
            ax3.grid(True)
            
            # 4. L/D vs Alpha
            ax4 = fig.add_subplot(2, 3, 4)
            LD_orig = [cl/cd for cl, cd in zip(Cls_orig, Cds_orig)]
            LD_opt = [cl/cd for cl, cd in zip(Cls_opt, Cds_opt)]
            ax4.plot(alphas, LD_orig, 'b', label='Original')
            ax4.plot(alphas, LD_opt, 'r', label='Optimized')
            ax4.set_xlabel("Angle of Attack (degrees)")
            ax4.set_ylabel("Lift-to-Drag Ratio (L/D)")
            ax4.set_title("L/D vs Angle of Attack")
            ax4.legend()
            ax4.grid(True)
            
            # 5. Cm vs Alpha
            ax5 = fig.add_subplot(2, 3, 5)
            Cms_orig = [calculate_coefficients(a, m, p, t, Re, M)[2] for a in alphas]
            Cms_opt = [calculate_coefficients(a, opt_m, opt_p, opt_t, Re, M)[2] for a in alphas]
            ax5.plot(alphas, Cms_orig, 'b', label='Original')
            ax5.plot(alphas, Cms_opt, 'r', label='Optimized')
            ax5.set_xlabel("Angle of Attack (degrees)")
            ax5.set_ylabel("Moment Coefficient (Cm)")
            ax5.set_title("Moment Coefficient vs Angle of Attack")
            ax5.legend()
            ax5.grid(True)
            
            # 6. Pressure Distribution (at design alpha)
            ax6 = fig.add_subplot(2, 3, 6)
            # This is a simplified pressure distribution calculation
            Cp_orig = 1 - (yu-yl)**2
            Cp_opt = 1 - (yu_opt-yl_opt)**2
            ax6.plot(x, -Cp_orig, 'b', label='Original')
            ax6.plot(x, -Cp_opt, 'r', label='Optimized')
            ax6.set_xlabel("x/c")
            ax6.set_ylabel("-Cp")
            ax6.set_title(f"Pressure Distribution at α = {alpha}°")
            ax6.legend()
            ax6.grid(True)
            ax6.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Performance improvement summary
            st.subheader("Performance Improvement Summary:")
            st.write(f"Lift Coefficient Improvement: {(opt_Cl - Cl) / Cl * 100:.2f}%")
            st.write(f"Drag Coefficient Reduction: {(Cd - opt_Cd) / Cd * 100:.2f}%")
            st.write(f"L/D Ratio Improvement: {(opt_Cl/opt_Cd - Cl/Cd) / (Cl/Cd) * 100:.2f}%")
    
    # Original airfoil plots (without optimization)
    st.subheader("Original Airfoil Analysis")
    fig_orig = plt.figure(figsize=(20, 10))
    
    # 1. Airfoil Shape
    ax1 = fig_orig.add_subplot(2, 3, 1)
    x = np.linspace(0, 1, 100)
    xu, yu, xl, yl, yc = naca_4digit(m/100, p/10, t/100, x)
    xu_rotated,yu_rotated = rotate_airfoil(xu,yu,-alpha)
    xl_rotated,yl_rotated = rotate_airfoil(xl,yl,-alpha)
    x_rotated,yc_rotated = rotate_airfoil(x,yc,-alpha)
    ax1.plot(xu_rotated, yu_rotated, 'b')
    ax1.plot(xl_rotated, yl_rotated, 'b')
    ax1.plot(x_rotated, yc_rotated, 'r--')
    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")
    ax1.set_title(f"NACA {m}{p}{t:02d} Airfoil")
    ax1.axis('equal')
    ax1.grid(True)
    
    # 2. Cl vs Alpha
    ax2 = fig_orig.add_subplot(2, 3, 2)
    alphas = np.linspace(-5, 20, 100)
    Cls = [calculate_coefficients(a, m, p, t, Re, M)[0] for a in alphas]
    ax2.plot(alphas, Cls)
    ax2.set_xlabel("Angle of Attack (degrees)")
    ax2.set_ylabel("Lift Coefficient (Cl)")
    ax2.set_title("Lift Curve")
    ax2.grid(True)
    
    # 3. Drag Polar
    ax3 = fig_orig.add_subplot(2, 3, 3)
    Cds = [calculate_coefficients(a, m, p, t, Re, M)[1] for a in alphas]
    ax3.plot(Cds, Cls)
    ax3.set_xlabel("Drag Coefficient (Cd)")
    ax3.set_ylabel("Lift Coefficient (Cl)")
    ax3.set_title("Drag Polar")
    ax3.grid(True)
    
    # 4. L/D vs Alpha
    ax4 = fig_orig.add_subplot(2, 3, 4)
    LDs = [cl/cd for cl, cd in zip(Cls, Cds)]
    ax4.plot(alphas, LDs)
    ax4.set_xlabel("Angle of Attack (degrees)")
    ax4.set_ylabel("Lift-to-Drag Ratio (L/D)")
    ax4.set_title("L/D vs Angle of Attack")
    ax4.grid(True)
    
    # 5. Cm vs Alpha
    ax5 = fig_orig.add_subplot(2, 3, 5)
    Cms = [calculate_coefficients(a, m, p, t, Re, M)[2] for a in alphas]
    ax5.plot(alphas, Cms)
    ax5.set_xlabel("Angle of Attack (degrees)")
    ax5.set_ylabel("Moment Coefficient (Cm)")
    ax5.set_title("Moment Coefficient vs Angle of Attack")
    ax5.grid(True)
    
    # 6. Pressure Distribution (at design alpha)
    ax6 = fig_orig.add_subplot(2, 3, 6)
    Cp = 1 - (yu-yl)**2  # Simplified pressure distribution
    ax6.plot(x, -Cp)
    ax6.set_xlabel("x/c")
    ax6.set_ylabel("-Cp")
    ax6.set_title(f"Pressure Distribution at α = {alpha}°")
    ax6.grid(True)
    ax6.invert_yaxis()
    
    plt.tight_layout()
    st.pyplot(fig_orig)

if __name__ == "__main__":
    main()