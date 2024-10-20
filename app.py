import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import utils
from GA import AirfoilGAOptimization

def optimize_airfoil(alpha, Re, M, initial_m, initial_p, initial_t):
    initial_params = [initial_m, initial_p, initial_t]
    
    bounds = [(0, 9), (1, 9), (6, 24)]
    
    result = minimize(
        utils.objective_function,
        initial_params,
        args=(alpha, Re, M),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 100}
    )
    
    return result.x

def main():
    st.title("Comprehensive NACA 4-Digit Airfoil Analysis and Optimization")
    
    # Common parameters
    st.header("Common Parameters")
    alpha = st.slider("Design Angle of Attack (degrees)", -5.0, 20.0, 5.0, 0.1)
    Re = st.slider("Reynolds Number", 1e5, 1e8, 1e6, 1e5)
    M = st.slider("Mach Number", 0.0, 0.9, 0.2, 0.05)
    
    # Gradient-based Optimization Parameters
    st.header("Gradient-based Optimization Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        m = st.slider("Initial Maximum Camber (% chord)", 0, 9, 2, 1)
    with col2:
        p = st.slider("Initial Position of Max Camber (tenths of chord)", 1, 9, 4, 1)
    with col3:
        t = st.slider("Initial Maximum Thickness (% chord)", 6, 24, 12, 1)
    
    # Genetic Algorithm Parameters
    st.header("Genetic Algorithm Parameters")
    col1, col2 = st.columns(2)
    with col1:
        mu = st.number_input("Population Size (mu)", min_value=1, max_value=1000, value=20, step=1)
        lambda_ = st.number_input("Number of Offspring (lambda)", min_value=1, max_value=1000, value=40, step=1)
        ngen = st.number_input("Number of Generations", min_value=1, max_value=1000, value=50, step=1)
    with col2:
        mutpb = st.number_input("Mutation Probability", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        elitism_ratio = st.number_input("Elitism Ratio", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        tournament_size = st.number_input("Tournament Size", min_value=1, max_value=int(mu), value=3, step=1)
    
    # Calculate coefficients for the initial airfoil
    Cl, Cd, Cm = utils.calculate_coefficients(alpha, m, p, t, Re, M)
    
    st.subheader("Current Airfoil Results (Gradient-based Optimization Initial Airfoil):")
    st.write(f"NACA {int(m)}{int(p)}{int(t):02d}")
    st.write(f"Lift Coefficient (Cl): {Cl:.4f}")
    st.write(f"Drag Coefficient (Cd): {Cd:.6f}")
    st.write(f"Moment Coefficient (Cm): {Cm:.4f}")
    st.write(f"Lift-to-Drag Ratio (L/D): {Cl/Cd:.2f}")
    
    # Original airfoil plots (before optimization)
    st.subheader("Original Airfoil Analysis")
    fig_orig = plt.figure(figsize=(20, 10))
    
    # Airfoil Shape
    ax1 = fig_orig.add_subplot(2, 3, 1)
    x = np.linspace(0, 1, 100)
    xu, yu, xl, yl, yc = utils.naca_4digit(m/100, p/10, t/100, x)
    xu_rotated, yu_rotated = utils.rotate_airfoil(xu, yu, -alpha)
    xl_rotated, yl_rotated = utils.rotate_airfoil(xl, yl, -alpha)
    x_rotated, yc_rotated = utils.rotate_airfoil(x, yc, -alpha)
    ax1.plot(xu_rotated, yu_rotated, 'b')
    ax1.plot(xl_rotated, yl_rotated, 'b')
    ax1.plot(x_rotated, yc_rotated, 'r--')
    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")
    ax1.set_title(f"NACA {int(m)}{int(p)}{int(t):02d} Airfoil")
    ax1.axis('equal')
    ax1.grid(True)
    
    # Cl vs Alpha
    ax2 = fig_orig.add_subplot(2, 3, 2)
    alphas = np.linspace(-5, 20, 100)
    Cls = [utils.calculate_coefficients(a, m, p, t, Re, M)[0] for a in alphas]
    ax2.plot(alphas, Cls)
    ax2.set_xlabel("Angle of Attack (degrees)")
    ax2.set_ylabel("Lift Coefficient (Cl)")
    ax2.set_title("Lift Curve")
    ax2.grid(True)
    
    # Drag Polar
    ax3 = fig_orig.add_subplot(2, 3, 3)
    Cds = [utils.calculate_coefficients(a, m, p, t, Re, M)[1] for a in alphas]
    ax3.plot(Cds, Cls)
    ax3.set_xlabel("Drag Coefficient (Cd)")
    ax3.set_ylabel("Lift Coefficient (Cl)")
    ax3.set_title("Drag Polar")
    ax3.grid(True)
    
    # L/D vs Alpha
    ax4 = fig_orig.add_subplot(2, 3, 4)
    LDs = [cl/cd for cl, cd in zip(Cls, Cds)]
    ax4.plot(alphas, LDs)
    ax4.set_xlabel("Angle of Attack (degrees)")
    ax4.set_ylabel("Lift-to-Drag Ratio (L/D)")
    ax4.set_title("L/D vs Angle of Attack")
    ax4.grid(True)
    
    # Cm vs Alpha
    ax5 = fig_orig.add_subplot(2, 3, 5)
    Cms = [utils.calculate_coefficients(a, m, p, t, Re, M)[2] for a in alphas]
    ax5.plot(alphas, Cms)
    ax5.set_xlabel("Angle of Attack (degrees)")
    ax5.set_ylabel("Moment Coefficient (Cm)")
    ax5.set_title("Moment Coefficient vs Angle of Attack")
    ax5.grid(True)
    
    # Pressure Distribution (at design alpha)
    ax6 = fig_orig.add_subplot(2, 3, 6)
    Cp = 1 - (yu - yl)**2  # Simplified pressure distribution
    ax6.plot(x, -Cp)
    ax6.set_xlabel("x/c")
    ax6.set_ylabel("-Cp")
    ax6.set_title(f"Pressure Distribution at α = {alpha}°")
    ax6.grid(True)
    ax6.invert_yaxis()
    
    plt.tight_layout()
    st.pyplot(fig_orig)
    
    # Optimization section
    if st.button("Optimize Airfoil"):
        with st.spinner("Optimizing airfoil... This may take a moment."):
            start_time = time.time()
            
            # Gradient-based Optimization
            gradient_start_time = time.time()
            opt_m, opt_p, opt_t = optimize_airfoil(alpha, Re, M, m, p, t)
            opt_Cl, opt_Cd, opt_Cm = utils.calculate_coefficients(alpha, opt_m, opt_p, opt_t, Re, M)
            gradient_end_time = time.time()
            gradient_duration = gradient_end_time - gradient_start_time

            # Genetic Algorithm Optimization
            ga_start_time = time.time()
            ga_optimizer = AirfoilGAOptimization(alpha, Re, M)
            best_individual = ga_optimizer.optimize(
                mu=int(mu),
                lambda_=int(lambda_),
                mutation_prob=float(mutpb),
                generations=int(ngen),
                elitism_ratio=float(elitism_ratio),
                tournament_size=int(tournament_size)
            )
            best_individual = ga_optimizer.optimize(
                lambda_=40,
                mu=20,
                generations=100,    # Number of generations
                crossover_prob=0.9, # Crossover probability
                mutation_prob=0.05, # Mutation probability
                elitism_ratio=0.1,  # Elitism ratio
                tournament_size=3   # Tournament selection size
            )
            ga_m, ga_p, ga_t = ga_optimizer.decode(best_individual)
            ga_Cl, ga_Cd, ga_Cm = utils.calculate_coefficients(alpha, ga_m, ga_p, ga_t, Re, M)
            ga_end_time = time.time()
            ga_duration = ga_end_time - ga_start_time

            end_time = time.time()
            total_duration = end_time - start_time
            
            st.success(f"Optimization completed in {total_duration:.2f} seconds!")
            
            # Show comparison table
            st.subheader("Optimization Results Comparison:")
            # Prepare data for the table
            data = {
                'Metric': ['NACA Code', 'Lift Coefficient (Cl)', 'Drag Coefficient (Cd)', 'Moment Coefficient (Cm)', 'Lift-to-Drag Ratio (L/D)', 'Time Taken (s)'],
                'Initial Airfoil': [f"NACA {int(m)}{int(p)}{int(t):02d}", f"{Cl:.4f}", f"{Cd:.6f}", f"{Cm:.4f}", f"{Cl/Cd:.2f}", "-"],
                'Gradient-based Optimization': [f"NACA {int(round(opt_m))}{int(round(opt_p))}{int(round(opt_t)):02d}", f"{opt_Cl:.4f}", f"{opt_Cd:.6f}", f"{opt_Cm:.4f}", f"{opt_Cl/opt_Cd:.2f}", f"{gradient_duration:.2f}"],
                'Genetic Algorithm Optimization': [f"NACA {int(round(ga_m))}{int(round(ga_p))}{int(round(ga_t)):02d}", f"{ga_Cl:.4f}", f"{ga_Cd:.6f}", f"{ga_Cm:.4f}", f"{ga_Cl/ga_Cd:.2f}", f"{ga_duration:.2f}"]
            }
            st.table(data)
            
            # Airfoil Shape Comparison
            st.subheader("Airfoil Shape Comparison")
            fig = plt.figure(figsize=(10, 6))
            x = np.linspace(0, 1, 100)
            
            # Original Airfoil
            xu, yu, xl, yl, yc = utils.naca_4digit(m/100, p/10, t/100, x)
            xu_rotated, yu_rotated = utils.rotate_airfoil(xu, yu, -alpha)
            xl_rotated, yl_rotated = utils.rotate_airfoil(xl, yl, -alpha)
            plt.plot(xu_rotated, yu_rotated, 'b', label=f'Initial NACA {int(m)}{int(p)}{int(t):02d}')
            plt.plot(xl_rotated, yl_rotated, 'b')
            
            # Gradient-based Optimized Airfoil
            xu_opt, yu_opt, xl_opt, yl_opt, yc_opt = utils.naca_4digit(opt_m/100, opt_p/10, opt_t/100, x)
            xu_opt_rotated, yu_opt_rotated = utils.rotate_airfoil(xu_opt, yu_opt, -alpha)
            xl_opt_rotated, yl_opt_rotated = utils.rotate_airfoil(xl_opt, yl_opt, -alpha)
            plt.plot(xu_opt_rotated, yu_opt_rotated, 'g', label=f'Gradient Opt NACA {int(round(opt_m))}{int(round(opt_p))}{int(round(opt_t)):02d}')
            plt.plot(xl_opt_rotated, yl_opt_rotated, 'g')
            
            # Genetic Algorithm Optimized Airfoil
            xu_ga, yu_ga, xl_ga, yl_ga, yc_ga = utils.naca_4digit(ga_m/100, ga_p/10, ga_t/100, x)
            xu_ga_rotated, yu_ga_rotated = utils.rotate_airfoil(xu_ga, yu_ga, -alpha)
            xl_ga_rotated, yl_ga_rotated = utils.rotate_airfoil(xl_ga, yl_ga, -alpha)
            plt.plot(xu_ga_rotated, yu_ga_rotated, 'r', label=f'GA Opt NACA {int(round(ga_m))}{int(round(ga_p))}{int(round(ga_t)):02d}')
            plt.plot(xl_ga_rotated, yl_ga_rotated, 'r')
            
            plt.xlabel("x/c")
            plt.ylabel("y/c")
            plt.title("Airfoil Shape Comparison")
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            st.pyplot(fig)
            
            # Comprehensive plotting
            st.subheader("Performance Curves Comparison")
            fig_perf = plt.figure(figsize=(20, 15))
            
            # 1. Cl vs Alpha
            ax1 = fig_perf.add_subplot(2, 3, 1)
            alphas = np.linspace(-5, 20, 100)
            Cls_init = [utils.calculate_coefficients(a, m, p, t, Re, M)[0] for a in alphas]
            Cls_opt = [utils.calculate_coefficients(a, opt_m, opt_p, opt_t, Re, M)[0] for a in alphas]
            Cls_ga = [utils.calculate_coefficients(a, ga_m, ga_p, ga_t, Re, M)[0] for a in alphas]
            ax1.plot(alphas, Cls_init, 'b', label=f'Initial NACA {int(m)}{int(p)}{int(t):02d}')
            ax1.plot(alphas, Cls_opt, 'g', label=f'Gradient Opt NACA {int(round(opt_m))}{int(round(opt_p))}{int(round(opt_t)):02d}')
            ax1.plot(alphas, Cls_ga, 'r', label=f'GA Opt NACA {int(round(ga_m))}{int(round(ga_p))}{int(round(ga_t)):02d}')
            ax1.set_xlabel("Angle of Attack (degrees)")
            ax1.set_ylabel("Lift Coefficient (Cl)")
            ax1.set_title("Lift Curve Comparison")
            ax1.legend()
            ax1.grid(True)
            
            # 2. Drag Polar
            ax2 = fig_perf.add_subplot(2, 3, 2)
            Cds_init = [utils.calculate_coefficients(a, m, p, t, Re, M)[1] for a in alphas]
            Cds_opt = [utils.calculate_coefficients(a, opt_m, opt_p, opt_t, Re, M)[1] for a in alphas]
            Cds_ga = [utils.calculate_coefficients(a, ga_m, ga_p, ga_t, Re, M)[1] for a in alphas]
            ax2.plot(Cds_init, Cls_init, 'b', label=f'Initial NACA {int(m)}{int(p)}{int(t):02d}')
            ax2.plot(Cds_opt, Cls_opt, 'g', label=f'Gradient Opt NACA {int(round(opt_m))}{int(round(opt_p))}{int(round(opt_t)):02d}')
            ax2.plot(Cds_ga, Cls_ga, 'r', label=f'GA Opt NACA {int(round(ga_m))}{int(round(ga_p))}{int(round(ga_t)):02d}')
            ax2.set_xlabel("Drag Coefficient (Cd)")
            ax2.set_ylabel("Lift Coefficient (Cl)")
            ax2.set_title("Drag Polar Comparison")
            ax2.legend()
            ax2.grid(True)
            
            # 3. L/D vs Alpha
            ax3 = fig_perf.add_subplot(2, 3, 3)
            LD_init = [cl/cd for cl, cd in zip(Cls_init, Cds_init)]
            LD_opt = [cl/cd for cl, cd in zip(Cls_opt, Cds_opt)]
            LD_ga = [cl/cd for cl, cd in zip(Cls_ga, Cds_ga)]
            ax3.plot(alphas, LD_init, 'b', label=f'Initial NACA {int(m)}{int(p)}{int(t):02d}')
            ax3.plot(alphas, LD_opt, 'g', label=f'Gradient Opt NACA {int(round(opt_m))}{int(round(opt_p))}{int(round(opt_t)):02d}')
            ax3.plot(alphas, LD_ga, 'r', label=f'GA Opt NACA {int(round(ga_m))}{int(round(ga_p))}{int(round(ga_t)):02d}')
            ax3.set_xlabel("Angle of Attack (degrees)")
            ax3.set_ylabel("Lift-to-Drag Ratio (L/D)")
            ax3.set_title("L/D vs Angle of Attack")
            ax3.legend()
            ax3.grid(True)
            
            # 4. Cm vs Alpha
            ax4 = fig_perf.add_subplot(2, 3, 4)
            Cms_init = [utils.calculate_coefficients(a, m, p, t, Re, M)[2] for a in alphas]
            Cms_opt = [utils.calculate_coefficients(a, opt_m, opt_p, opt_t, Re, M)[2] for a in alphas]
            Cms_ga = [utils.calculate_coefficients(a, ga_m, ga_p, ga_t, Re, M)[2] for a in alphas]
            ax4.plot(alphas, Cms_init, 'b', label=f'Initial NACA {int(m)}{int(p)}{int(t):02d}')
            ax4.plot(alphas, Cms_opt, 'g', label=f'Gradient Opt NACA {int(round(opt_m))}{int(round(opt_p))}{int(round(opt_t)):02d}')
            ax4.plot(alphas, Cms_ga, 'r', label=f'GA Opt NACA {int(round(ga_m))}{int(round(ga_p))}{int(round(ga_t)):02d}')
            ax4.set_xlabel("Angle of Attack (degrees)")
            ax4.set_ylabel("Moment Coefficient (Cm)")
            ax4.set_title("Moment Coefficient vs Angle of Attack")
            ax4.legend()
            ax4.grid(True)
            
            # 5. Pressure Distribution at design alpha
            ax5 = fig_perf.add_subplot(2, 3, 5)
            Cp_init = 1 - (yu - yl)**2
            Cp_opt = 1 - (yu_opt - yl_opt)**2
            Cp_ga = 1 - (yu_ga - yl_ga)**2
            ax5.plot(x, -Cp_init, 'b', label=f'Initial NACA {int(m)}{int(p)}{int(t):02d}')
            ax5.plot(x, -Cp_opt, 'g', label=f'Gradient Opt NACA {int(round(opt_m))}{int(round(opt_p))}{int(round(opt_t)):02d}')
            ax5.plot(x, -Cp_ga, 'r', label=f'GA Opt NACA {int(round(ga_m))}{int(round(ga_p))}{int(round(ga_t)):02d}')
            ax5.set_xlabel("x/c")
            ax5.set_ylabel("-Cp")
            ax5.set_title(f"Pressure Distribution at α = {alpha}°")
            ax5.legend()
            ax5.grid(True)
            ax5.invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig_perf)

if __name__ == "__main__":
    main()
