import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt

# Define a function to generate NACA 4-digit airfoil coordinates
def naca_4digit(m, p, t, x):
    # Convert inputs from percentages to decimals
    m /= 100
    p /= 10
    t /= 100
    
    # Calculate camber line
    yc = np.where(x < p,
                  m * (2*p*x - x**2) / p**2,
                  m * ((1-2*p) + 2*p*x - x**2) / (1-p)**2)
    
    # Calculate gradient of camber line
    dyc_dx = np.where(x < p,
                      2*m * (p - x) / p**2,
                      2*m * (p - x) / (1 - p)**2)
    
    theta = np.arctan(dyc_dx)
    
    # Calculate thickness distribution
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    
    # Calculate upper and lower surface coordinates
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    return xu, yu, xl, yl


def plot_airfoil_comparison(initial_naca, sgd_naca, ga_naca):
    x = np.linspace(0, 1, 100)  # Chordwise positions
    
    # Generate coordinates for each NACA airfoil
    m, p, t = int(initial_naca[:1]), int(initial_naca[1:2]), int(initial_naca[2:])
    xu_initial, yu_initial, xl_initial, yl_initial = naca_4digit(m, p, t, x)
    
    m, p, t = int(sgd_naca[:1]), int(sgd_naca[1:2]), int(sgd_naca[2:])
    xu_sgd, yu_sgd, xl_sgd, yl_sgd = naca_4digit(m, p, t, x)
    
    m, p, t = int(ga_naca[:1]), int(ga_naca[1:2]), int(ga_naca[2:])
    xu_ga, yu_ga, xl_ga, yl_ga = naca_4digit(m, p, t, x)

    # Plot each airfoil shape
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xu_initial, yu_initial, 'b', label=f"Initial NACA {initial_naca}")
    ax.plot(xl_initial, yl_initial, 'b')
    
    ax.plot(xu_sgd, yu_sgd, 'g', label=f"SGD Opt NACA {sgd_naca}")
    ax.plot(xl_sgd, yl_sgd, 'g')
    
    ax.plot(xu_ga, yu_ga, 'r', label=f"GA Opt NACA {ga_naca}")
    ax.plot(xl_ga, yl_ga, 'r')
    
    # Plot settings
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")
    ax.set_title("Airfoil Shape Comparison")
    ax.legend(loc="upper right")
    ax.grid(True)
    ax.axis("equal")

    # Display the plot in Streamlit
    st.pyplot(fig)


def main():
    st.title("Airfoil Shape Comparison Tool")
    
    st.sidebar.header("Enter NACA Indices")
    initial_naca = st.sidebar.text_input("Initial Airfoil (e.g., '2412')", "2412")
    sgd_naca = st.sidebar.text_input("SGD Optimized Airfoil (e.g., '9406')", "9406")
    ga_naca = st.sidebar.text_input("GA Optimized Airfoil (e.g., '9306')", "9306")
    
    st.write("### Airfoil Shape Comparison")
    
    if st.button("Plot Airfoils"):
        plot_airfoil_comparison(initial_naca, sgd_naca, ga_naca)

if __name__ == "__main__":
    main()
