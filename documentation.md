# Airfoil Optimization Using Gradient-Based Methods and Evolutionary Strategies

## Abstract

This paper details the methodology used to design and optimize airfoils using a combination of analytical methods, gradient-based optimization, and evolutionary strategies. The goal is to maximize the lift-to-drag ratio of the airfoil, which is a critical parameter in aerodynamic efficiency. The results can be accessed [here](https://genetic-airfoil.streamlit.app/).

## 1. Introduction

Airfoil optimization is a key aspect in the design of aerodynamic surfaces, where maximizing the lift-to-drag ratio ($L/D$) is of utmost importance. This study involves using gradient-based methods to fine-tune airfoil parameters, followed by exploration with evolutionary strategies to further enhance the design.

## 2. Background Concepts

### 2.1 NACA Parameters

The NACA 4-digit series, such as NACA 2412, is a widely used airfoil parameterization that defines the airfoil shape using three main parameters:
- **$m$: Maximum camber (in percentage of chord)** - In NACA 2412, $m = 2\%$, indicating the maximum camber is 2% of the chord length.
- **$p$: Position of maximum camber (in tenths of chord length)** - In NACA 2412, $p = 40\%$, meaning the maximum camber is located at 40% of the chord length.
- **$t$: Maximum thickness (in percentage of the chord)** - In NACA 2412, $t = 12\%$, representing the maximum thickness of the airfoil is 12% of the chord length.

These parameters allow for the systematic generation of airfoil shapes for different aerodynamic applications.

### 2.2 Mach Number, Angle of Attack, and Reynolds Number

- **Mach Number ($M$)**: This is the ratio of the speed of the object to the speed of sound in the surrounding medium. It's crucial in determining compressibility effects in the flow. For subsonic flows ($M < 0.8$), compressibility corrections like the Prandtl-Glauert factor are used in lift coefficient calculations.
- **Angle of Attack (AOA, $\alpha$)**: The angle between the chord line of the airfoil and the oncoming airflow. It directly influences the lift produced by the airfoil and, if increased too much, can lead to flow separation and stall.
- **Reynolds Number ($Re$)**: A dimensionless quantity representing the ratio of inertial forces to viscous forces in the flow. It helps determine the nature of the flow (laminar or turbulent) and affects the calculation of the skin friction drag coefficient.

## 3. Mathematical Formulation

### 3.1 Airfoil Parameterization

We use the NACA 4-digit series to parameterize the airfoil, defined by three parameters:
- $m$: maximum camber (in percentage of the chord)
- $p$: position of maximum camber (in tenths of chord length)
- $t$: maximum thickness (in percentage of the chord)

The shape of the airfoil is defined using the following equations:

#### Camber Line Equation
The camber line $y_c(x)$ is calculated as:

$$
y_c(x) = 
\begin{cases} 
\frac{m}{p^2} \left( 2px - x^2 \right), & 0 \leq x \leq p \\
\frac{m}{(1-p)^2} \left( (1-2p) + 2px - x^2 \right), & p < x \leq 1 
\end{cases}
$$

#### Thickness Distribution Equation
The thickness distribution $y_t(x)$ is defined as:

$$
y_t = 5t \left( 0.2969\sqrt{x} - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1015x^4 \right)
$$

The upper and lower surfaces of the airfoil can then be computed as:

$$
x_u = x - y_t \sin(\theta), \quad y_u = y_c + y_t \cos(\theta)
$$
$$
x_l = x + y_t \sin(\theta), \quad y_l = y_c - y_t \cos(\theta)
$$

where $\theta$ is the slope of the camber line, defined as:

$$
\theta = \arctan \left( \frac{dy_c}{dx} \right)
$$

### 3.2 Coefficient Calculations

#### Lift Coefficient ($C_l$)
The lift coefficient is calculated using a compressibility-corrected approach:

$$
C_l = C_{l,\text{max}} \cdot \tanh\left( \frac{C_{l,\text{linear}}}{C_{l,\text{max}}} \right)
$$

where $C_{l,\text{linear}} = \frac{2 \pi (\alpha - \alpha_0)}{\beta}$, and $\beta = \sqrt{1 - M^2}$ is the Prandtl-Glauert factor.

#### Drag Coefficient ($C_d$)
The total drag coefficient is a sum of various contributions:

$$
C_d = C_f \cdot FF + C_{d_p} + C_{d_i} + C_{d_w}
$$

- **Skin friction drag coefficient $C_f$**: Estimated using correlations like Schiller-Naumann.
- **Form factor $FF$**: Accounts for drag due to the airfoil's shape.
- **Pressure drag coefficient $C_{d_p}$**: A function of the angle of attack and airfoil shape.
- **Induced drag coefficient $C_{d_i}$**: Depends on the lift coefficient and aspect ratio.
- **Wave drag $C_{d_w}$**: Significant near transonic and supersonic speeds (high Mach numbers).

### 3.3 Assumptions for the Models
- **Incompressible Flow**: Assumes Mach number $M < 0.3$ where compressibility effects are negligible.
- **Steady Flow**: Assumes that the airflow remains constant over time.
- **Attached Flow**: Assumes the flow does not separate significantly from the airfoil surface.

## 4. Gradient-Based Optimization

### 4.1 L-BFGS-B Algorithm

The **Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS-B)** method is employed for the initial optimization. This is a quasi-Newton method that approximates the Hessian matrix to determine the search direction. It is particularly efficient for problems with large dimensions and bounded constraints.

#### Assumptions:
- **Smoothness**: The objective function is assumed to be continuously differentiable.
- **Local Optimality**: Gradient-based methods may converge to local optima depending on the initial guess.
- **Bound Constraints**: We enforce bounds on the design variables $m$, $p$, and $t$ to ensure physically realistic airfoil shapes.

## 5. Evolutionary Strategies and Evolutionary Algorithms

To further explore the design space beyond local optima, we utilize evolutionary strategies (ES) and genetic algorithms (GA). These methods do not require derivative information and are well-suited for non-convex, multimodal problems.

### 5.1 Assumptions for Evolutionary Strategies and Genetic Algorithms
- **Population Size**: A sufficiently large population is required to ensure diverse genetic representation.
- **Generational Evolution**: Assumes that multiple iterations will gradually improve the design's fitness.
- **Stochastic Nature**: Convergence is not guaranteed; randomness plays a role in exploration.

## 6. Conclusion

The combined use of gradient-based optimization and evolutionary strategies has proven effective in airfoil design. While the L-BFGS-B method efficiently locates local optima, evolutionary techniques enhance the search, leading to potentially better solutions in complex design spaces.

## References

- Anderson, J. D. (2007). *Fundamentals of Aerodynamics*. McGraw-Hill.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
- Schilling, R. J., and Harris, S. L. (2000). *Applied Numerical Methods for Engineers Using MATLAB and C*. Brooks/Cole.