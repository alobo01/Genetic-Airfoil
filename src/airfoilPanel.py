import math
import os
import numpy as np
from matplotlib import path, pyplot as plt
from PANELS.COMPUTE_CIRCULATION import COMPUTE_CIRCULATION
from PANELS.COMPUTE_IJ_SPM import COMPUTE_IJ_SPM
from PANELS.STREAMLINE_SPM import STREAMLINE_SPM
from PANELS.XFOIL import XFOIL

class AirfoilPanelMethod:
    def __init__(self, Vinf=1, AoA=0, NACA='2412', airfoil_file="circle.dat", useNACA=False, ReynoldsNumber= 600000):
        """
        Initialize the airfoil panel method with the given parameters.

        Args:
            Vinf (float): Freestream velocity.
            AoA (float): Angle of attack in degrees.
            NACA (str): NACA airfoil series or airfoil name.
            airfoil_file (str): Filename for the airfoil (e.g., 'circle.dat').
            useNACA (bool): Whether to use the NACA airfoil (True) or load a custom airfoil (False).
        """
        self.Vinf = Vinf
        self.AoA = AoA
        self.NACA = NACA
        self.airfoil_file = airfoil_file
        self.useNACA = useNACA
        self.ReynoldsNumber = ReynoldsNumber
        self.setup()

    def setup(self):
        """Setup the airfoil, calculate coefficients, and solve for velocities."""
        # Convert angle of attack to radians
        self.AoAR = self.AoA * (np.pi / 180)
        
        # PPAR menu options for XFOIL
        PPAR = ['170', '4', '1', '1', '1 1', '1 1']
        
        # Get XFOIL results for the prescribed airfoil
        xFoilResults = XFOIL(self.NACA, PPAR, self.AoA, self.airfoil_file, useNACA=self.useNACA, Re=self.ReynoldsNumber)

        # Separate out XFOIL results and make them available as attributes
        self.afName = xFoilResults[0]        # Airfoil name
        self.xFoilX = xFoilResults[1]        # X-coordinate for Cp result
        self.xFoilY = xFoilResults[2]        # Y-coordinate for Cp result
        self.xFoilCP = xFoilResults[3]       # Pressure coefficient
        self.XB = xFoilResults[4]            # Boundary point X-coordinate
        self.YB = xFoilResults[5]            # Boundary point Y-coordinate
        self.xFoilCL = xFoilResults[6]       # Lift coefficient
        self.xFoilCD = xFoilResults[7]       # Drag coefficient
        self.xFoilCM = xFoilResults[8]       # Moment coefficient

        # Number of boundary points and panels
        self.numPts = len(self.XB)
        self.numPan = self.numPts - 1

        # Flip panels if needed (ensure clockwise order)
        edge = np.zeros(self.numPan)
        for i in range(self.numPan):
            edge[i] = (self.XB[i+1] - self.XB[i]) * (self.YB[i+1] + self.YB[i])
        if np.sum(edge) < 0:
            self.XB = np.flipud(self.XB)
            self.YB = np.flipud(self.YB)

        # Initialize geometric values
        self.XC = np.zeros(self.numPan)
        self.YC = np.zeros(self.numPan)
        self.S = np.zeros(self.numPan)
        self.phi = np.zeros(self.numPan)
        
        for i in range(self.numPan):
            dx = self.XB[i+1] - self.XB[i]
            dy = self.YB[i+1] - self.YB[i]
            self.XC[i] = 0.5 * (self.XB[i] + self.XB[i+1])
            self.YC[i] = 0.5 * (self.YB[i] + self.YB[i+1])
            self.S[i] = np.sqrt(dx**2 + dy**2)
            self.phi[i] = math.atan2(dy, dx)
            if self.phi[i] < 0:
                self.phi[i] += 2 * np.pi

        self.delta = self.phi + (np.pi / 2)
        self.beta = self.delta - self.AoAR
        self.beta[self.beta > 2 * np.pi] -= 2 * np.pi

        # Compute source panel strengths
        I, J = COMPUTE_IJ_SPM(self.XC, self.YC, self.XB, self.YB, self.phi, self.S)
        A = np.pi * np.eye(self.numPan) + I
        b = -self.Vinf * 2 * np.pi * np.cos(self.beta)
        self.lam = np.linalg.solve(A, b)

        # Compute velocity and pressure coefficients
        self.Vt = np.zeros(self.numPan)
        self.Cp = np.zeros(self.numPan)
        for i in range(self.numPan):
            addVal = np.sum((self.lam / (2 * np.pi)) * J[i, :])
            self.Vt[i] = self.Vinf * np.sin(self.beta[i]) + addVal
            self.Cp[i] = 1 - (self.Vt[i] / self.Vinf) ** 2

        # Compute forces
        CN = -self.Cp * self.S * np.sin(self.beta)
        CA = -self.Cp * self.S * np.cos(self.beta)
        self.CL = np.sum(CN * np.cos(self.AoAR)) - np.sum(CA * np.sin(self.AoAR))
        self.CD = np.sum(CN * np.sin(self.AoAR)) + np.sum(CA * np.cos(self.AoAR))
        self.CM = sum(self.Cp*(self.XC-0.25)*self.S*np.cos(self.phi))


    def print_comparison(self):
        """Print the comparison of results between our panel method (SPM) and XFOIL."""
        print("======= RESULTS =======")
        print("Lift Coefficient (CL)")
        print("\tSPM  : %2.8f" % self.CL)
        print("\tXFOIL: %2.8f" % self.xFoilCL)
        print("Drag Coefficient (CD)")
        print("\tSPM  : %2.8f" % self.CD)
        print("\tXFOIL: %2.8f" % self.xFoilCD)
        print("Moment Coefficient (CM)")
        print("\tSPM  : %2.8f" % self.CM)
        print("\tXFOIL: %2.8f" % self.xFoilCM)


    def setup_plots(self):
        """Setup the grid for plotting streamlines and pressure contours."""
        self.nGridX, self.nGridY = 100, 100
        self.xVals = [-0.5, 1]
        self.yVals = [-0.5, 0.5]
        self.Xgrid = np.linspace(self.xVals[0], self.xVals[1], self.nGridX)
        self.Ygrid = np.linspace(self.yVals[0], self.yVals[1], self.nGridY)
        self.XX, self.YY = np.meshgrid(self.Xgrid, self.Ygrid)

        # Initialize velocities and pressure coefficients for the grid
        self.Vx = np.zeros([self.nGridX, self.nGridY])
        self.Vy = np.zeros([self.nGridX, self.nGridY])
        AF = np.vstack((self.XB.T, self.YB.T)).T
        afPath = path.Path(AF)

        # Compute the velocities on the grid
        for m in range(self.nGridX):
            for n in range(self.nGridY):
                XP, YP = self.XX[m, n], self.YY[m, n]
                Mx, My = STREAMLINE_SPM(XP, YP, self.XB, self.YB, self.phi, self.S)
                if afPath.contains_points([(XP, YP)]):
                    self.Vx[m, n] = 0
                    self.Vy[m, n] = 0
                else:
                    self.Vx[m, n] = self.Vinf * np.cos(self.AoAR) + np.sum(self.lam * Mx / (2 * np.pi))
                    self.Vy[m, n] = self.Vinf * np.sin(self.AoAR) + np.sum(self.lam * My / (2 * np.pi))

        self.Vxy = np.sqrt(self.Vx**2 + self.Vy**2)
        self.CpXY = 1 - (self.Vxy / self.Vinf)**2

    def plot_streamlines(self, ax):
        """Plot streamlines on the provided axis."""
        slPct = 40
        Ysl = np.linspace(self.yVals[0], self.yVals[1], int((slPct / 100) * self.nGridY))
        Xsl = self.xVals[0] * np.ones(len(Ysl))
        XYsl = np.vstack((Xsl.T, Ysl.T)).T

        ax.streamplot(self.XX, self.YY, self.Vx, self.Vy, linewidth=0.5, density=10, color='r', arrowstyle='-', start_points=XYsl)
        ax.fill(self.XB, self.YB, 'k')
        ax.set_aspect('equal')
        ax.set_xlim(self.xVals)
        ax.set_ylim(self.yVals)
        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')

    def plot_pressure_contour(self, ax):
        """Plot pressure coefficient contour on the provided axis."""
        ax.contourf(self.XX, self.YY, self.CpXY, 500, cmap='jet')
        ax.fill(self.XB, self.YB, 'k')
        ax.set_aspect('equal')
        ax.set_xlim(self.xVals)
        ax.set_ylim(self.yVals)
        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')


if __name__=="__main__":
    datFilename = "circle.dat"
    #apm = AirfoilPanelMethod(airfoil_file=datFilename) 
    apm = AirfoilPanelMethod(NACA="4616",useNACA=True,AoA=10,ReynoldsNumber=600000)
    apm.print_comparison()
    objective = apm.xFoilCL
    if apm.xFoilCD>=0.01:
        objective /= apm.xFoilCD

    print(f"Objective: {objective}")
    apm.setup_plots()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    apm.plot_pressure_contour(ax1)
    apm.plot_streamlines(ax2)
    plt.tight_layout()

    # Display the plot
    plt.show(block=True)
