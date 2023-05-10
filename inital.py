import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit


def func(x, a, c, d):
    return a*np.exp(-c*x)+d

def power_law(x, a, b):
    return a * np.power(x, b)


def calculate_free_energy(phi_grid:np.ndarray, params:list) -> float:
    """Obtain the free energy in the system."""

    return np.sum(
                -params[0]/2 * np.power(phi_grid, 2) +
                params[0]/4 * np.power(phi_grid, 4) +
                params[1]/(2*params[3]**2) *
                (np.power(np.roll(phi_grid, 1, axis=0)-phi_grid, 2) +
                np.power(np.roll(phi_grid, 1, axis=1)-phi_grid, 2))
                )


def update_grid(phi_grid: np.ndarray, sigma, k, D, dx, dt, grid_size) -> np.ndarray:
    """Perform a single sweep, updating phi at each point."""


    xcoords, ycoords = np.indices([grid_size, grid_size])
    centre = [phi_grid.shape[0]/2, phi_grid.shape[1]/2]
    r = np.sqrt((xcoords - centre[0])**2 + (ycoords - centre[1])**2)
    rho = np.exp(-1 * r**2 / sigma**2)
    
    phi_grid += dt * (D * ((
            np.roll(phi_grid, 1, axis=1) +
            np.roll(phi_grid, -1, axis=1) +
            np.roll(phi_grid, 1, axis=0) +
            np.roll(phi_grid, -1, axis=0) -
            4 * phi_grid) / dx**2) +
            rho -
            k * phi_grid)


    # return the new updated phi grid.
    return phi_grid

def animation(phi_grid: np.ndarray, sigma, k, D, dx, dt, grid_size, vis: bool) -> None:
    """Run the animation of the cahn-hiliard equation."""

    # if we wanted a visualisation, make a figure for it.
    if vis:
        fig, ax = plt.subplots()
        im = ax.imshow(phi_grid, animated=True)
        cbar = fig.colorbar(im, ax=ax)

    # aver = []
    # t = []

    # choose a large range so that it will likely converge before then, but will never
    # continue forever.
    for i in range(10000):

        # move one step forward in the simulation, updating phi at every point.
        phi_grid = update_grid(phi_grid, sigma, k, D, dx, dt, grid_size)

        # every 50 sweeps update the animation.
        if i % 50 == 0 and vis:
            
            plt.cla()
            im = ax.imshow(phi_grid, interpolation='bilinear', animated=True)
            plt.draw()
            plt.pause(0.00001)

        if i % 50 == 0 and i > 2000:
            xcoords, ycoords = np.indices([grid_size, grid_size])
            centre = [phi_grid.shape[0]/2, phi_grid.shape[1]/2]
            r = np.sqrt((xcoords - centre[0])**2 + (ycoords - centre[1])**2)

            x = r.flatten()

            popt, pcov = curve_fit(func, x, phi_grid.flatten())
            plt.plot(x, func(x, *popt))
            popt, pcov = curve_fit(power_law, x, phi_grid.flatten(), p0=[40,-0.5])
            plt.plot(x, power_law(x, *popt))
            plt.plot(x, phi_grid.flatten())
            plt.show()
            break


        # if i % 50 == 0:
        #     aver.append(np.average(phi_grid))
        #     t.append(i)

        # every 10 sweeps record the free energy.
        # if i % 10 == 0:
        #     free_energy = calculate_free_energy(phi_grid, sigma, k, D, dx, dt, grid_size)

        #     with open("free_energy.dat", "a") as f:
        #         f.write(f"{i}, {free_energy}\n")

        # check if the simulation has converged every 100 sweeps.
        # if i % 100 == 0 and i > 100:
        #     data = np.genfromtxt('free_energy.dat', delimiter=',', skip_header=1)
        #     if len(set(np.round(data[-10:-1][:,1], 7))) == 1:
        #         # convergence
        #         break
    # plt.plot(t, aver)
    # plt.show()


def main():
    """Evaluate command line args to choose a function.
    """

    try:
        _, vis, dt = sys.argv
    except:
        print("Usage cahn_hiliard.py <vis> <dt>")
        sys.exit()

    grid_size = 50
    dx = 1
    dt = float(dt)

    # show a visualisation or not.
    if vis == "vis":
        vis = True
    elif vis == "novis":
        vis = False
    else:
        sys.exit()

    # choose the simulation parameters.
    # we set them to one as it is simpler and fully general.
    sigma = 10
    k = 0.01
    D = 1
    phi0 = 0.5

    # clear a file to write free energy data to.
    with open("free_energy.dat", "w") as f:
        f.write("iteration, free energy\n")

    # create custom phi_grid with base value of phi0 with noise between 0.1 and -0.1.
    phi_grid = float(phi0) + (np.random.rand(grid_size, grid_size) * 0.2) - 0.1

    animation(phi_grid, sigma, k, D, dx, dt, grid_size, vis)


if __name__=="__main__":
    main()