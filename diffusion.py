import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit


def func(x, a, c, d):
    return a*np.exp(-c*x)+d

def power_law(x, a, b):
    return a * np.power(x, b)


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

    return phi_grid


def update_grid_v(phi_grid, sigma, k, D, dx, dt, grid_size, v0):
    """Perform a single sweep, updating phi at each point."""

    # add rho.
    xcoords, ycoords = np.indices([grid_size, grid_size])
    centre = [grid_size/2, grid_size/2]
    r = np.sqrt((xcoords - centre[0])**2 + (ycoords - centre[1])**2)
    rho = np.exp(-1 * r**2 / sigma**2)

    # add drift velocity.
    vx = -v0 * np.sin(2 * np.pi * ycoords[0] / grid_size)
    vx = np.tile(vx, [grid_size, 1])

    d_phi_dx = (
        np.roll(phi_grid, 1, axis=0) -
        np.roll(phi_grid, -1, axis=0)) / (2 * dx)

    # perform the grid update.
    phi_grid += dt * (D * ((
            np.roll(phi_grid, 1, axis=1) +
            np.roll(phi_grid, -1, axis=1) +
            np.roll(phi_grid, 1, axis=0) +
            np.roll(phi_grid, -1, axis=0) -
            (4 * phi_grid)) / dx**2) +
            rho -
            (k * phi_grid) +    # one minus sign was messing things up so change this one.
            (d_phi_dx * vx))

    return phi_grid


def animation(phi_grid, sigma, k, D, dx, dt, grid_size):


    fig, ax = plt.subplots()
    im = ax.imshow(phi_grid, animated=True)
    cbar = fig.colorbar(im, ax=ax)

    # choose a large range so that it will likely converge before then, but will never
    # continue forever.
    for i in range(1000000):

        # move one step forward in the simulation, updating phi at every point.
        phi_grid = update_grid(phi_grid, sigma, k, D, dx, dt, grid_size)

        # every 50 sweeps update the animation.
        if i % 50 == 0:
            
            plt.cla()
            cbar.remove()
            im = ax.imshow(phi_grid, interpolation='bilinear', animated=True)
            cbar = fig.colorbar(im, ax=ax)
            plt.draw()
            plt.pause(0.00001)


def task3(phi_grid, sigma, k, D, dx, dt, grid_size):
    """Average value over time"""

    aver = []
    t = []

    # choose a large range so that it will likely converge before then, but will never
    # continue forever.
    for i in range(10000):

        # move one step forward in the simulation, updating phi at every point.
        phi_grid = update_grid(phi_grid, sigma, k, D, dx, dt, grid_size)

        if i % 50 == 0:
            aver.append(np.average(phi_grid))
            t.append(i)

    plt.plot(t, aver)
    plt.show()


def task4(phi_grid, sigma, k, D, dx, dt, grid_size):
    """Radial distribution"""

    # choose a large range so that it will likely converge before then, but will never
    # continue forever.
    for i in range(10000):

        # move one step forward in the simulation, updating phi at every point.
        phi_grid = update_grid(phi_grid, sigma, k, D, dx, dt, grid_size)

        if i % 50 == 0 and i > 2000:
            xcoords, ycoords = np.indices([grid_size, grid_size])
            centre = [phi_grid.shape[0]/2, phi_grid.shape[1]/2]
            r = np.sqrt((xcoords - centre[0])**2 + (ycoords - centre[1])**2)

            x = r.flatten()

            popt, pcov = curve_fit(func, x, phi_grid.flatten())
            plt.plot(x, func(x, *popt), label="exponential")
            popt, pcov = curve_fit(power_law, x, phi_grid.flatten(), p0=[40,-0.5])
            plt.plot(x, power_law(x, *popt), label="power law")
            plt.plot(x, phi_grid.flatten(), label="data")
            plt.legend()
            plt.ylabel("$\phi$")
            plt.xlabel("Radial distance from centre")
            plt.show()
            break


def task5(phi_grid, sigma, k, D, dx, dt, grid_size, v0):


    fig, ax = plt.subplots()
    im = ax.imshow(phi_grid, animated=True)
    cbar = fig.colorbar(im, ax=ax)

    # choose a large range so that it will likely converge before then, but will never
    # continue forever.
    for i in range(1000000):

        # move one step forward in the simulation, updating phi at every point.
        phi_grid = update_grid_v(phi_grid, sigma, k, D, dx, dt, grid_size, v0)

        # every 50 sweeps update the animation.
        if i % 50 == 0:
            
            plt.cla()
            cbar.remove()
            im = ax.imshow(phi_grid, interpolation='bilinear', animated=True)
            cbar = fig.colorbar(im, ax=ax)
            plt.draw()
            plt.pause(0.00001)


def main():
    """Evaluate command line args to choose a function.
    """

    mode = sys.argv[1]

    grid_size = 50
    dx = 1

    # choose the simulation parameters.
    sigma = 10
    k = 0.01
    D = 1
    phi0 = 0.5

    # create custom phi_grid with base value of phi0 with noise between 0.1 and -0.1.
    phi_grid = float(phi0) + (np.random.rand(grid_size, grid_size) * 0.2) - 0.1

    dt = float(sys.argv[2])
    if mode == "vis":
        animation(phi_grid, sigma, k, D, dx, dt, grid_size)
    elif mode == "3":
        task3(phi_grid, sigma, k, D, dx, dt, grid_size)
    elif mode == "4":
        task4(phi_grid, sigma, k, D, dx, dt, grid_size)
    elif mode == "5":
        v0 = float(sys.argv[3])
        task5(phi_grid, sigma, k, D, dx, dt, grid_size, v0)


if __name__=="__main__":
    main()