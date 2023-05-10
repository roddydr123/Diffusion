import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def linear(x, a, b):
    return a*x + b


def free_energy():
    data = np.genfromtxt('0.5.free_energy.dat', delimiter=',', skip_header=1)
    plt.title("Free energy vs no. iterations for phi = 0.5")
    plt.plot(data[:,0], data[:,1])
    plt.xlabel("Number of iterations")
    plt.ylabel("Free energy")
    plt.show()


def scalar_field_plot():
    data = np.loadtxt("B_potential_R.dat", skiprows=1)
    field = data[:,1].reshape(100,100)
    im = plt.imshow(field)
    plt.title("Magnetic potential around a point charge, 100x100 grid")
    # plt.title("Electric scalar potential around a point charge, 100x100 grid")
    plt.colorbar(im)
    plt.show()


def radial():
    data = np.loadtxt("B_potential_R.dat", skiprows=1)
    x0 = data[:,0]
    y0 = data[:,1]
    mask = np.array([x0 != 0]).astype(int)
    mask2 = np.array([y0 != 0]).astype(int)
    mask3 = (mask * mask2).astype(bool)[0]
    x = np.log(x0[mask3])
    y = y0[mask3]
    plt.scatter(x, y, s=1)
    popt, pcov = curve_fit(linear, x[x < 2.7], y[x < 2.7])
    plt.plot(x, linear(x, *popt), color="k", label=f"y = {np.round(popt[0], 2)} x + {np.round(popt[1], 2)}")
    # plt.ylabel("Ln(Electric potential)")
    plt.ylabel("Magnetic potential")
    plt.xlabel("Ln(Radial distance from charge)")
    plt.title("Log magnetic potential around point charge")
    plt.legend()
    plt.show()


def vector():
    data = np.loadtxt("B_field.dat", skiprows=1)
    Fx = data[:,4].reshape(100,100).T
    Fy = data[:,5].reshape(100,100).T
    F = data[:,3].reshape(100,100)
    Fx[F == 0] = 0
    Fy[F == 0] = 0
    F[F == 0] = 1
    plt.quiver(Fx/F, Fy/F)
    plt.title("Magnetic field in 2D around a point charge, 100x100 grid")
    # plt.title("Electric field in 2D around a point charge, 100x100 grid")
    plt.show()


def omega_plot():
    data = np.loadtxt("omega.dat", skiprows=1)
    plt.scatter(data[:,0], data[:,1], s=1)
    plt.ylabel("No. iterations to converge")
    plt.xlabel("Relaxation coefficient $\omega$")
    plt.title("No. iterations to solve vs relaxation coefficient for point charge potential")
    plt.show()


# radial()
# scalar_field_plot()
# vector()
# omega_plot()
free_energy()