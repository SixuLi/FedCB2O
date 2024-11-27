import numpy as np

def L(theta):
    return (np.sin(3*theta) + np.cos(theta-1)) / 2

def L_2d(theta):
    return np.sin(0.5*(theta[:,0]-0.1)**2) + np.cos(theta[:,1]-np.pi)

def G(theta):
    return np.linalg.norm(theta, axis=1)

class contrained_OPT_obj:
    def __init__(self, d):
        # self.global_min = np.array([1.76, 1.53, 1.3, 1.06, 0.83])
        # self.d = 5
        self.d = d
        self.global_min = np.array([0.4]*self.d)

    def sphere(self, theta):
        return (np.linalg.norm(theta, axis=1) - 1) ** 2

    def ellipse_2d(self, theta):
        return ((theta[:, 0] + 1)**2 / 2 + theta[:, 1]**2 - 1)**2

    def G(self, theta):
        return (20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.linalg.norm(theta - self.global_min, axis=1) ** 2 / self.d)) \
                - np.exp(np.sum(np.cos(2 * np.pi * (theta - self.global_min)), axis=1) / self.d))

    def simple_2d(self, theta):
        return np.linalg.norm(theta, axis=1) ** 2
