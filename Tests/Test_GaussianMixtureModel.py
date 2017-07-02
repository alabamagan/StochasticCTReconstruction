from Algorithm.GaussianMixtureModel import *
import numpy as np
import matplotlib.pyplot as plt

def TestGMM():
    x = np.linspace(-10, 10, 1000)

    G = GMM()
    G.AddGaussian(10, 5, 3)
    G.AddGaussian(2, -3, 1)
    G.AddGaussian(5, 0, 10)

    print G[2]
    pass


TestGMM()



