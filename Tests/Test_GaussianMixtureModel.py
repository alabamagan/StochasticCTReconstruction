from Algorithm.GaussianMixtureModel import *
import numpy as np
import matplotlib.pyplot as plt

def TestGMM():
    x = np.linspace(-10, 10, 1000)

    G1 = GMM()
    G1.AddGaussian(10, 5, 3)
    G1.AddGaussian(2, -3, 1)
    G1.AddGaussian(5, 0, 10)
    G1.AddGaussian(10,2, 20)

    G2 = GMM()
    G2.AddGaussian(10, 2, 20)
    G2.AddGaussian(5, 0, 10)
    G2.AddGaussian(10, 5, 3)
    G2.AddGaussian(2, -3, 1)

    print GMM.GaussianComponenetMatching(G1, G2)


    pass


TestGMM()



