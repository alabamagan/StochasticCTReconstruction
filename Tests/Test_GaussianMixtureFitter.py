import Algorithm.GaussianMixtureFitter as GFitter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use("ggplot")

def main():
    # Generate random parameters
    numOfGaussians = np.random.randint(1, 5)
    params = [(np.random.rand() * 10,np.random.rand()*20 - 10, np.random.rand() * 5) for i in xrange(numOfGaussians)]
    print params

    x = np.linspace(-10, 10, 1000)
    y = np.sum(np.array([params[i][0] * GFitter.Gaussian(x, params[i][1], params[i][2])
                         for i in xrange(numOfGaussians)]), axis=0)

    initGuess = [[1,0,1] for i in xrange(numOfGaussians)]
    res = GFitter.Fitter1D(y, inDomain=x, initialGuess=initGuess, numOfGaussians=numOfGaussians)

    resy = np.sum(np.array([res[i][0] * GFitter.Gaussian(x, res[i][1], res[i][2])
                         for i in xrange(numOfGaussians)]), axis=0)



    plt.plot(x, y)
    plt.plot(x, resy)
    plt.show()
    pass


def TestFitter2():
    # Generate random parameters
    numOfGaussians = np.random.randint(2, 5)
    params = [(np.random.rand() * 10,np.random.rand()*20 - 10, np.random.rand() * 5) for i in xrange(numOfGaussians)]
    print params

    x = np.linspace(-10, 10, 1000)
    y = np.sum(np.array([params[i][0] * GFitter.Gaussian(x, params[i][1], params[i][2])
                         for i in xrange(numOfGaussians)]), axis=0)

    res = GFitter.SKLearnFitter(y, numOfGaussians=numOfGaussians)

    resy = np.sum(np.array([res[i][0] * GFitter.Gaussian(x, res[i][1], res[i][2])
                         for i in xrange(numOfGaussians)]), axis=0)
    print res

    plt.plot(x, y)
    plt.plot(x, resy)
    plt.show()


if __name__ == '__main__':
    TestFitter2()