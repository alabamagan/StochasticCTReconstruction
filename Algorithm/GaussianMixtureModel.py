import numpy as np

class Gaussian(object):
    def __init__(self, weight, mu, sd):
        self.weight = weight
        self.mean = mu
        self.sd = sd
        np.array

    def __str__(self):
        return "[Gaussian Object] - Weight: %.3f \t\tMean: %.3f \t\tSD: %.3f "%(self.weight, self.mean, self.sd)

    @staticmethod
    def Gaussian(x, mu, sd):
        return 1. / np.sqrt(2 * np.pi * sd ** 2) * np.exp(-(x - mu) ** 2 / (2. * sd ** 2))


    def Eval(self, x):
        y = self.weight * Gaussian.Gaussian(x, self.mean, self.sd)
        return y

class GMM(object):
    def __init__(self):
        self._numOfGaussians=0
        self._components = []


    def __getitem__(self, item):
        return self._components[item]


    def AddGaussian(self, weight, mu, sd):
        """
        Descriptions
        ------------
            Add a Gaussian component to this mixture

        :param weight:
        :param mu:
        :param sd:
        :return:
        """

        self._numOfGaussians += 1
        self._components.append(Gaussian(weight, mu, sd))
        pass


    def Eval(self, x):
        """
        Descriptions
        ------------
            Evaluate this Gaussian mixture at given x

        Detials
        -------
            y = \sum_{i}^{N} c_i * N(mu_i, sd_i)

        :param x: Linspace of domain to evaluate
        :return:  Array of values evaluated at each compoent of x
        """

        y = np.sum(np.array([self._components[i].Eval(x) for i in xrange(self._numOfGaussians)]), axis=0)
        return y

    @staticmethod
    def GaussianComponenetMatching(GMM1, GMM2):
        """
        Descriptions
        ------------

          This function pair up Gaussian components of the input based on their distance.

        Example:
        --------

        >>> GMM1 = [(10, 1, 2), (20, -3, 4)]
        >>> GMM2 = [(2, 1.3, 4), (5, -2, 4)]
        >>> GMMPairs = GaussianComponenetMatching(GMM1, GMM2)
        >>> print GMMPairs # [[0,0,0.3], [1,1,1]]

        :param GMM1:    Gaussian mixture model 1
        :param GMM2:    Gaussian mixture model 2
        :return:        Pairs inform of [[GMM1 index, GMM2 index, distance],...]
        """

        #=============================================================
        # Error check
        #=============================================================
        if (type(GMM1) != GaussianMixtureModel or type(GMM2) != GaussianMixtureModel):
            raise TypeError("Input must be GaussianMextureModel object")

        if GMM1._numOfGaussians != GMM2._numOfGaussians:
            raise ArithmeticError("Two input has different size.")

        #=============================================================
        # Pair up gaussians
        #=============================================================
        gaussiansPair = []
        s = GMM1._numOfGaussians
        paired = []

        #----------------------------------------
        # Starts pairing the two GMMs
        try:
            for i in xrange(s):
                mu_1 = GMM1[i][1]

                pairIndex = -1
                dist = 1E10
                for j in unpaired:
                    mu_2 = GMM2[j][1]
                    d = np.abs(mu_1 - mu2)

                    # Replace index if the distance is smaller
                    if d < dist and paired.count(j) == 1:
                        dist = d
                        pairIndex= j

                # Pair only when an index is found
                if (pairIndex != -1):
                    gaussiansPair.append([i,j, pairIndex])
                    paired.append(j)
        except(IndexError):
            raise IndexError("Gaussian mixtures has wrong number of parameters.")

        return paired