import numpy as np

class Gaussian(object):
    def __init__(self, weight, mu, sd):
        self.weight = float(weight)
        self.mean = float(mu)
        self.sd = float(sd)
        np.array

    def __str__(self):
        return "[Gaussian Object] - Weight: %.3f \t\tMean: %.3f \t\tSD: %.3f "%(self.weight, self.mean, self.sd)

    def __getitem__(self, item):
        return (self.weight, self.mean, self.sd)[item]


    @staticmethod
    def Gaussian(x, mu, sd):
        return 1. / np.sqrt(2 * np.pi * sd ** 2) * np.exp(-(x - mu) ** 2 / (2. * sd ** 2))


    def Eval(self, x):
        y = self.weight * Gaussian.Gaussian(x, self.mean, self.sd)
        return y

    def GetTuple(self):
        return (self.weight, self.mean, self.sd)

class GMM(object):
    """
    Class: GMM
    ----------

       This class is a Gassian Mixture Model class written for Fitter.


    Inputs
    ------
       Creation of a GMM object can be done in three ways:

       1. No arguments, creates a new GMM object without any components
           >>> g = GMM()
       2. List of tuples
           >>> g = GMM([(weight_1, mu_1, sd_1), ..., (weight_N, mu_N, sd_N)])
       3. Another GMM object
           >>> g1 = GMM([(1, 2, 3), (2, 3, 4)])
           >>> g2 = GMM(g1)

    :param args: (Explained in Inputs sessgion)
    """
    def __init__(self, *args):
        #--------------------------------------
        # Create GMM without any components
        if len(args) == 0:
            self._numOfGaussians=0
            self._components = []
        elif len(args) == 1:
            l = args[0]
            #----------------------------------
            # Copy content from another GMM
            if (type(l) == self.__class__):
                self._numOfGaussians = int(l._numOfGaussians)
                self._components = list(l._components)
                return

            #----------------------------------
            # Creats from a list of tuple
            if len(l[0]) != 3:
                raise IndexError("List input must be  N x 3")
            self._numOfGaussians = 0
            self._components = []
            for tup in l:
                self.AddGaussian(tup[0], tup[1], tup[2])
        else:
            raise ArithmeticError("GMM constructor takes one or no arguments!")



    def __getitem__(self, item):
        return self._components[item]

    def __str__(self):
        return "[GMM Object] - \n" + str.join("", ["\t%s\n"%G for G in self._components])


    def AddGaussian(self, weight, mu, sd):
        """
        Descriptions
        ------------
            Add a Gaussian component to this mixture

        :param float weight:
        :param float mu:
        :param float sd:
        :return:
        """

        self._numOfGaussians += 1
        self._components.append(Gaussian(weight, mu, sd))
        pass

    def GetMeans(self):
        """
        Descriptions
        ------------

            Return a list of unsorted means in the model

        :return:
        """
        return [self._components[i].mean for i in xrange(self._numOfGaussians)]

    def GetSDs(self):
        """
        Descriptions
        ------------

            Return a list of unsorted sd in the model

        :return:
        """
        return [self._components[i].sd for i in xrange(self._numOfGaussians)]

    def SwapGaussianComponents(self, i, j):
        """
        Descriptions
        ------------
            Swap the index position of i-th and j-th Gaussian position. Used to align
            a GMM with another when there are mismatch.

        :param int i: i-th component
        :param int j: j-th component
        :return:
        """
        i = int(i)
        j = int(j)

        if (i >= self._numOfGaussians or j >= self._numOfGaussians or
            i < 0 or j < 0):
            raise IndexError("One or both of the input indexes doesn't exist")

        temp = self._components[i]
        self._components[i] = self._components[j]
        self._components[j] = temp
        return 1

    def Eval(self, x):
        """
        Descriptions
        ------------
            Evaluate this Gaussian mixture at given x

        Detials
        -------
            y = \sum_{i}^{N} c_i * N(mu_i, sd_i)

        :param numpy.array x:  Linspace of domain to evaluate
        :return:                Array of values evaluated at each compoent of x
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

        :param GMM GMM1:    Gaussian mixture model 1
        :param GMM GMM2:    Gaussian mixture model 2
        :return:        Pairs inform of [[GMM1 index, GMM2 index, distance],...]
        """

        #=============================================================
        # Error check
        #=============================================================
        if (type(GMM1) != GMM or type(GMM2) != GMM):
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
            unpaired = range(s)
            for i in xrange(s):
                mu_1 = GMM1[i].mean

                pairIndex = -1
                dist = 1E10
                for j in unpaired:
                    mu_2 = GMM2[j].mean
                    d = np.abs(mu_1 - mu_2)
                    # Replace index if the distance is smaller
                    if d < dist:
                        dist = d
                        pairIndex= j

                # Pair only when an index is found
                if (pairIndex != -1):
                    gaussiansPair.append([i,pairIndex, dist])
                    paired.append(j)
                    unpaired.pop(unpaired.index(pairIndex))
        except(IndexError):
            raise IndexError("Gaussian mixtures has wrong number of parameters.")

        return gaussiansPair


    @staticmethod
    def SortGMMs(GMMList, groupParameters=False):
        """
        Descriptions
        ------------

          Sort the Gaussian components of the input GMM so that all of them
          aligns with the first element of the GMMList by closest mean

        :param list GMMList:         List of GMMs
        :param bool groupParameters: True to return the sorted dictionary of
                                     mean, weight and sd
        :return:
        """
        # raise NotImplemented
        #=======================================
        # Preprocessing
        #=======================================
        numOfGMMs = len(GMMList)

        #========================================
        # Identify GMM paris
        #---------------------------------------------------
        # Make sure the Gaussian components are paired up
        for j in xrange(1, numOfGMMs):
            pairs = GMM.GaussianComponenetMatching(GMMList[0],
                                                   GMMList[j])

            pairs = np.array(pairs)
            #========================================
            # Detects and swap unaligned pairs
            #========================================
            swapped = []
            for k in xrange(len(pairs)):
                pair = pairs[k]
                if pair[0] != pair[1] and not pair[1] in swapped:
                    GMMList[j].SwapGaussianComponents(pair[1], pair[0])

                    index = np.where(pairs[:,1] == pair[0])
                    pairs[int(index[0])][1] = pair[1]
                    pairs[k][1] = pair[0]


        if (groupParameters):
            outdict = {}
            outdict['weight'] = [[G[i].weight for G in GMMList] for i in xrange(GMMList[0]._numOfGaussians)]
            outdict['mean'] = [[G[i].mean for G in GMMList] for i in xrange(GMMList[0]._numOfGaussians)]
            outdict['sd'] = [[G[i].sd for G in GMMList] for i in xrange(GMMList[0]._numOfGaussians)]
            return outdict