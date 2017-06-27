import numpy as np
import scipy.optimize as optimize
import scipy as sp
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt

def Gaussian(x, mu, sd):
    return 1./np.sqrt(2*np.pi*sd**2) * np.exp(-(x - mu)**2 / (2.*sd**2))


def RemoveDistinctValues(input, threshold):
    if (threshold < 1):
        raise ValueError("Threshold must be larger then 1.")

    for i in xrange(1, len(input) - 1):
        if (np.abs(input[i+1] - input[i]) > input[i+1] * threshold and
            np.abs(input[i] - input[i-1] > input[i-1] * threshold )):
            input[i] = (input[i+1] + input[i-1]) / 2.

    return input

def GaussianSmoothCurve(input, inDomain, radius):

    x = np.linspace(inDomain[0], inDomain[-1], input.shape[0])
    kernel = Gaussian(x, 0, radius)
    kernel /= np.sum(kernel)

    output = np.convolve(input, kernel, mode='same')
    return output


def Fitter1D(inCurve, inDomain=None, initialGuess=None, energy='distancesq', numOfGaussians=2, smoothing=False,
             removeDistinct = False):
    """
    Description:

      Fit the input with a guassian mixture, i.e:

         inCurve(x) = \sum_{i}^{numOfGaussians} c_i * N(mu_i, sd_i)

      This function use energy minimization for curve fitting, it is
      recommended to use SKLearnFitter for finding an initial guess.

    Details:

      Available energy functions:
      - Squared Distance ('distancesq')
          Energy = \sum_{x} [y(x) - inCurve(x)]^2
      - Absolute Distance ('absdistance')
          Energy = \sum_{x} |y(x) - inCurve(x)|
      - Smoothed-Squared Distance ('smoothdistsq')
          Energy = \sum_{x} StepFunc([y(x) - inCurve(x)]^2, SD)
          Where StepFunc(y, K) = 1 ; y < K
                StepFunc(y, K) = 1E5 ; otherwise
          Written to penalize sharp changes.

    Return:

        [(c_1, mu_1, sd_1), ... ,(c_n, mu_n, sd_n)]

      c_i  is the weight of each gaussian
      mu_i is the normal of each gaussian
      sd_i is the standard diviation f each gaussian
      n    is number of gaussians

    :param inCurve:         Input numpy array with uniform domain
    :param inDomain:        Evaluation domain of the input curve
    :param initialGuess:    Initial guess for the optimization solver
    :param energy:          String assigns which energy function to be used
    :param numOfGaussians:  Number of Gaussian in the output gaussian mixtures
    :return:                Formated parameter array
    """

    #=====================================================================
    # Error Check!
    #=====================================================================
    energyList = ['distancesq', 'absdistance', 'smoothdistsq']
    if (energyList.count(energy) == 0):
        raise ValueError("Currently available energy functions are %s"%energyList)
    if (inDomain == None):
        inDomain = np.linspace(0, inCurve.shape[0], inCurve.shaspe[0])
    if inDomain.shape != inCurve.shape:
        raise ArithmeticError("Input curve cannot have different shape as domain!")

    #=====================================================================
    # Optimization function
    #=====================================================================
    def CompareMixture(parameters):
        # Parameters format is [{c, mu, sd}, ...]
        parameters = parameters.reshape([numOfGaussians, 3])
        y = np.sum(np.array([parameters[i][0] * Gaussian(inDomain, parameters[i][1], parameters[i][2])
                         for i in xrange(numOfGaussians)]), axis=0)

        #--------------------------------------------------------------------
        # The energy can be choosen from the following options:
        # - Squared Distance
        #        Energy = \_sum_{x} [y(x) - inCurve(x)]^2
        # - Absolute Distance
        #        Energy = \_sum_{x} |y(x) - inCurve(x)|
        # - Smoothed-Squared Distance
        #        Energy = \sum_{x} StepFunc([y(x) - inCurve(x)]^2, SD)
        if (energy == 'distancesq'):
            E = np.sum(np.power(y - inCurve, 2.))
        elif (energy == 'absdistance'):
            E = np.sum(np.abs(y - inCurve))
        elif (energy == 'smoothdistsq'):
            diffs = np.power(y - inCurve, 2.)
            sd = np.sqrt(np.var(diffs))
            diffs[diffs > sd] *= 1E10
            E = np.sum(diffs)
        else:
            raise ValueError("Currently available energy functions are %s" % energyList)

        #----------------------------------------------------------
        # Debug informations
        # print ["%.2f x N(%.2f, %.2f)"%(parameters[i, 0], parameters[i,1], parameters[i,2])
        #        for i in xrange(numOfGaussians)], \
        #        "E: %.3f" % E
        return E


    #====================================================================
    # Preprocess
    #====================================================================
    #-----------------------------------
    # initial guess
    if (initialGuess == None):
        initialGuess = np.zeros([numOfGaussians, 3])
        initialGuess[:,2] = 1

    #-----------------------------------
    # Smoothing
    if (smoothing):
        inCurve = GaussianSmoothCurve(inCurve, inDomain, inDomain[2] - inDomain[0])

    #-----------------------------------
    # Remove distinct Value
    if (removeDistinct):
        inCurve = RemoveDistinctValues(inCurve, 1.3)

    #====================================================================
    # Start optimization
    #====================================================================
    res = optimize.minimize(CompareMixture, initialGuess, method='powell', tol=1e-8, options={'maxiter':1e3 ,'disp':True})
    return [(res.x[i*3+0], res.x[i*3+1], res.x[i*3+2]) for i in xrange(numOfGaussians)]


def SKLearnFitter(inData, numOfGaussians=[2]):
    """
    Descriptions:

      This function wraps the sklearn gaussian mixture model fitter. It is good as
      an initial guess finder for the Fitter1D function. The output is normed, and to
      use this one must also normalize the histogram of the inData.

      In some cases, this function fits better than the distance minimization. However,
      this method doesn't not account for very distinct values.

    Return:

        [(c_1, mu_1, sd_1), ... ,(c_n, mu_n, sd_n)]

      c_i  is the weight of each gaussian
      mu_i is the normal of each gaussian
      sd_i is the standard diviation f each gaussian
      n    is number of gaussians

    :param inData:          This should be a numpy data set, with each row a datapoint.
    :param numOfGaussians:  A sequence of integer deciding no of gaussian used.
    :return:                Return best fit parameters for normed histogram
    """

    #=====================================================================
    # Error Check!
    #====================================================================
    if (type(numOfGaussians) != list):
        numOfGaussians = [numOfGaussians]

    #-----------------------------------
    # Reshape data to fit sklearn input
    if (len(inData.shape) == 1):
        inData = inData.reshape(-1, 1)

    #=====================================================================
    # Run sklearn fitter
    #====================================================================
    model = [GMM(numOfGaussians[i]).fit(inData) for i in xrange(len(numOfGaussians))]

    #--------------------------------------------
    # Find most probable model
    AIC = [model[i].aic(inData) for i in xrange(len(numOfGaussians))]
    bestModel = model[np.argmin(AIC)]

    #--------------------------------------------
    # Pack parameters of the best model
    parameters = np.array(zip(bestModel.weights_, bestModel.means_, bestModel.covariances_))
    parameters = parameters.flatten()

    return [(parameters[i*3+0], parameters[i*3+1], np.sqrt(parameters[i*3+2])) for i in xrange(numOfGaussians[np.argmin(AIC)])]