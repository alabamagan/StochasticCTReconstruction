"""
The pipeline of this algorithms is inspired the paper written by H. Li et al. [1], [2] in attempts to generalize their
method for the use of medical computer tomography (CT). The algorithm consist of th e



References:

  [1] Li, Hechao, et al. Accurate stochastic reconstruction of heterogeneous microstructures by limited x-ray
      tomographic projections. Journal of Microscopy 264.3 2016: 339-350.

  [2] Li, Hechao, Nikhilesh Chawla, and Yang Jiao. "Reconstruction of heterogeneous materials via stochastic
      optimization of limited-angle X-ray tomographic projections." Scripta Materialia 86 (2014): 48-51.

"""


import skimage.transform as tr
import numpy as np
import SimpleITK as sitk
import skimage.transform as tr


# Testing
import matplotlib.pyplot as plt
import Algorithm.GaussianMixtureFitter as GFitter


def SampleImage(prob, sd):
    return np.random.normal(prob, sd)


def main(inImage):
    N = 65

    #====================================================================
    # Preprocessing
    #--------------------------------------------------------------------
    # Obtain n projections from input (psuedo raw data, theta in degrees)
    # The projections are then normalized based on the sum of all pixels
    # in the sinogram
    thetas = np.linspace(0, 180, N)
    D = tr.radon(inImage, theta=thetas, circle=True)
    normD = D / abs(np.sum(D, axis=0))

    #--------------------------------------------------------------------
    # Filtered back projection based on N projectionsm, use ramp filter
    reconHalf = tr.iradon(D[:,::2],theta=thetas[::2], circle=True)
    reconFull = tr.iradon(D, theta=thetas, circle=True)
    recon = reconHalf*reconFull/np.mean(reconHalf)

    #-------------------------------------------
    # Plot the reconstruction images
    PlotGaussianFit(reconFull, reconHalf)
    return

    #===========================================
    # Simulated Annealling
    #-------------------------------------------
    # While Tempreture != 0, do while loop
    trial = None
    T = 100 #init temperature
    E = 1.e31 # float 32 max
    while(T > 10):
        #------------------------------------
        # Sample new image configuration
        # from the probability distribution
        trial = SampleImage(recon, 10)

        #------------------------------------
        # Calculate energy by equation (3)
        D_star = tr.radon(trial, theta=thetas)
        normD_star = D_star / abs(np.sum(D_star, axis=0))
        trial_energy = np.sum(np.abs(normD - normD_star))

        #-----------------------------------------------------------
        # Establish probability of excepting this configuration
        try:
            prob = np.min([1., np.exp(E/T)/np.exp(trial_energy/T)])
        except(OverflowError):
            print "Overflow error"
            prob = 1
        except(ValueError):
            print "Value Error"
            prob = 1

        #----------------------------------------------------
        # Base on above probability roll dice see if except
        roll = np.random.rand()
        ###   if except, renew energy, reduce temperature
        if (roll <= prob):
            dE = trial_energy - E
            E = trial_energy
            T -= 1
            print "Current E %.05f  dE %.05f excepted at prob: %.2f"%(E, dE, prob)


    #==============================================
    # Plot Results
    #==============================================
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(212)
    ax1.imshow(recon, cmap="Greys_r", vmax=0, vmin=-3000)
    ax2.imshow(trial, cmap="Greys_r", vmax=0, vmin=-3000)
    ax3.imshow(inImage, cmap="Greys_r", vmax=0, vmin=-3000)
    plt.show()
    pass


def PlotGaussianFit(reconFull, reconHalf):
    """
    Plot the histogram and the gaussian mixture model fitting result

    :param reconFull: Image1
    :param reconHalf: Image2, use half of the projections
    :return:
    """
    #============================================
    # Setting up the figure
    #---------------------------------------------
    # Displaying a total of four subplots, two
    # holds the image inputs two holds the
    # histogram generate from them
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    ax1.imshow(reconFull, cmap="Greys_r")
    ax2.imshow(reconHalf, cmap="Greys_r")

    histhalf = ax4.hist(reconHalf.flatten(), alpha=0.5, bins=200, normed=True)
    histfull = ax3.hist(reconFull.flatten(), alpha=0.5, bins=200, normed=True)

    ax1.set_title("Image ReconFull")
    ax2.set_title("Image ReconHalf")
    ax3.set_title("Hist ReconFull")
    ax4.set_title("Hist ReconHalf")

    #--------------------------
    # Maximize the plot window
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    # ============================================
    # Gaussian fitting
    #-------------------------------
    # Fit gaussian mixture for full
    initGuess = GFitter.SKLearnFitter(reconFull.flatten(), numOfGaussians=[2])  # initial fit
    initGuess = np.array(initGuess)
    resfull = GFitter.Fitter1D(histfull[0], histfull[1][:-1], energy='distancesq' ,
                           numOfGaussians=2, initialGuess=initGuess, removeDistinct=True)
    resfully = np.sum(np.array([resfull[i][0] * GFitter.Gaussian(histfull[1][:-1], resfull[i][1], resfull[i][2])
                            for i in xrange(2)]), axis=0)


    #-------------------------------------------
    # Fix Gaussian mixture for half
    initGuess = GFitter.SKLearnFitter(reconHalf.flatten(), numOfGaussians=[2])
    initGuess = np.array(initGuess)
    reshalf = GFitter.Fitter1D(histhalf[0], histhalf[1][:-1], energy='distancesq',
                           numOfGaussians=2, initialGuess=initGuess, removeDistinct=True)
    reshalfy = np.sum(np.array([reshalf[i][0] * GFitter.Gaussian(histhalf[1][:-1], reshalf[i][1], reshalf[i][2])
                            for i in xrange(2)]), axis=0)

    print reshalf
    print resfull

    #--------------------------------------------
    # Finish the remaining plot
    ax3.plot(histfull[1][:-1], resfully)
    ax4.plot(histhalf[1][:-1], reshalfy)
    plt.show()


if __name__ == '__main__':
    n = 100
    filename = "../TestData/LCTSP.nii.gz"
    input = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    input[input == -3024] = 0
    main(input[n])