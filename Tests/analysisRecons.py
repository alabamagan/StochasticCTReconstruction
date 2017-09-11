import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import Algorithm.GaussianMixtureFitter as GFitter
import numpy as np
import Algorithm.GaussianMixtureModel as MyGMM
import matplotlib.animation as animation
import multiprocessing
import matplotlib as mpl
mpl.style.use('ggplot')


def Animate(N, imdict, imageKeys):
    """
    Output plots of all slices as png

    :param N:
    :param imdict:
    :param imageKeys:
    :return:
    """
    reconImages = {}
    for keys in imageKeys:
        reconImages[keys] = imdict[keys]

    numOfImages = len(imageKeys)

    fig = plt.figure(N, figsize=(16, 13), dpi=80)
    print N
    ret = []
    resArray = []
    for i in xrange(len(reconImages)):
        ax1 = fig.add_subplot(numOfImages, 3, i * 3 + 1)
        ax2 = fig.add_subplot(numOfImages, 3, i * 3 + 2)
        ax1.imshow(reconImages[imageKeys[i]][N], cmap="Greys_r")
        ax1.set_title("Input %s" % imageKeys[i])
        hist = ax2.hist(reconImages[imageKeys[i]][N].flatten(), alpha=0.8, bins=200, normed=True)
        # ax2.set_title("Gaussian fitting for input %s"%imageKeys[i])

        # ============================================
        # Gaussian fitting
        # --------------------------------------------
        # Fit initial guess first, then fit 1D curve
        initGuess = GFitter.SKLearnFitter(reconImages[imageKeys[i]][N].flatten(), numOfGaussians=[2])
        initGuess = np.array(initGuess)
        numOfFittedGauss = len(initGuess)
        res = GFitter.Fitter1D(hist[0], hist[1][:-1], energy='distancesq', numOfGaussians=numOfFittedGauss,
                               initialGuess=initGuess, removeDistinct=True)
        resArray.append(res)
        res = res.Eval(hist[1][:-1])
        ax2.plot(hist[1][:-1], res)
        ax2.set_ylim([0, 0.010])

        ax2.set_xlim([-2000, 1000])
        ret.append(ax1)
        ret.append(ax2)

    changes = MyGMM.GMM.SortGMMs(resArray, True)

    bx1 = fig.add_subplot(numOfImages, 3, 1 * 3 + 3)
    bx2 = fig.add_subplot(numOfImages, 3, 2 * 3 + 3)
    bx3 = fig.add_subplot(numOfImages, 3, 3 * 3 + 3)
    [bx1.plot(changes['mean'][i]) for i in xrange(len(changes['mean']))]
    [bx2.plot(changes['sd'][i]) for i in xrange(len(changes['sd']))]
    [bx3.plot(changes['weight'][i]) for i in xrange(len(changes['weight']))]

    ret.append(bx1)
    ret.append(bx2)
    ret.append(bx3)

    fig.savefig("../TestData/gifs/Recon_%03d.png"%N)
    return

def main():
    global imdict, imageKeys
    imdict = {}
    imageKeys = ['128', '64', '42', '32']
    for keys in imageKeys:
        im = sitk.ReadImage("../TestData/Recon_SIRT_i1000_s000u180_%03d.nii.gz"%int(keys))

        imdict[keys] = sitk.GetArrayFromImage(im)

    processes = []
    pool = multiprocessing.Pool(processes=11)
    for i in xrange(imdict['32'].shape[0]):
        p = pool.apply_async(Animate, args=[i, imdict, imageKeys])
        processes.append(p)

    for p in processes:
        p.wait()
    os.system("convert -delay 1.6 ../TestData/gifs/*png ../TestData/gifs/Recon_SIRT_1000.gif; "
           "rm ../TestData/gifs/*png")
    pass

if __name__ == '__main__':
    main()