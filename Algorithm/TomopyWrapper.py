import tomopy
import numpy as np


def Project(image, theta):
    """
    Description
    -----------

      This function wraps tomopy's projection function to follow skimage
      conventions. Output sinogram theta is along the horizontal axis.

    Example
    -------

    >>> image = np.zeros([128,128])
    >>> meshgrid = np.meshgrid(arange(128), arange(128))
    >>> mask = (meshgrid[0] - 128/2)**2 + (meshgrid[1] - 128/2)**2 < 25**2
    >>> circleImage = image[mask] = 255
    >>> TomopyWrapper.Project(circleImage, np.arange(180))

    :param image: Input image, can be 2D or 3D
    :param theta: Sequence of angles in degrees
    :return:
    """


    #-----------------------------
    # Reshape image if it is 2D
    Flag2D = False
    if (len(image.shape) == 2):
        image = image.reshape([1, image.shape[0], image.shape[1]])
        Flag2D = True


    #-------------------------------------
    # Convert theta from degrees to radian
    theta = np.deg2rad(theta)

    sinogram = tomopy.project(image, theta, sinogram_order=True, emission=True, ncore=1)
    sinogram = sinogram.transpose(0, 2, 1)
    if Flag2D:
        sinogram = sinogram.reshape(sinogram.shape[2], sinogram.shape[1])
    return sinogram


def Recon(sinogram, theta, algorithm='gridrec'):
    """
    Description
    -----------

      This function wraps tomopy's recon function to follow skimage
      conventions. The input sinogram should have theta along horizontal
      axis.

    :param sinogram:  Input sinogram, theta along horizontal
    :param theta:     Sequence of projection angle in degrees
    :param algorithm: Algorithm used for reconstruction
    :return:
    """

    #=========================================
    # Error Check
    #=========================================
    if (len(sinogram.shape) == 2):
        if (len(theta) != sinogram.shape[1]):
            raise AssertionError("Input sinogram and theta have mismatch dimensions!")
    else:
        if (len(theta) != sinogram.shape[2]):
            raise AssertionError("Input sinogram and theta have mismatch dimensions!")

    #=========================================
    # Wrappings
    #-------------------------------------
    # Reshape image if it is 2D
    Flag2D = False
    if (len(sinogram.shape) == 2):
        sinogram = sinogram.reshape([1, image.shape[0], shape[1]])
        Flag2D = True
    sinogram = sinogram.transpose(0,2,1)


    #-------------------------------------
    # Convert theta from degrees to radian
    theta = np.deg2rad(theta)

    reconimage = tomopy.recon(sinogram, theta, algorithm=algorithm, sinogram_order=True, ncore=None)
    if Flag2D:
        reconimage = reconimage.reshape(sinogram.shape[2], sinogram.shape[1])
    return reconimage