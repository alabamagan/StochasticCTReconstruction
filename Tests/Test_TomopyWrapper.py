from SimpleITK import ReadImage, GetArrayFromImage
import numpy as np
import matplotlib.pyplot as plt
import Algorithm.TomopyWrapper as wrapper
import ctypes
import pyfftw

def main():
    """
    Seems that tomopy is broken on windows build, cannot import library

    :return:
    """

    print "running"
    try:
        # testload = ctypes.CDLL("E:\\Toolkits\\Anaconda2\\Lib\\site-packages\\tomopy\\libtomopy.pyd", handle=ctypes._dlopen("E:\\Toolkits\\Anaconda2\\Lib\\site-packages\\tomopy\\libtomopy.pyd"))
        handle = ctypes._dlopen("E:\\Toolkits\\Anaconda2\\Lib\\site-packages\\tomopy\\libtomopy.pyd")
        print handle
    except(OSError, IndexError):
        raise OSError("Fuck")

    im = GetArrayFromImage(ReadImage("../TestData/LCTSP.nii.gz"))



    sinogram = wrapper.Project(im, np.arange(180))
    recon = wrapper.TomopyWrapper.Recon(sinogram, np.arange(180))

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(sinogram, cmap="Greys_r")
    ax2.imshow(recon, cmap="Greys_r")
    plt.show()
    pass

main()