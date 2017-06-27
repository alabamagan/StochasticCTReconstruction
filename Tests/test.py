import matplotlib.pyplot as plt
import SimpleITK as sitk
import skimage.transform as tr
import numpy as np


def main():
    # Read image
    s = 128
    # im = sitk.ReadImage("./TestData/LCTSP.nii-subvolume-scale_1.nii.gz")
    im = sitk.ReadImage("./TestData/LCTSP.nii.gz")
    im = sitk.GetArrayFromImage(im)
    im[im == -3024] = 0
    # Generate projections
    reconlist = []
    increments = np.power(2, np.arange(1, 8)) + 1
    for N in increments:
        print "Doing ", N, "..."
        nSlice = im[s]
        thetas = np.linspace(0, 180, N)
        sinogram = tr.radon(nSlice, theta=thetas, circle=True)
        reconlist.append(tr.iradon(sinogram, theta=thetas, circle=True))

    # Plot
    for i in xrange(len(reconlist)):
        plt.imsave("./TestData/Recon_%03d_Projections.png"%i, reconlist[i], cmap="Greys_r")



    pass


if __name__ == '__main__':
    main()