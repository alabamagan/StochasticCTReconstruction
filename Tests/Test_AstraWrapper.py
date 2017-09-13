from Algorithm import AstraWrapper as aw
import numpy as np
import SimpleITK as sitk
import matplotlib as mpl
import matplotlib.pyplot as plt


def main(*args):
    im = sitk.GetArrayFromImage(sitk.ReadImage(args[0]))
    im[im == -3024] = 0
    pro = aw.Projector()
    pro.SetInputImage3D(im)
    projected = pro.Project(np.linspace(0, np.pi, 64), 'fanbeam')
    print projected.shape
    # plt.ion()
    # for i in xrange(projected.shape[0]):
    #     plt.cla()
    #     plt.imshow(projected[i])
    #     plt.draw()
    #     plt.pause(0.05)
    re = aw.Reconstructor()
    re.SetInputSinogram(projected, np.linspace(0, np.pi, 64), 'fanbeam')
    re.SetReconVolumeGeom(imageShape = im.shape, circle_mask=True)
    recon = re.Recon('SART_CUDA', 1000)

    fig = plt.figure(figsize=(13,13))
    ax1 = fig.add_subplot(212)
    ax2 = fig.add_subplot(211)
    for i in xrange(im.shape[0]):
        plt.ion()
        ax1.cla()
        ax2.cla()
        ax1.imshow(im[i], cmap="Greys_r")#, vmin=-1000, vmax=100)
        ax2.imshow(recon[i], cmap="Greys_r")#, vmin=-1000, vmax=100)
        plt.draw()
        plt.pause(0.2)

    pass

if __name__ == '__main__':
    imgfile ="../TestData/LCTSP.nii.gz"
    main(imgfile)


