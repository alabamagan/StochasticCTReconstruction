from Algorithm import AstraWrapper as aw
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def main(*args):
    im = sitk.GetArrayFromImage(sitk.ReadImage(args[0]))
    im[im == -3024] = 0
    pro = aw.Projector()
    pro.SetInputImage3D(im)
    projected = pro.Project(np.linspace(0, np.pi, 100), 'parallel3d')

    re = aw.Reconstructor()
    re.SetInputSinogram(projected, np.linspace(0, np.pi, 100))
    re.SetReconVolumeGeom(imageShape = im.shape)
    recon = re.Recon('CGLS3D_CUDA', 150)

    fig = plt.figure()
    ax1 = fig.add_subplot(212)
    ax2 = fig.add_subplot(211)
    ax1.imshow(im[50], cmap="Greys_r")
    ax2.imshow(recon[50], cmap="Greys_r")
    plt.show()

    pass

if __name__ == '__main__':
    imgfile ="../TestData/LCTSP.nii.gz"
    main(imgfile)


