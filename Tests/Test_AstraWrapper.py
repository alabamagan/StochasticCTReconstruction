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

    print projected.shape

    plt.imshow(projected[50], cmap="Greys")
    plt.show()

    pass

if __name__ == '__main__':
    imgfile ="../TestData/LCTSP.nii.gz"
    main(imgfile)


