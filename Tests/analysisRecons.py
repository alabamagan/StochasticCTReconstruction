from main import PlotGaussianFit
import SimpleITK as sitk
import matplotlib.pyplot as plt

def main():
    imdict = {}
    imageKeys = ['128', '42', '64', '32']
    for keys in imageKeys:
        im = sitk.ReadImage("../TestData/Recon_%03d.nii.gz"%int(keys))
        imdict[keys] = sitk.GetArrayFromImage(im)

    for i in xrange(10, 30):
        reconImages = {}
        for keys in imageKeys:
            reconImages[keys] = imdict[keys][i]
        PlotGaussianFit(reconImages)


    pass

if __name__ == '__main__':
    main()