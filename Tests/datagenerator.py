import numpy as np
import SimpleITK as sitk
import skimage.transform as tr
import matplotlib.pyplot as plt
import Algorithm.AstraWrapper as awrapper
import os
import fnmatch
import multiprocessing as mp

def Reconstruction(input, outprefix, astraMethod='SIRT_CUDA', iteration=150):
    """
    Description
    -----------
      Process input with astra projection and reconstruction method, then write it with
      numpy format

    :param sitk.Image input:
    :param str outfname:
    :param str astraMethod:
    :param int iteration:
    :return:
    """

    spacing = input.GetSpacing()
    npIm = sitk.GetArrayFromImage(input)
    # npIm[npIm == npIm.min()] = 0

    thetas = np.linspace(0, 180, 180)

    D = awrapper.Projector()
    D.SetInputImage3D(npIm)
    D = D.Project(thetas, circle_mask=True)

    #============================
    # Debug
    #------------
    # plt.ion()
    # for i in xrange(D.shape[0]):
    #     plt.imshow(D[i])
    #     plt.draw()
    #     plt.pause(0.2)

    # plt.imshow(D)
    # plt.show()
    recons = [awrapper.Reconstructor() for i in xrange(3)]
    for i in xrange(3):
        recons[i].SetReconVolumeGeom(imageShape= npIm.shape, circle_mask=True)
        recons[i].SetInputSinogram(D[:,::i + 2], thetas=thetas[::i + 2])

    reconImages = {'64': recons[0].Recon(astraMethod, iteration),
                   '42': recons[1].Recon(astraMethod, iteration),
                   '32': recons[2].Recon(astraMethod, iteration)}

    np.save(outprefix + "_ori", npIm)
    for key in reconImages:
        outname = outprefix + "_%03d"%(int(key))
        np.save(outname, reconImages[key])
        print "Saving to ", outname

    # np.savez_compressed(outprefix, [reconImages[key] for key in reconImages])
    pass

def RecursiveSearchDCM(startdir):
    """
    Description
    -----------
      Search and returns all the folder directory which contains at least one .dcm file

    :param str startdir:
    :param int max_depth:
    :return:
    """
    assert os.path.isdir(startdir), "Directory doesn't exist!"

    matches = []
    for root, dirnames, filenames in os.walk(startdir):
        for filenames in fnmatch.filter(filenames, '*.dcm'):
            matches.append(root)

    matches = set(matches)
    return matches

def ProcessData(foldername, outputprefix):
    if not(os.path.isdir(outputprefix)):
        os.mkdir(outputprefix)

    print "Working on ", foldername
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(foldername))
    im = reader.Execute()
    Reconstruction(im, outputprefix)

def GenerateData(rootdir, outputdir):
    dirs = RecursiveSearchDCM(rootdir)

    # if not os.path.isdir(outputdir):
    #     os.mkdir(outputdir)

    processes = []
    pool = mp.Pool(processes=8)
    for dir in dirs:
        # print dir + "/Output/" + dir.split('/')[-1]
        p = pool.apply_async(ProcessData, args=[dir, outputdir + "/" + dir.split('/')[-1]])
        processes.append(p)

    for p in processes:
        p.wait()

GenerateData("/home/lwong/Storage/Data/CTReconstruction/LCTSC/", "/home/lwong/Storage/Data/CTReconstruction/LCTSC/Output")
# ProcessData("/home/lwong/Storage/Data/CTReconstruction/LCTSC/LCTSC-Test-S1-101/1.3.6.1.4.1.14519.5.2.1.7014.4598.492964872630309412859177308186/1.3.6.1.4.1.14519.5.2.1.7014.4598.106943890850011666503487579262/",
#             "/home/lwong/Storage/Data/CTReconstruction/LCTSC/Output/")