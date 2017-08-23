import numpy as np
import SimpleITK as sitk
import skimage.transform as tr
import Algorithm.AstraWrapper as awrapper
import os
import fnmatch
import multiprocessing as mp

def Reconstruction(input, outprefix, astraMethod='SIRT_CUDA', iteration=1500):
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

    thetas = np.linspace(0, 180, 180)

    D = awrapper.Projector()
    D.SetInputImage3D(npIm)
    D = D.Project(thetas)

    recons = [awrapper.Reconstructor() for i in xrange(3)]
    for i in xrange(3):
        recons[i].SetReconVolumeGeom(imageShape= npIm.shape, circle_mask=True)
        recons[i].SetInputSinogram(D[:,::i + 2], thetas=thetas[::i + 2])

    reconImages = {'64': recons[0].Recon(astraMethod, iteration),
                   '42': recons[1].Recon(astraMethod, iteration),
                   '32': recons[2].Recon(astraMethod, iteration)}

    for key in reconImages:
        outname = outprefix + "_%03d"%(int(key))
        np.savez_compressed(outname, reconImages[key])
        print "Saving to ", outname

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
    assert os.path.isdir(foldername), "Folder name incorrect!"

    print "Working on ", foldername
    fnames = fnmatch.filter(os.listdir(foldername), '*.dcm')
    fnames = [foldername + "/" + fname for fname in fnames]
    im = sitk.ReadImage(fnames)
    Reconstruction(im, outputprefix)

def GenerateData():
    dirs = RecursiveSearchDCM("/home/lwong/Storage/Data/CTReconstruction/DOI")

    if not os.path.isdir("/home/lwong/Storage/Data/CTReconstruction/DOI/Output"):
        os.mkdir("/home/lwong/Storage/Data/CTReconstruction/DOI/Output")

    processes = []
    pool = mp.Pool(processes=7)
    for dir in dirs:
        # print dir + "/Output/" + dir.split('/')[-1]
        p = pool.apply_async(ProcessData, args=[dir, "/home/lwong/Storage/Data/CTReconstruction/DOI/Output/" + dir.split('/')[-1]])
        processes.append(p)

    for p in processes:
        p.wait()

GenerateData()