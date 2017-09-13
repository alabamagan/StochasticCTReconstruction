import astra
import numpy as np
import SimpleITK as sitk

class Projector(object):
    def __init__(self):
        self._imageVol = 0
        self._proj_id = 0
        pass

    def __delete__(self, instance):
        astra.data3d.delete(self._proj_id)
        astra.data3d.delete(self._imageVol)

    def Project(self, thetas, algorithm = 'parallel3d', **kwargs):
        """
        Descriptions
        ------------
          Start the projection. Call only after input image is set
          Algorithm takes on the values:
            - 'parallel3d'


        :param thetas np.ndarray:
        :param algorithm str: Select from {'parallel3d'}
        :return:
        """
        #--------------------------------------------------------------------
        # Create Circular mask
        self._circle_mask = False
        if (kwargs.has_key('circle_mask')):
            if (kwargs['circle_mask']):
                self._circle_mask = True


        self._Project(thetas, algorithm)
        return self._proj_array

    def GetProjection(self):
        if (not self._proj_array):
            raise BufferError("Projection was never performed!")
        return self._proj_array

    def SetInputImage3D(self, im):
        """
        Descriptions
        ------------
          Set the input image to be projected.

        :param np.array im:
        :return:
        """
        self._image = im
        self._xSize = im.shape[2]
        self._ySize = im.shape[1]
        self._zSize = im.shape[0]
        print self._xSize, self._ySize, self._zSize

        if (self._imageVol):
            astra.data3d.delete(self._imageVol)
        self._vol_geom = astra.create_vol_geom(self._ySize, self._xSize, self._zSize)
        self._imageVol = astra.data3d.create('-vol', self._vol_geom)
        astra.data3d.store(self._imageVol, self._image)

    def _Project(self, thetas, algorithm):
        if (not self._imageVol):
            raise BufferError("self._imageVol was not created")
            return

        #--------------------------------------------------------------------
        # Create Circular mask
        if (self._circle_mask):
            center = np.array([self._zSize, self._ySize, self._xSize])/2.
            y, z, x = np.meshgrid(np.arange(self._ySize),
                                  np.arange(self._zSize),
                                  np.arange(self._xSize))
            mask = (y - center[1])**2 + \
                   (x - center[2])**2 < ((self._ySize+1)/2.)**2
            mask = mask == False
            self._image[mask] = 0
            astra.data3d.store(self._imageVol, self._image)

        if (algorithm in ['parallel', 'cone']):
            self._Project3D(thetas, algorithm)
        elif (algorithm == 'fanbeam'):
            self._Project2DTo3D(thetas, algorithm)

    def _Project2DTo3D(self, thetas, algorithm):
        if algorithm == 'fanbeam' >= 0:
            proj_geom = astra.create_proj_geom(
                'fanflat',
                1,
                int((self._xSize**2 + self._ySize**2)**0.5),
                thetas,
                (self._xSize**2 + self._ySize**2)**0.5 *2.,
                (self._xSize**2 + self._ySize**2)**0.5 /2.
                )
        else:
            print "Error! algorithm has incorrect input"

        subvol_geom = astra.create_vol_geom(self._xSize, self._ySize)
        self._proj_array = None
        for i in xrange(self._zSize):
            subproj_id = astra.create_projector('cuda', proj_geom, subvol_geom)
            self._proj_id, self._proj_data = astra.create_sino(self._image[i], subproj_id)
            s = self._proj_data.shape
            if (self._proj_array is None):
                self._proj_array = self._proj_data.reshape(1, s[0], s[1])
            else:
                self._proj_array = np.concatenate([self._proj_array, self._proj_data.reshape(1,s[0], s[1])], 0)
            astra.projector.delete(subproj_id)
            astra.data2d.delete(self._proj_id)


    def _Project3D(self, thetas, algorithm):
        if (algorithm.find('parallel') >= 0):
            proj_geom = astra.create_proj_geom(
                algorithm,
                1.,
                1.,
                self._zSize,
                int(np.ceil(np.sqrt(2*self._ySize**2))),
                thetas)
        elif algorithm.find('cone') >= 0:
            proj_geom = astra.create_proj_geom(
                'cone',
                1., # spacing x
                1., # spacing y
                self._zSize,
                int(np.ceil(np.sqrt(2*self._ySize**2))),
                thetas,
                int(np.ceil(np.sqrt(2*self._ySize**2))) *2., # dist_source_origin
                int(np.ceil(np.sqrt(2*self._ySize**2))) /2. # dist_origin_det
                )


        if (self._proj_id):
            astra.data3d.delete(self._proj_id)
        self._proj_id, self._proj_data = astra.create_sino3d_gpu(self._imageVol, proj_geom, self._vol_geom)
        self._proj_array = astra.data3d.get(self._proj_id)

        # Release GRAM
        astra.projector.delete(self._proj_id)
        astra.data3d.delete(self._imageVol)
        pass


class Reconstructor(object):
    def __init__(self):
        self._rec_id = -1
        self._vol_geom = -1
        self._sino_id = -1
        self._alg_id = -1
        self._mask_id = -1
        pass

    def __delete__(self, instance):
        if self._rec_id >= 0:
            astra.data3d.delete(self._rec_id)
        if self._sino_id >= 0:
            astra.data3d.delete(self._sino_id)
        if self._mask_id >= 0:
            astra.data3d.delete(self._mask_id)
        pass

    def SetReconVolumeGeom(self, **kwargs):
        """
        Description
        -----------
          Accepted kwargs = {'vol_geom', 'imageShape'}
          Use one of the above to specify the geometry of the outputimage

        :return:
        """
        if (self._rec_id >= 0):
            astra.data3d.delete(self._rec_id)

        if (kwargs.has_key('vol_geom')):
            vol_geom = kwargs['vol_geom']
            xSize = vol_geom['GridColCount']
            ySize = vol_geom['GridRowCount']
            zSize = vol_geom['GridSliceCount']
        elif (kwargs.has_key('imageShape')):
            imshape = kwargs['imageShape']
            xSize = imshape[2]
            ySize = imshape[1]
            zSize = imshape[0]
            vol_geom = astra.create_vol_geom(ySize, xSize, zSize)
        else:
            raise AssertionError("Argument must contain either 'vol_geom' or 'imageShape'" )

        #--------------------------------------------------------------------
        # Create Circular mask
        if (kwargs.has_key('circle_mask')):
            if (kwargs['circle_mask']):
                center = np.array([zSize, ySize, xSize])/2.
                meshgridY, meshgridZ, meshgridX = np.meshgrid(xrange(ySize),
                                                              xrange(zSize),
                                                              xrange(xSize))
                # for unknown reasons, meshgrid will transpose along axis 2
                mask = (meshgridY - center[1])**2 + \
                       (meshgridX - center[2])**2 < ((ySize+1)/2.)**2
                im = np.zeros([zSize, ySize, xSize])
                im[mask] = 255
                # im = sitk.GetImageFromArray(im)
                # sitk.WriteImage(im, "../TestData/mask.nii.gz")
                self._mask_id = astra.data3d.create('-vol', vol_geom, mask)
                self._mask = mask

        self._vol_geom = vol_geom
        self._rec_id = astra.data3d.create('-vol', vol_geom)

    def SetInputSinogram(self, sino, thetas, algorithm='parallel3d'):
        if(self._sino_id >= 0):
            astra.data3d.delete(self._sino_id)

        self._sinoInput = sino

        if (isinstance(sino, np.ndarray)):
            # Process into into an astra geom if it is ndarray
            imshape = sino.shape
            xSize = imshape[1]
            ySize = imshape[2]
            zSize = imshape[0]
            if (algorithm == 'parallel3d'):
                proj_geom = astra.create_proj_geom('parallel3d', 1., 1.,
                                                   zSize, ySize,
                                                   thetas)
                self._sino_id = astra.data3d.create('-proj3d', proj_geom)
                astra.data3d.store(self._sino_id, sino)
            elif (algorithm == 'cone'):
                proj_geom = astra.create_proj_geom('cone',
                                                   1.,
                                                   1.,
                                                   zSize,
                                                   ySize,
                                                   thetas,
                                                   int(np.ceil(np.sqrt(2*ySize**2))) *2.,
                                                   int(np.ceil(np.sqrt(2*ySize**2))) /2. # dist_origin_det
                                                   )
                self._sino_id = astra.data3d.create('-proj3d', proj_geom)
                astra.data3d.store(self._sino_id, sino)

            elif (algorithm == 'fanbeam'):
                # Pass variable to _Recon2Dto3D method
                self._proj_geom = None
                self._sino_id = sino
                self._thetas = thetas

        elif (type(sino) == int):
            self._sino_id = sino
        pass

    def Recon(self, algorithm, iterations):
        """
        Descriptions
        ------------
          Returns the reconstructed image.
          algorithm takes on values:
          - 'CGLS3D_CUDA'


        :param algorithm str:  Method of reconstruction
        :param iterations int: Number of iteration ran for;acd0;fir
        :return:
        """

        #==============================================================
        # Different handling for fan and parallel
        #-----------------------------------------------------------
        if isinstance(self._sino_id, int):
            output = self._Recon3D(iterations, algorithm)
        elif isinstance(self._sino_id, np.ndarray):
            output = self._Recon2DTo3D(iterations, algorithm)
        return output

    def _Recon2DTo3D(self, iterations, algorithm):
        """
        Descriptions
        ------------
          Reconstructino slice by slice
        :param int iterations:
        :return:
        """

        # Delete some geom which will not be used
        astra.data3d.delete(self._rec_id)
        astra.data3d.delete(self._mask_id)

        output = None
        for i in xrange(self._sinoInput.shape[0]):
            # Create mask
            if isinstance(self._mask, np.ndarray):
                mask_size = self._mask.shape

            proj_geom = astra.create_proj_geom('fanflat',
                                               1,
                                               # mask_size[0],
                                               int((mask_size[1]**2 + mask_size[2]**2)**0.5),
                                               self._thetas,
                                               (mask_size[1]**2 + mask_size[2]**2)**0.5 *2.,
                                               (mask_size[1]**2 + mask_size[2]**2)**0.5 /2.
                                               )
            vol_geom = astra.create_vol_geom(mask_size[1], mask_size[2])
            mask_id = astra.data2d.create('-vol', vol_geom, self._mask[i])
            rec_id = astra.data2d.create('-vol', vol_geom)
            sino_id = astra.data2d.create('-sino', proj_geom, self._sinoInput[i])

            cfg = astra.astra_dict(algorithm)
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sino_id
            if (mask_id >= 0):
                cfg['option'] = {}
                cfg['option']['ReconstructionMaskId'] = mask_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, iterations)
            if (output == None):
                output = astra.data2d.get(rec_id).reshape(1, mask_size[1], mask_size[2])
            else:
                output = np.concatenate([output, astra.data2d.get(rec_id).reshape(1, mask_size[1], mask_size[2])], 0)

            astra.data2d.delete(mask_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(alg_id)
            astra.data2d.delete(sino_id)

        return output

    def _Recon3D(self, iterations, algorithm):
        cfg = astra.astra_dict(algorithm)
        cfg['ReconstructionDataId'] = self._rec_id
        cfg['ProjectionDataId'] = self._sino_id
        if (self._mask_id >= 0):
            cfg['option'] = {}
            cfg['option']['ReconstructionMaskId'] = self._mask_id
        self._alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(self._alg_id, iterations)
        output = astra.data3d.get(self._rec_id)
        # Release GRAM
        astra.algorithm.delete(self._alg_id)
        astra.data3d.delete(self._sino_id)
        astra.data3d.delete(self._rec_id)
        if (self._mask_id >= 0):
            astra.data3d.delete(self._mask_id)
        return output