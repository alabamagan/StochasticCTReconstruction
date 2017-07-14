import astra
import numpy as np

class Projector(object):
    def __init__(self):
        self._imageVol = 0
        self._proj_id = 0
        pass

    def __delete__(self, instance):
        astra.data3d.delete(self._proj_id)
        astra.data3d.delete(self._imageVol)

    def Project(self, thetas, algorithm):
        """

        :param thetas:
        :param algorithm:
        :return:
        """
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

        if (self._imageVol):
            astra.data3d.delete(self._imageVol)
        self._vol_geom = astra.create_vol_geom(self._ySize, self._xSize, self._zSize)
        self._imageVol = astra.data3d.create('-vol', self._vol_geom)
        astra.data3d.store(self._imageVol, self._image)

    def _Project(self, thetas, algorithm):
        if (not self._imageVol):
            raise BufferError("self._imageVol was not created")
            return

        proj_geom = proj_geom = astra.create_proj_geom(
            'parallel3d', 1., 1., self._zSize, int(np.ceil(np.sqrt(2*self._ySize**2))), thetas)

        if (self._proj_id):
            astra.data3d.delete(self._proj_id)
        self._proj_id, self._proj_data = astra.create_sino3d_gpu(self._imageVol, proj_geom, self._vol_geom)
        self._proj_array = astra.data3d.get(self._proj_id)
        pass

class Reconstructor(object):
    def __init__(self):
        self._rec_id = 0
        pass

    def __delete__(self, instance):
        pass


    def SetReconVolumeGeom(self, **kwargs):
        if (self._rec_id):
            astra.data3d.delete(self._rec_id)

        if (kwargs['vol_geom']):
            vol_geom = kwargs['vol_geom']
            self._rec_id = astra.data3d.create('-vol', vol_geom)
        elif (kwargs['imageShape']):
            imshape = kwargs['imageShape']
            xSize = imshape[2]
            ySize = imshape[1]
            zSize = imshape[0]
            vol_geom = astra.create_vol_geom(ySize, xSize, zSize)


    def SetInputSinogram(self, sino):
        pass