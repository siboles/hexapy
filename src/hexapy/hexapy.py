import numpy as np
from scipy.spatial import cKDTree
import itertools

class Mesher(object):
    def __init__(self):
        r"""
        This class provides functionality for creating hexahedral meshes of primitives.


        Attributes
        ----------
        meshes : [, dict, ...]
            List of meshes stored as dictionaries. Dictionary fields are as follows:
            * 'Name' -- str, optional
               Name assigned to mesh - defaults to "Part_i" as appended
            * 'Nodes' -- ndarray((N, 3), dtype=float)
               Array of node ids and coordinates
            * 'Node IDs' -- ndarray(N, dtype=int)
            * 'Elements' -- ndarray((N, 8), dtype=(int, ...) )
               Array of element ids [:,0] and nodal connectivity [:,1:]
            * 'Element IDs' -- ndarray(N, dtype=int)
            * 'Element Sets' -- dict('name', ndarray((N,1), dtype=int))
               Each dictionary key is the set name and the value is an
               array of element ids in the set. Default key:
                 "all": all elements in the mesh
            * 'Node Sets' -- dict('name', ndarray((N,1), dtype=int))
               Each dictionary key is the set name and the value is an
               array of node ids in the set. Default key:
                 "all": all nodes in the mesh
            * 'Surface Sets' -- dict('name', ndarray((N,4)))
              Each dictionary key is the set name and the value is an
              array of faces (rows) indicated by nodal connectivity
        _coordinates : [, ndarray(N, 3), ...]
            List of x, y, and z coordinate ranges in undeformed block for mesh in same index.
        """
        self.meshes = []
        self._coordinates = []

    def _makeBlock(self,
                  x_origin,
                  y_origin,
                  z_origin,
                  x_length,
                  y_length,
                  z_length,
                  x_div,
                  y_div,
                  z_div):
        block = {}
        # start numbering nodes from last node id in last appended mesh + 1
        if len(self.meshes) > 0:
            node_start = self.meshes[-1]["NodeIDs"][-1] + 1
        else:
            node_start = 1
        halfx = x_length / 2.0
        halfy = y_length / 2.0
        halfz = z_length / 2.0

        x = np.linspace(x_origin, x_origin + x_length, x_div)
        y = np.linspace(y_origin, y_origin + y_length, y_div)
        z = np.linspace(z_origin, z_origin + z_length, z_div)

        mesh = np.meshgrid(x, y, z, indexing='xy')
        block["nodes"] = np.hstack((mesh[0].reshape((-1, 1)),
                                    mesh[1].reshape((-1, 1)),
                                    mesh[2].reshape((-1, 1))))

        block["node_ids"] = np.arange(node_start, block["nodes"].shape[0] + node_start)

        nelems = (x_div - 1) * (y_div - 1) * (z_div - 1)
        if len(self.meshes) > 0:
            element_start = self.meshes[-1]["ElementIDs"][-1] + 1
        else:
            element_start = 1
        block["element_ids"] = np.arange(element_start, nelems + element_start)
        block["elements"] = np.zeros((nelems, 8), dtype=int)

        block["elements"][0,0:4] = [node_start, z_div+node_start, x_div*z_div+z_div+node_start, x_div*z_div+node_start]
        block["elements"][0,4:] = block["elements"][0,0:4]+1

        for i in xrange(x_div-2):
            block["elements"][i+1,:] = block["elements"][i,:]+z_div
        step = (x_div-1)
        for i in xrange(z_div-2):
            block["elements"][(i+1)*step:(i+2)*step,:] = block["elements"][i*step:(i+1)*step,:]+1

        step = (x_div-1) * (z_div-1)
        increment = x_div*z_div
        for i in xrange(y_div-2):
            block["elements"][(i+1)*step:(i+2)*step,:] = block["elements"][i*step:(i+1)*step,:]+increment

        self._coordinates.append(np.hstack((x, y, z)))
        return block


    def makeBox(self, name=None,
                x_origin=None,
                y_origin=None,
                z_origin=None,
                x_length=1.0,
                y_length=1.0,
                z_length=1.0,
                x_div=10,
                y_div=10,
                z_div=10,
                x_edge=-1.0,
                y_edge=-1.0,
                z_edge=-1.0):
        r"""
        Create a box.

        Parameters
        ----------
        name : str, optional
            Name identifier for mesh. Defaults to "Part_{:s}".format(index in meshes)
        x_length : float=1.0, optional
             The box x-length
        y_length : float=1.0, optional
             The box y-length
        z_length : float=1.0, optional
             The box z-length
        x_div : int=10, optional
             Number of element divisions in the x-direction
        y_div : int=10, optional
             Number of element divisions in the y-direction
        z_div : int=10, optional
             Number of element divisions in the z-direction
        x_edge : float=-1.0, optional
             Overrides *x_div* with absolute edge length (ignored if non-positive)
        y_edge : float=-1.0, optional
             Overrides *y_div* with absolute edge length (ignored if non-positive)
        z_edge : float=-1.0, optional
             Overrides *z_div* with absolute edge length (ignored if non-positive)

        Returns
        -------
        Appends a box mesh definition to *meshes*.
        """
        if name is None:
            name = "Part_{:d}".format(len(self.meshes) + 1)
        if x_edge > 0:
            if x_edge > x_length:
                raise SystemExit("x_edge must be less than x_length")
            x_div = np.ceil(x_length / x_edge)
        if y_edge > 0:
            if y_edge > y_length:
                raise SystemExit("y_edge must be less than y_length")
            y_div = np.ceil(y_length / y_edge)
        if z_edge > 0:
            if z_edge > z_length:
                raise SystemExit("z_edge must be less than z_length")
            y_div = np.ceil(z_length / z_edge)
        if x_origin is None:
            x_origin = -x_length / 2.0
        if y_origin is None:
            y_origin = -y_length / 2.0
        if z_origin is None:
            z_origin = -z_length / 2.0

        blockList = []

        block = self._makeBlock(x_origin,
                                y_origin,
                                z_origin,
                                x_length,
                                y_length,
                                z_length,
                                x_div,
                                y_div,
                                z_div)

        # node sets
        n_xn = np.argwhere(block["nodes"][:,0] < x_origin + 1e-7).ravel() + block["node_ids"][0]
        n_xp = np.argwhere(block["nodes"][:,0] > x_origin + x_length - 1e-7).ravel() + block["node_ids"][0]
        n_yn = np.argwhere(block["nodes"][:,1] < y_origin + 1e-7).ravel() + block["node_ids"][0]
        n_yp = np.argwhere(block["nodes"][:,1] > y_origin + y_length - 1e-7).ravel() + block["node_ids"][0]
        n_zn = np.argwhere(block["nodes"][:,2] < z_origin + 1e-7).ravel() + block["node_ids"][0]
        n_zp = np.argwhere(block["nodes"][:,2] > z_origin + z_length - 1e-7).ravel() + block["node_ids"][0]

        # face sets

        mesh_dict = {"Name": name,
                     "Nodes": block["nodes"],
                     "NodeIDs": block["node_ids"],
                     "Elements": block["elements"],
                     "ElementIDs": block["element_ids"],
                     "NodeSets": {"n_xn": n_xn,
                                  "n_xp": n_xp,
                                  "n_yn": n_yn,
                                  "n_yp": n_yp,
                                  "n_zn": n_zn,
                                  "n_zp": n_zp},
                     "ElementSets": {"e_"+name: block["element_ids"]},
                     "FaceSets": {}}

        self.meshes.append(mesh_dict)

    def makeCylinder(self, name=None,
                     r_major=1.0,
                     r_minor=1.0,
                     z_length=1.0,
                     r_major_div=10,
                     r_minor_div=10,
                     z_div=10,
                     r_major_edge=-1.0,
                     r_minor_edge=-1.0,
                     z_edge=-1.0
    ):
        r"""
        Create an elliptic cylinder.

        Parameters
        ----------
        name : str, optional
            Name identifier for mesh. Defaults to "Part_{:s}".format(index in meshes)
        r_major : float 1.0, optional
             Major radius of the elliptical z-slices
        r_minor : float 1.0, optional
             Minor radius of the elliptical z-slices
        z_length : float 1.0, optional
             Cylinder height.
        r_major_div : int=10, optional
             Number of element divisions in the major radius direction
        r_minor_div : int=10, optional
             Number of element divisions in the minor radius direction
        z_div : int=10, optional
             Number of element divisions in the z-direction
        r_major_edge : float=-1.0, optional
             Overrides *r_major_div* with absolute edge length (ignored if non-positive)
        r_minor_edge : float=-1.0, optional
             Overrides *r_minor_div* with absolute edge length (ignored if non-positive)
        z_edge : float=-1.0, optional
             Overrides *z_div* with absolute edge length (ignored if non-positive)

        Returns
        -------
        Appends an elliptic cylinder mesh definition to *meshes*.
        """
        if name is None:
            name = "Part_{:d}".format(len(self.meshes) + 1)

        xmin = r_major * np.cos(3*np.pi/4.0)
        xmax = r_major * np.cos(np.pi/4.0)
        ymin = r_minor * np.sin(3*np.pi/4.0)
        ymax = r_minor * np.sin(np.pi/4.0)
        xleft = xmin + ymin - r_minor / 2.0
        xright = r_minor / 2.0 - ymax + xmax
        inner_divx = int(np.ceil((xright - xleft) / r_major * r_major_div))
        outer_divx = np.ceil(2.0*r_major_div - inner_divx)
        outer_divy = np.ceil(r_minor / 2.0 * r_minor_div)
        outer_div = int(np.ceil((outer_divx + outer_divy) / 2.0))
        blockList = []
        # middle block
        blockList.append(self._makeBlock(x_origin=xleft,
                                         y_origin=-r_minor / 2.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=xright - xleft,
                                         y_length=r_minor,
                                         z_length=z_length,
                                         x_div=inner_divx,
                                         y_div=r_minor_div,
                                         z_div=z_div))
        # left block
        blockList.append(self._makeBlock(x_origin=-r_major,
                                         y_origin=-r_minor / 2.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=r_major + xleft,
                                         y_length=r_minor,
                                         z_length=z_length,
                                         x_div=outer_div,
                                         y_div=r_minor_div,
                                         z_div=z_div))
        # right block
        blockList.append(self._makeBlock(x_origin=xright,
                                         y_origin=-r_minor / 2.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=r_major - xright,
                                         y_length=r_minor,
                                         z_length=z_length,
                                         x_div=outer_div,
                                         y_div=r_minor_div,
                                         z_div=z_div))
        # top block
        blockList.append(self._makeBlock(x_origin=xleft,
                                         y_origin=r_minor / 2.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=xright - xleft,
                                         y_length=r_minor / 2.0,
                                         z_length=z_length,
                                         x_div=inner_divx,
                                         y_div=outer_div,
                                         z_div=z_div))
        # bottom block
        blockList.append(self._makeBlock(x_origin=xleft,
                                         y_origin=-r_minor,
                                         z_origin=-z_length / 2.0,
                                         x_length=xright - xleft,
                                         y_length=r_minor / 2.0,
                                         z_length=z_length,
                                         x_div=inner_divx,
                                         y_div=outer_div,
                                         z_div=z_div))

        block_positions = ('left', 'right', 'top', 'bottom')
        block_angles = ([5*np.pi / 4.0, 3*np.pi / 4.0],
                        [-np.pi / 4.0, np.pi / 4.0],
                        [3*np.pi / 4.0, np.pi / 4.0],
                        [5*np.pi / 4.0, 7*np.pi / 4.0])

        for i, b in enumerate(blockList[1:]):
            blockList[i+1] = self._interpCylindrical(b, r_major, r_minor, block_angles[i], block_positions[i], 'none')

        block = self._mergeBlocks(blockList)
        mesh_dict = {"Name": name,
                     "Nodes": block["nodes"],
                     "NodeIDs": block["node_ids"],
                     "Elements": block["elements"],
                     "ElementIDs": block["element_ids"],
                     "NodeSets": {},
                     "ElementSets": {},
                     "FaceSets": {}}

        self.meshes.append(mesh_dict)




    def makeHalfCylinder(self, name=None, r_major=1.0, r_minor=1.0, z_length=1.0,
                         r_major_div=10, r_minor_div=10, z_div=10,
                         r_major_edge=-1.0, r_minor_edge=-1.0, z_edge=-1.0):
        r"""
        Create a half-elliptic cylinder.

        Parameters
        ----------
        name : str, optional
            Name identifier for mesh. Defaults to "Part {:s}".format(index in meshes)
        r_major : float 1.0, optional
             Major radius of the elliptical z-slices
        r_minor : float 1.0, optional
             Minor radius of the elliptical z-slices
        z_length : float 1.0, optional
             Cylinder height.
        r_major_div : int=10, optional
             Number of element divisions in the major radius direction
        r_minor_div : int=10, optional
             Number of element divisions in the minor radius direction
        z_div : int=10, optional
             Number of element divisions in the z-direction
        r_major_edge : float=-1.0, optional
             Overrides *r_major_div* with absolute edge length (ignored if non-positive)
        r_minor_edge : float=-1.0, optional
             Overrides *r_minor_div* with absolute edge length (ignored if non-positive)
        z_edge : float=-1.0, optional
             Overrides *z_div* with absolute edge length (ignored if non-positive)

        Returns
        -------
        Appends a half elliptic cylinder mesh definition to *meshes*.
        """
        xmin = r_major * np.cos(3*np.pi/4.0)
        xmax = r_major * np.cos(np.pi/4.0)
        ymin = r_minor * np.sin(3*np.pi/4.0)
        ymax = r_minor * np.sin(np.pi/4.0)
        xleft = xmin + ymin - r_minor / 2.0
        xright = r_minor / 2.0 - ymax + xmax
        inner_divx = int(np.ceil((xright - xleft) / r_major * r_major_div))
        outer_divx = np.ceil(2.0*r_major_div - inner_divx)
        outer_divy = np.ceil(r_minor / 2.0 * r_minor_div)
        outer_div = int(np.ceil((outer_divx + outer_divy) / 2.0))
        blockList = []
        # middle block
        blockList.append(self._makeBlock(x_origin=xleft,
                                         y_origin=0.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=xright - xleft,
                                         y_length=r_minor / 2.0,
                                         z_length=z_length,
                                         x_div=inner_divx,
                                         y_div=r_minor_div,
                                         z_div=z_div))
        # left block
        blockList.append(self._makeBlock(x_origin=-r_major,
                                         y_origin=0.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=r_major + xleft,
                                         y_length=r_minor / 2.0,
                                         z_length=z_length,
                                         x_div=outer_div,
                                         y_div=r_minor_div,
                                         z_div=z_div))
        # right block
        blockList.append(self._makeBlock(x_origin=xright,
                                         y_origin=0.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=r_major - xright,
                                         y_length=r_minor / 2.0,
                                         z_length=z_length,
                                         x_div=outer_div,
                                         y_div=r_minor_div,
                                         z_div=z_div))
        # top block
        blockList.append(self._makeBlock(x_origin=xleft,
                                         y_origin=r_minor / 2.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=xright - xleft,
                                         y_length=r_minor / 2.0,
                                         z_length=z_length,
                                         x_div=inner_divx,
                                         y_div=outer_div,
                                         z_div=z_div))

        block_positions = ('left', 'right', 'top')
        block_angles = ([np.pi, 3*np.pi / 4.0],
                        [0, np.pi / 4.0],
                        [3*np.pi / 4.0, np.pi / 4.0])

        for i, b in enumerate(blockList[1:]):
            blockList[i+1] = self._interpCylindrical(b, r_major, r_minor, block_angles[i], block_positions[i], 'half')
        block = self._mergeBlocks(blockList)
        mesh_dict = {"Name": name,
                     "Nodes": block["nodes"],
                     "NodeIDs": block["node_ids"],
                     "Elements": block["elements"],
                     "ElementIDs": block["element_ids"],
                     "NodeSets": {},
                     "ElementSets": {},
                     "FaceSets": {}}

        self.meshes.append(mesh_dict)



    def makeQuarterCylinder(self, name=None, r_major=1.0, r_minor=1.0, z_length=1.0,
                            r_major_div=10, r_minor_div=10, z_div=10,
                            r_major_edge=-1.0, r_minor_edge=-1.0, z_edge=-1.0):
        r"""
        Create a quarter-elliptic cylinder.

        Parameters
        ----------
        name : str, optional
            Name identifier for mesh. Defaults to "Part {:s}".format(index in meshes)
        r_major : float 1.0, optional
             Major radius of the elliptical z-slices
        r_minor : float 1.0, optional
             Minor radius of the elliptical z-slices
        z_length : float 1.0, optional
             Cylinder height.
        r_major_div : int=10, optional
             Number of element divisions in the major radius direction
        r_minor_div : int=10, optional
             Number of element divisions in the minor radius direction
        z_div : int=10, optional
             Number of element divisions in the z-direction
        r_major_edge : float=-1.0, optional
             Overrides *r_major_div* with absolute edge length (ignored if non-positive)
        r_minor_edge : float=-1.0, optional
             Overrides *r_minor_div* with absolute edge length (ignored if non-positive)
        z_edge : float=-1.0, optional
             Overrides *z_div* with absolute edge length (ignored if non-positive)

        Returns
        -------
        Appends an elliptic cylinder mesh definition to *meshes*.
        """
        xmax = r_major * np.cos(np.pi/4.0)
        ymax = r_minor * np.sin(np.pi/4.0)
        xright = r_minor / 2.0 - ymax + xmax
        inner_divx = int(np.ceil(xright / r_major * r_major_div))
        outer_divx = np.ceil(2.0*r_major_div - inner_divx)
        outer_divy = np.ceil(r_minor / 2.0 * r_minor_div)
        outer_div = int(np.ceil((outer_divx + outer_divy) / 2.0))
        blockList = []
        # middle block
        blockList.append(self._makeBlock(x_origin=0.0,
                                         y_origin=0.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=xright,
                                         y_length=r_minor / 2.0,
                                         z_length=z_length,
                                         x_div=inner_divx,
                                         y_div=r_minor_div,
                                         z_div=z_div))
        # right block
        blockList.append(self._makeBlock(x_origin=xright,
                                         y_origin=0.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=r_major - xright,
                                         y_length=r_minor / 2.0,
                                         z_length=z_length,
                                         x_div=outer_div,
                                         y_div=r_minor_div,
                                         z_div=z_div))
        # top block
        blockList.append(self._makeBlock(x_origin=0.0,
                                         y_origin=r_minor / 2.0,
                                         z_origin=-z_length / 2.0,
                                         x_length=xright,
                                         y_length=r_minor / 2.0,
                                         z_length=z_length,
                                         x_div=inner_divx,
                                         y_div=outer_div,
                                         z_div=z_div))

        block_positions = ('right', 'top')
        block_angles = ([0, np.pi / 4.0],
                        [np.pi / 2.0, np.pi / 4.0])

        for i, b in enumerate(blockList[1:]):
            blockList[i+1] = self._interpCylindrical(b, r_major, r_minor, block_angles[i], block_positions[i], 'quarter')
        block = self._mergeBlocks(blockList)
        mesh_dict = {"Name": name,
                     "Nodes": block["nodes"],
                     "NodeIDs": block["node_ids"],
                     "Elements": block["elements"],
                     "ElementIDs": block["element_ids"],
                     "NodeSets": {},
                     "ElementSets": {},
                     "FaceSets": {}}

        self.meshes.append(mesh_dict)


    def makeEllipsoid(self, name=None, r_major=1.0, r_middle=1.0, r_minor=1.0,
                      r_major_div=10, r_middle_div=10, r_minor_div=10,
                      r_major_edge=-1.0, r_middle_edge=-1.0, r_minor_edge=-1.0):
        r"""
        Create a sphere.

        Parameters
        ----------
        name : str, optional
            Name identifier for mesh. Defaults to "Part {:s}".format(index in meshes)
        r_major : float 1.0, optional
            Major radius of ellipsoid
        r_middle : float 1.0, optional
            Middle radius of ellipsoid
        r_minor : float 1.0, optional
            Minor radius of ellipsoid
        r_major_div : int 10, optional
            Number of element divisions in the major radial direction
        r_middle_div : int 10, optional
            Number of element divisions in the middle radial direction
        r_minor_div : int 10, optional
            Number of element divisions in the minor radial direction
        r_major_edge : float=-1.0, optional
            Overrides *r_major_div* with absolute radial edge length (ignored if non-positive)
        r_middle_edge : float=-1.0, optional
            Overrides *r_middle_div* with absolute radial edge length (ignored if non-positive)
        r_minor_edge : float=-1.0, optional
            Overrides *r_minor_div* with absolute radial edge length (ignored if non-positive)

        Returns
        -------
        Appends an ellipsoidal mesh definition to *meshes*.
        """
        blockList = []
        #middle
        blockList.append(self._makeBlock(x_origin=-r_major / 2.0,
                                         y_origin=-r_middle / 2.0,
                                         z_origin=-r_minor / 2.0,
                                         x_length=r_major,
                                         y_length=r_middle,
                                         z_length=r_minor,
                                         x_div=r_major_div,
                                         y_div=r_middle_div,
                                         z_div=r_minor_div))
        # top
        blockList.append(self._makeBlock(x_origin=-r_major / 2.0,
                                         y_origin=-r_middle / 2.0,
                                         z_origin=r_minor / 2.0,
                                         x_length=r_major,
                                         y_length=r_middle,
                                         z_length=r_minor / 2.0,
                                         x_div=r_major_div,
                                         y_div=r_middle_div,
                                         z_div=int(r_minor_div / 2.0)))
        # bottom
        blockList.append(self._makeBlock(x_origin=-r_major / 2.0,
                                         y_origin=-r_middle / 2.0,
                                         z_origin=-r_minor,
                                         x_length=r_major,
                                         y_length=r_middle,
                                         z_length=r_minor / 2.0,
                                         x_div=r_major_div,
                                         y_div=r_middle_div,
                                         z_div=int(r_minor_div / 2.0)))
        # left
        blockList.append(self._makeBlock(x_origin=-r_major,
                                         y_origin=-r_middle / 2.0,
                                         z_origin=-r_minor / 2.0,
                                         x_length=r_major / 2.0,
                                         y_length=r_middle,
                                         z_length=r_minor,
                                         x_div=int(r_major_div / 2.0),
                                         y_div=r_middle_div,
                                         z_div=r_minor_div))
        # right
        blockList.append(self._makeBlock(x_origin=r_major / 2.0,
                                         y_origin=-r_middle / 2.0,
                                         z_origin=-r_minor / 2.0,
                                         x_length=r_major / 2.0,
                                         y_length=r_middle,
                                         z_length=r_minor,
                                         x_div=int(r_major_div / 2.0),
                                         y_div=r_middle_div,
                                         z_div=r_minor_div))
        # back
        blockList.append(self._makeBlock(x_origin=-r_major / 2.0,
                                         y_origin=-r_middle,
                                         z_origin=-r_minor / 2.0,
                                         x_length=r_major,
                                         y_length=r_middle / 2.0,
                                         z_length=r_minor,
                                         x_div=r_major_div,
                                         y_div=int(r_middle_div / 2.0),
                                         z_div=r_minor_div))
        # front
        blockList.append(self._makeBlock(x_origin=-r_major / 2.0,
                                         y_origin=r_middle / 2.0,
                                         z_origin=-r_minor / 2.0,
                                         x_length=r_major,
                                         y_length=r_middle / 2.0,
                                         z_length=r_minor,
                                         x_div=r_major_div,
                                         y_div=int(r_middle_div / 2.0),
                                         z_div=r_minor_div))
        cases = ["top", "bottom", "left", "right", "back", "front"]
        for i, b in enumerate(blockList[1:]):
            blockList[i+1] = self._interpSpherical(b, r_major, r_middle, r_minor, case=cases[i], sym="none")

        block = self._mergeBlocks(blockList)

        mesh_dict = {"Name": name,
                     "Nodes": block["nodes"],
                     "NodeIDs": block["node_ids"],
                     "Elements": block["elements"],
                     "ElementIDs": block["element_ids"],
                     "NodeSets": {},
                     "ElementSets": {},
                     "FaceSets": {}}

        self.meshes.append(mesh_dict)

    def makeHalfSphere(self, **kwargs):
        r"""
        Create a half-sphere.

        Parameters
        ----------
        name : str, optional
            Name identifier for mesh. Defaults to "Part {:s}".format(index in meshes)
        r : float 1.0, optional
            Radius of sphere
        r_div : int 10, optional
            Number of element divisions in the radial direction
        r_edge : float=-1.0, optional
            Overrides *r_div* with absolute radial edge length (ignored if non-positive)
        subset : str="zn", optional
            Subset of sphere to mesh.
            "xn" : negative x
            "xp" : positive x
            "yn" : negative y
            "yp" : positive y
            "zn" : negative z
            "zp" : positive z

        Returns
        -------
        Appends a half-sphere mesh definition to *meshes*.
        """
        pass


    def makeQuarterSphere(self, **kwargs):
        r"""
        Create a quarter-sphere.

        Parameters
        ----------
        name : str, optional
            Name identifier for mesh. Defaults to "Part {:s}".format(index in meshes)
        r : float 1.0, optional
            Radius of sphere
        r_div : int 10, optional
            Number of element divisions in the radial direction
        r_edge : float=-1.0, optional
            Overrides *r_div* with absolute radial edge length (ignored if non-positive)
        subset : str="xp_zn", optional
            Subset of sphere to mesh.
            "xn_yn" : negative x and negative y
            "xn_yp" : negtive x and positive y
            "xp_yn" : postive x and negative y
            "xp_yp" : positive x and positive y
            "xn_zn" : negative x and negative z
            "xn_zp" : negative x and positive z
            "xp_zn" : postive x and negative z
            "xp_zp" : positive x and positive z
            "yn_zn" : negative y and negative z
            "yn_zp" : negative y and positive z
            "yp_zn" : postive y and negative z
            "yp_zp" : positive y and positive z

        Returns
        -------
        Appends a quarter-sphere mesh definition to *meshes*.
        """
        pass


    def makeEighthSphere(self, **kwargs):
        r"""
        Create an eighth-sphere.

        Parameters
        ----------
        name : str, optional
            Name identifier for mesh. Defaults to "Part {:s}".format(index in meshes)
        r : float 1.0, optional
            Radius of sphere
        r_div : int 10, optional
            Number of element divisions in the radial direction
        r_edge : float=-1.0, optional
            Overrides *r_div* with absolute radial edge length (ignored if non-positive)
        subset : str="xp_yp_zn", optional
            Subset of sphere to mesh.
            "xn_yn_zn" : negative x and negative y negative z
            "xn_yn_zp" : negative x and negative y positive z
            "xn_yp_zn" : negative x and positive y negative z
            "xn_yp_zp" : negative x and positive y positive z
            "xp_yn_zn" : positive x and negative y negative z
            "xp_yn_zp" : positive x and negative y positive z
            "xp_yp_zn" : positive x and positive y negative z
            "xp_yp_zp" : positive x and positive y positive z

        Returns
        -------
        Appends an eighth-sphere mesh definition to *meshes*.
        """
        pass

    def _interpCylindrical(self, block, a, b, t, case, sym):
        r"""
        Map nodal coordinates of passed block to cylinder defined in polar.

        Parameters
        ----------
        block : dict
            The block to map.
        a : float
            The major radius of the elliptical cross-section
        b : float
            The minor radius of the elliptical cross-section
        t : [float, float]
            The domain of angles to interpolate between (specified in radians)
        case : str
            Side of block to project to arc.
            Options - 'left', 'right', 'top', 'bottom'
        sym : str
            Symmetry case.
            Options - 'none', 'half', 'quarter'

        Returns
        -------
        The block with mapped nodal coordinates.
        """
        # define nodal x and y coordinates in natural space [0,1]
        natural = np.copy(block["nodes"])
        natural[:,0:2] -= np.min(natural[:,0:2], 0)
        natural[:,0:2] /= np.max(natural[:,0:2], 0)

        if case == 'top':
            arc = np.vstack((np.cos((t[1] - t[0])*natural[:,0] + t[0]), np.sin((t[1] - t[0])*natural[:,0] + t[0]), natural[:,2]))
            arc *= np.array([[a], [b], [1]])
            yint = np.min(block["nodes"][:, 1])
            if sym == 'none' or sym == 'half':
                line = np.vstack((2* natural[:,0] * np.max(block["nodes"][:,0]) + np.min(block["nodes"][:,0]),
                                yint * np.ones(block["nodes"].shape[0]),
                                natural[:, 2]))
            elif sym == 'quarter':
                line = np.vstack((natural[:,0] * np.max(block["nodes"][:,0]),
                                yint * np.ones(block["nodes"].shape[0]),
                                natural[:, 2]))
            image = natural[:,1] * arc + (1 - natural[:,1]) * line
            block["nodes"] = image.T
        elif case == 'bottom':
            arc = np.vstack((np.cos((t[1] - t[0])*natural[:,0] + t[0]), np.sin((t[1] - t[0])*natural[:,0] + t[0]), natural[:,2]))
            arc *= np.array([[a], [b], [1]])
            yint = np.max(block["nodes"][:, 1])
            line = np.vstack((2* natural[:,0] * np.max(block["nodes"][:,0]) + np.min(block["nodes"][:,0]),
                              yint * np.ones(block["nodes"].shape[0]),
                              natural[:, 2]))

            image = natural[:,1] * line + (1 - natural[:,1]) * arc
            block["nodes"] = image.T
        elif case == 'left':
            arc = np.vstack((np.cos((t[1] - t[0])*natural[:,1] + t[0]), np.sin((t[1] - t[0])*natural[:,1] + t[0]), natural[:,2]))
            arc *= np.array([[a], [b], [1]])
            if sym == 'none':
                line = np.vstack((np.max(block["nodes"][:,0]) * np.ones(block["nodes"].shape[0]),
                                2*natural[:,1] * np.max(block["nodes"][:,1]) + np.min(block["nodes"][:,1]),
                                natural[:, 2]))
            elif sym == 'half':
                line = np.vstack((np.max(block["nodes"][:,0]) * np.ones(block["nodes"].shape[0]),
                                  natural[:,1] * np.max(block["nodes"][:,1]),
                                  natural[:, 2]))

            image = natural[:,0] * line + (1 - natural[:,0]) * arc
            block["nodes"] = image.T
        elif case == 'right':
            arc = np.vstack((np.cos((t[1] - t[0])*natural[:,1] + t[0]), np.sin((t[1] - t[0])*natural[:,1] + t[0]), natural[:,2]))
            arc *= np.array([[a], [b], [1]])
            if sym == 'none':
                line = np.vstack((np.min(block["nodes"][:,0]) * np.ones(block["nodes"].shape[0]),
                                2*natural[:,1] * np.max(block["nodes"][:,1]) + np.min(block["nodes"][:,1]),
                                natural[:, 2]))
            elif sym == 'quarter' or sym == 'half':
                line = np.vstack((np.min(block["nodes"][:,0]) * np.ones(block["nodes"].shape[0]),
                                natural[:,1] * np.max(block["nodes"][:,1]),
                                natural[:, 2]))
            image = natural[:,0] * arc + (1 - natural[:,0]) * line
            block["nodes"] = image.T

        return block

    def _interpSpherical(self, block, a, b, c, case, sym):
        r"""
        Map nodal coordinates of passed block to cylinder defined in polar.

        Parameters
        ----------
        block : dict
            The block to map.
        a : float
            The major radius of the ellipsoid
        b : float
            The middle radius of the ellipsoid
        c : float
            The minor radius of the ellipsoid
        case : str
            Side of block to project to arc.
            Options - 'left', 'right', 'top', 'bottom', 'back', 'front'
        sym : str
            Symmetry case.
            Options - 'none', 'half', 'quarter', 'eighth'

        Returns
        -------
        The block with mapped nodal coordinates.
        """

        # find direction vectors for projection to ellipsoid
        p, v = self._findProjectionDirection(block["nodes"], case)
        # the intersection of each ray with the ellipsoid is the solution to a quadratic
        # ray: p[0] = p_0[0] + v[0]*t, p[1] = p_0[1] + v[1]*t, p[2] = p_0[2] + v[2]*t
        # ellipsoid: p[0]**2 / a**2 + p[1]**2 / b**2 + p[2]**2 / c**2 = 1
        # substituting, expanding, and grouping terms yields quadratic formula terms:
        # qa = (v[0]**2 / a**2 + v[1]**2 / b**2 +v[2]**2 / c**2)
        # qb = 2*(p[0]*v[0] / a**2 + p[1]*v[1] / b**2 + p[2]*v[2 / c**2)
        # qc = p_0[0]**2 / a**2 + p_0[1]**2 / b**2 + p_0[2]**2 / c**2 - 1
        qa = np.sum((v / [a, b, c]) ** 2, 1)
        qb = np.sum(2 * p * v / [a ** 2, b ** 2, c ** 2], 1)
        qc = np.sum((p / [a, b, c]) ** 2, 1) - 1.0
        i1 = (-qb + np.sqrt(qb ** 2 - 4 * qa * qc)) / (2 * qa)
        i2 = (-qb - np.sqrt(qb ** 2 - 4 * qa * qc)) / (2 * qa)
        wrong_ints = np.argwhere(i1 < 0.0)
        i1[wrong_ints] = i2[wrong_ints]
        intersections = p + np.multiply(v, i1.reshape(-1, 1))
        if case == 'top':
            t = block["nodes"][:,2] - np.min(block["nodes"][:,2])
            t /= np.max(t)
            image = p + t.reshape(-1, 1)*v*(np.linalg.norm(intersections - p, axis=1).reshape(-1, 1))
            block["nodes"] = image
        elif case == 'bottom':
            t = block["nodes"][:,2] - np.max(block["nodes"][:,2])
            t /= np.min(t)
            image = p + t.reshape(-1, 1)*v*(np.linalg.norm(intersections - p, axis=1).reshape(-1, 1))
            block["nodes"] = image
        elif case == 'left':
            t = block["nodes"][:,0] - np.max(block["nodes"][:,0])
            t /= np.min(t)
            image = p + t.reshape(-1, 1)*v*(np.linalg.norm(intersections - p, axis=1).reshape(-1, 1))
            block["nodes"] = image
        elif case == 'right':
            t = block["nodes"][:,0] - np.min(block["nodes"][:,0])
            t /= np.max(t)
            image = p + t.reshape(-1, 1)*v*(np.linalg.norm(intersections - p, axis=1).reshape(-1, 1))
            block["nodes"] = image
        elif case == 'back':
            t = block["nodes"][:,1] - np.max(block["nodes"][:,1])
            t /= np.min(t)
            image = p + t.reshape(-1, 1)*v*(np.linalg.norm(intersections - p, axis=1).reshape(-1, 1))
            block["nodes"] = image
        elif case == 'front':
            t = block["nodes"][:,1] - np.min(block["nodes"][:,1])
            t /= np.max(t)
            image = p + t.reshape(-1, 1)*v*(np.linalg.norm(intersections - p, axis=1).reshape(-1, 1))
            block["nodes"] = image

        return block

    def _mergeBlocks(self, blockList, tol=1e-14):
        r"""
        Merges a list of blocks into one mesh. Nodes within a tolerance are merged.


        Parameters
        ----------
        blockList : [dict,]
              list of block dictionaries
        tol : float 1e-14, optional
              merge nodes within this distance

        Returns
        -------
        Mesh with coincident nodes of blocks in list merged.
        """

        # construct new superblock from block list shifting node and element IDs appropriately
        new_block = {"nodes": blockList[0]["nodes"],
                     "node_ids": blockList[0]["node_ids"].ravel(),
                     "elements": blockList[0]["elements"],
                     "element_ids": blockList[0]["element_ids"].ravel()}
        for b in blockList[1:]:
            new_block["nodes"] = np.vstack((new_block["nodes"], b["nodes"]))
            new_block["elements"] = np.vstack((new_block["elements"], b["elements"] + new_block["node_ids"][-1]))
            new_block["node_ids"] = np.concatenate((new_block["node_ids"], b["node_ids"] + new_block["node_ids"][-1])).ravel()
            new_block["element_ids"] = np.concatenate((new_block["element_ids"], b["element_ids"] + new_block["element_ids"][-1])).ravel()

        # build a KD-tree to quickly find coincident nodes
        tree = cKDTree(new_block["nodes"])
        remove = tree.query_pairs(tol, output_type='ndarray') + 1
        updateMap = dict(zip(range(1, new_block["node_ids"].size + 1), range(1, new_block["node_ids"].size + 1)))
        for i in xrange(remove.shape[0]):
            if remove[i,0] < updateMap[remove[i,1]]:
                updateMap[remove[i,1]] = remove[i,0]
        updateFunc = lambda k: updateMap.get(k, -1)
        new_block["elements"] = np.vectorize(updateFunc)(new_block["elements"])
        unique_nodes = np.unique(np.array(updateMap.values()))
        new_block["nodes"] = new_block["nodes"][unique_nodes - 1, :]
        listOfNodes = range(1, unique_nodes.shape[0] + 1)
        updateMap = dict(zip(sorted(unique_nodes), listOfNodes))
        new_block["elements"] = np.vectorize(updateFunc)(new_block["elements"])
        new_block["node_ids"] = np.array(listOfNodes)
        new_block["element_ids"] = np.arange(new_block["elements"].shape[0]) + 1
        return new_block

    def _findProjectionDirection(self, points, case):
        directions = np.copy(points)
        mins  = np.min(directions, 0)
        maxes = np.max(directions, 0)
        h = np.min(np.abs(np.vstack((mins, maxes))), 0)
        l = maxes - mins
        t45 = np.tan(np.pi / 4.0)
        x = h / t45
        r = 1.0 - 2 * x / l
        scaled = points * r
        if case == "top":
            directions[:,2] = mins[2]
            scaled[:,2] = 0.0
        elif case == "bottom":
            directions[:,2] = maxes[2]
            scaled[:,2] = 0.0
        elif case == "left":
            directions[:,0] = maxes[0]
            scaled[:,0] = 0.0
        elif case == "right":
            directions[:,0] = mins[0]
            scaled[:,0] = 0.0
        elif case == "back":
            directions[:,1] = maxes[1]
            scaled[:,1] = 0.0
        elif case == "front":
            directions[:,1] = mins[1]
            scaled[:,1] = 0.0

        d = directions - scaled
        return directions, d / np.linalg.norm(d, axis=1).reshape(-1, 1)

    def _findEllipIntersect(self, a, b, c, directions):
        pass

    def applyRigidTransform(self, mesh_index=None, **kwargs):
        r"""
        Applies a rigid transformation to the mesh indicated by its index.
        Specify rotations either by a matrix or as x, y, and z rotation angles (degrees).

        .. note::

        When using angles, the order of application is x then y then z.

        Parameters
        ----------
        translation : [float=0.0, float=0.0, float=0.0], optional
        method : str="angles", optional
            "angles" : use rotation angles
            "matrix" : use rotation matrix
        rot_matrix : ndarray((3,3), dtype=float), optional
            If unspecified defaults to identity.
        rx : float=0.0
            x rotation angle (degrees)
        ry : float=0.0
            y rotation angle (degrees)
        rz : float=0.0
            z rotation angle (degrees)

        Returns
        -------
        Modifies mesh nodal coordinates for specified *meshes* item.
        """
        pass


    def defineNodeSet(self, mesh_index=None, **kwargs):
        r"""
        Define an additonal node set for mesh indicated by *mesh_index*.

        Parameters
        ----------
        name : str, required
            Name to assign to the set. WARNING: If the name already exists
            either it will be overwritten or be replaced by the union depending
            on value of *overwrite*.
        node_ids : ndarray((n,1), dtype=int)
        overwrite : bool=True, optional
            If a set with the indicated name exists overwrite is True. Otherwise,
            take the union of new set with old.

        Returns
        -------
        Appends mesh dictionary key: "Node Sets" the key (name, node_ids)
        """


    def defineElementSet(self, mesh_index=None, **kwargs):
        r"""
        Define an additonal element set for mesh indicated by *mesh_index*.

        Parameters
        ----------
        name : str, required
            Name to assign to the set. WARNING: If the name already exists
            either it will be overwritten or be replaced by the union depending
            on value of *overwrite*.
        element_ids : ndarray((n,1), dtype=int)
        overwrite : bool=True, optional
            If a set with the indicated name exists overwrite is True. Otherwise,
            take the union of new set with old.

        Return
        -------
        Appends mesh dictionary key: "Element Sets" the key (name, element_ids)
        """

    def defineSurfaceSet(self, mesh_index=None, **kwargs):
        r"""
        Define an additonal surface set for mesh indicated by *mesh_index*.

        Parameters
        ----------
        name : str, required
            Name to assign to the set. WARNING: If the name already exists
            either it will be overwritten or be replaced by the union depending
            on value of *overwrite*.
        faces : ndarray((n,4), dtype=int)
            Each row is a nodal connectivity list for a face.
        overwrite : bool=True, optional
            If a set with the indicated name exists overwrite is True. Otherwise,
            take the union of new set with old.

        Returns
        -------
        Appends mesh dictionary key: "Face Sets" the key (name, faces)
        """

    def writeToFile(self, filename="mesh.inp", file_format="abaqus"):
        r"""
        Parameters
        ----------
        filename : str, optional
            If provided, this will write the mesh to file. Currently, Abaqus (.inp) and VTK (.vtu)
            formats are supported. The format will be deduced by the file extension unless overridden
            by *format* parameter. If no filename is specified the mesh will be stored in memory.
        file_format : str="file", optional
            File format to save mesh in.
                * "file": determine from extension
                * "vtk": use VTK format
                * "abaqus": use Abaqus format
        """
        fid = open(filename, "wt")
        if file_format == "abaqus":
            for m in self.meshes:
                fid.write("*PART,NAME={:s}\n".format(m["Name"]))
                fid.write("*NODE\n")
                for i in xrange(m["NodeIDs"].shape[0]):
                    fid.write("{:d},{:.5e},{:.5e},{:.5e}\n".format(
                        m["NodeIDs"][i],
                        m["Nodes"][i, 0],
                        m["Nodes"][i, 1],
                        m["Nodes"][i, 2]))
                fid.write("*ELEMENT,TYPE=C3D8\n")
                for i in xrange(m["ElementIDs"].shape[0]):
                    fid.write("{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}\n".format(
                        m["ElementIDs"][i],
                        m["Elements"][i, 0],
                        m["Elements"][i, 1],
                        m["Elements"][i, 2],
                        m["Elements"][i, 3],
                        m["Elements"][i, 4],
                        m["Elements"][i, 5],
                        m["Elements"][i, 6],
                        m["Elements"][i, 7]))
                for key, value in m["NodeSets"].items():
                    fid.write("*NSET,NSET={:s}\n".format(key)),
                    split = np.array_split(value, np.ceil(value.shape[0] / 10.0))
                    for s in split:
                        for i in xrange(s.shape[0]):
                            fid.write("{:d},".format(s[i]))
                        fid.write("\n")
                for key, value in m["ElementSets"].items():
                    fid.write("*ELSET,ELSET={:s}\n".format(key)),
                    split = np.array_split(value, np.ceil(value.shape[0] / 10.0))
                    for s in split:
                        for i in xrange(s.shape[0]):
                            fid.write("{:d},".format(s[i]))
                        fid.write("\n")
        fid.close()
