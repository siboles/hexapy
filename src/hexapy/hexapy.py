import numpy as np

class Mesher(object):
    def __init__(self, filename=None, file_format="file"):
        r"""
        This class provides functionality for creating hexahedral meshes of primitives.

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

        Attributes
        ----------
        meshes : [, dict, ...]
            List of meshes stored as dictionaries. Dictionary fields are as follows:
            * 'Name' -- str, optional
               Name assigned to mesh - defaults to "Part i" as appended
            * 'Nodes' -- ndarray((N, 4), dtype=(int, float, ... )
               Array of node ids and coordinates
            * 'Elements' -- ndarray((N, 9), dtype=(int, ...) )
               Array of element ids [:,0] and nodal connectivity [:,1:]
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
        """

        self.filename = filename
        self.fileformat = file_format


    def makeBox(self, **kwargs):
        r"""
        Create a box.

        Parameters
        ----------
        name : str, optional
            Name identifier for mesh. Defaults to "Part {:s}".format(index in meshes)
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

        pass


    def makeCylinder(self, **kwargs):
        r"""
        Create an elliptic cylinder.

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
        pass


    def makeHalfCylinder(self, **kwargs):
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
        Appends an elliptic cylinder mesh definition to *meshes*.
        """
        pass


    def makeQuarterCylinder(self, **kwargs):
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
        pass


    def makeSphere(self, **kwargs):
        r"""
        Create a sphere.

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

        Returns
        -------
        Appends a sphere mesh definition to *meshes*.
        """
        pass


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
        Modifies mesh nodal coordinates for specifies *meshes* item.
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

        Returns
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
