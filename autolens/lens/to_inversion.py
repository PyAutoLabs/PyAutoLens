from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np

from autoconf import cached_property

import autoarray as aa
import autogalaxy as ag

from autoarray.inversion.inversion.factory import inversion_from


class TracerToInversion(ag.AbstractToInversion):
    def __init__(
        self,
        dataset: Optional[Union[aa.Imaging, aa.Interferometer, aa.DatasetInterface]],
        tracer,
        adapt_images: Optional[ag.AdaptImages] = None,
        settings: aa.Settings = None,
        xp=np,
    ):
        """
        Interfaces a dataset and tracer with the inversion module, to setup a linear algebra calculation.

        The tracer's galaxies may contain linear light profiles whose `intensity` values are solved for via linear
        algebra in order to best-fit the data. In this case, this class extracts the linear light profiles of all
        galaxies, performs ray-tracing and computes their images and passes them to the `inversion` module such that
        they become  the `mapping_matrix` used in the linear algebra calculation.

        The galaxies may also contain pixelizations, which use a mesh (e.g. a Voronoi mesh) and regularization scheme
        to reconstruct the galaxy's light. This class extracts all pixelizations, performs ray-tracing and uses the
        pixelizations to set up `Mapper` objects which pair the dataset and pixelization to again set up the
        appropriate `mapping_matrix` and other linear algebra matrices (e.g. the `regularization_matrix`).

        This class does not perform the inversion or compute any of the linear algebra matrices itself. Instead,
        it acts as an interface between the dataset and galaxies and the inversion module, extracting the
        necessary information from galaxies and passing it to the inversion module.

        The tracer's galaxies may also contain standard light profiles which have an input `intensity` which is not
        solved for via linear algebra. These profiles should have already been evaluated and subtracted from the
        dataset before the inversion is performed. This is how an inversion is set up in the fit
        modules (e.g. `FitImaging`).

        Parameters
        ----------
        dataset
            The dataset containing the data which the inversion is performed on.
        tracer
            The tracer whose galaxies are fitted to the dataset via the inversion.
        adapt_images
            Images which certain pixelizations use to adapt their properties to the dataset, for example congregating
            the pixelization's pixels to the brightest regions of the image.
        settings
            The settings of the inversion, which controls how the linear algebra calculation is performed.
        """
        self.tracer = tracer

        super().__init__(
            dataset=dataset,
            adapt_images=adapt_images,
            settings=settings,
            xp=xp,
        )

    @property
    def planes(self) -> List[List[ag.Galaxy]]:
        """
        The planes object of a tracer is a list of list of galaxies grouped into their planes, where planes
        contained all galaxies at the same unique redshift.

        The planes are used to set up the inversion, whereby linear light profiles and pixelizations are extracted
        and grouped based on the plane they are in.

        The reason for this is that ray-tracking is performed on a plane-by-plane basis, therefore using planes
        makes it more straight forward to extract the appropriate traced grid for each galaxy.

        Returns
        -------
        The planes of the tracer, which are used to set up the inversion.
        """
        return self.tracer.planes

    @property
    def has_mapper(self) -> bool:
        """
        Checks whether the tracer has a pixelization, which is required to set up the inversion.

        This function is used to ensure computation run time is not wasted performing certain calculations if they are
        not needed because the tracer does not have a pixelization.

        Returns
        -------
        True if the tracer has a pixelization, False if not.
        """
        for galaxies in self.planes:
            if galaxies.has(cls=aa.Pixelization):
                return True

    @cached_property
    def traced_grid_2d_list_of_inversion(self) -> List[aa.type.Grid2DLike]:
        """
        Returns a list of the traced grids of the inversion.

        For a standard two-plane lens system (e.g. a lens galaxy and source galaxy), assuming the lens galaxy
        has linear light profiles and source galaxy has a pixelization, this function would return an image-plane
        grid which has not been lensed and a source-plane grid which has been lensed.

        This function is short and could be called where it is used, however it is used in multiple functions
        and therefore is cached to ensure it is not recalculated multiple times.

        Returns
        -------
        The traced grids of the inversion, which are cached for efficiency.
        """
        return self.tracer.traced_grid_2d_list_from(
            grid=self.dataset.grids.pixelization, xp=self._xp
        )

    @cached_property
    def lp_linear_func_list_galaxy_dict(
        self,
    ) -> Dict[ag.LightProfileLinearObjFuncList, ag.Galaxy]:
        """
        Returns a dictionary associating each list of linear light profiles with the galaxy they belong to.

        You should first refer to the docstring of the `cls_light_profile_func_list_galaxy_dict_from` method in the
        parent project PyAutoGalaxy for a description of this method.

        In brief, this method iterates over all galaxies and their light profiles, extracting their linear light
        profiles and for each galaxy grouping them into a `LightProfileLinearObjFuncList` object, which is associated
        with the galaxy via the dictionary. It also extracts linear light profiles from `Basis` objects and makes this
        associated.

        When extracting the linear light profiles, ray-tracing is also performed to ensure that each grid input
        into the linear light profile corresponds to the grid for the plane the galaxy is in, derived from the
        galaxy's redshift.

        This function also handles some aspects of over-sampling, because the implementation of adaptive
        over-sampling in a tracer is quite confusing. My hope is that this will be removed in the future,
        so just try ignore it for now.

        The `LightProfileLinearObjFuncList` object contains the attributes (e.g. the data `grid` after ray tracing,
        `light_profiles`) and functionality (e.g. a `mapping_matrix` method) that are required to perform the inversion.

        This function first creates a dictionary of linear light profiles associated with each galaxy for each plane,
        and then does the same for all `Basis` objects. The two dictionaries are then combined and returned.

        Returns
        -------
        A dictionary associating each list of linear light profiles and basis objects with the galaxy they belong to.
        """
        if not self.tracer.perform_inversion:
            return {}

        lp_linear_galaxy_dict_list = {}

        traced_grids_of_planes_list = self.tracer.traced_grid_2d_list_from(
            grid=self.dataset.grids.lp, xp=self._xp
        )

        if self.dataset.grids.blurring is not None:
            traced_blurring_grids_of_planes_list = self.tracer.traced_grid_2d_list_from(
                grid=self.dataset.grids.blurring, xp=self._xp
            )
        else:
            traced_blurring_grids_of_planes_list = [None] * len(
                traced_grids_of_planes_list
            )

        for plane_index, galaxies in enumerate(self.planes):
            grids = aa.GridsInterface(
                lp=traced_grids_of_planes_list[plane_index],
                blurring=traced_blurring_grids_of_planes_list[plane_index],
            )

            dataset = aa.DatasetInterface(
                data=self.dataset.data,
                noise_map=self.dataset.noise_map,
                grids=grids,
                psf=self.psf,
                transformer=self.transformer,
                sparse_operator=self.dataset.sparse_operator,
            )

            galaxies_to_inversion = ag.GalaxiesToInversion(
                dataset=dataset,
                galaxies=galaxies,
                settings=self.settings,
                adapt_images=self.adapt_images,
                xp=self._xp,
            )

            lp_linear_galaxy_dict_of_plane = (
                galaxies_to_inversion.lp_linear_func_list_galaxy_dict
            )

            lp_linear_galaxy_dict_list = {
                **lp_linear_galaxy_dict_list,
                **lp_linear_galaxy_dict_of_plane,
            }

        return lp_linear_galaxy_dict_list

    def cls_pg_list_from(self, cls: Type) -> List[List]:
        """
        Returns a list of lists of objects in the tracer which are an instance of the input `cls`, where each inner
        list corresponds to a single plane,

        By grouping the objects extracted from this function (e.g. pixelizations, regularizations) by plane, it makes
        it straight forward to pair them with the appropriate ray-traced grid.

        The notation `_pg_` stands for `plane galaxy`, and indicates that the objects are grouped by plane
        after being extracted from galaxies in the tracer.

        Parameters
        ----------
        cls
            The type of class that a list of instances of this class in the galaxy are returned for.

        Returns
        -------
            The list of lists of objects that inherit from input `cls` in the galaxy grouped by plane.
        """
        return [galaxies.cls_list_from(cls=cls) for galaxies in self.planes]

    @cached_property
    def adapt_galaxy_image_pg_list(self) -> List[List[np.ndarray]]:
        """
        Returns a list of lists of adapt images, where each inner list corresponds to a single plane.

        An adapt image is an image that certain pixelizations use to adapt their properties to the dataset, for example
        congregating the pixelization's pixels to the brightest regions of the image.

        By grouping adapt images by plane, it makes it straight forward to pair them with the appropriate ray-traced
        grid.

        The notation `_pg_` stands for `plane galaxy`, and indicates that the objects are grouped by plane
        after being extracted from galaxies in the tracer.

        Returns
        -------
            The list of lists of adapt images grouped by plane.
        """
        adapt_galaxy_image_pg_list = []

        for galaxies in self.planes:
            if galaxies.has(cls=aa.Pixelization):
                plane_image_list = []

                galaxies_with_pixelization_list = galaxies.galaxies_with_cls_list_from(
                    cls=aa.Pixelization
                )

                for galaxy in galaxies_with_pixelization_list:
                    try:
                        image = self.adapt_images.galaxy_image_dict[galaxy]
                    except (AttributeError, KeyError, TypeError):
                        image = None

                    # Bug fix whereby for certain models the galaxy doesnt pair correctly.

                    if image is None and len(galaxies_with_pixelization_list) == 1:
                        galaxy_list = self.adapt_images.galaxy_image_dict.keys()
                        galaxy_with_pixelization = [
                            galaxy
                            for galaxy in galaxy_list
                            if galaxy.has(cls=aa.Pixelization)
                        ][0]

                        image = self.adapt_images.galaxy_image_dict[
                            galaxy_with_pixelization
                        ]

                    plane_image_list.append(image)

                adapt_galaxy_image_pg_list.append(plane_image_list)

            else:
                adapt_galaxy_image_pg_list.append([])

        return adapt_galaxy_image_pg_list

    @cached_property
    def image_plane_mesh_grid_pg_list(self) -> List[List]:
        """
        Returns a list of lists of image-plane mesh grids, where each inner list corresponds to a single plane.

        Certain pixelizations (e.g. the `VoronoiMagnification`) begin by placing what will become its the
        source-pixel centres in the image-plane. This is done by calculating the centres in the image-plane
        using an `image_mesh` object, and then ray-tracing these centres to the source-plane.

        This function computes the image-plane mesh grids for each plane, and returns them as a list of lists
        grouped by plane.

        By grouping the image-plane mesh grids by plane, it makes it straight forward to pair them with the appropriate
        ray-traced grid.

        The notation `_pg_` stands for `plane galaxy`, and indicates that the objects are grouped by plane
        after being extracted from galaxies in the tracer.

        Returns
        -------
            The list of lists of image-plane mesh grids grouped by plane.
        """

        image_plane_mesh_grid_list_of_planes = []

        for galaxies in self.planes:
            to_inversion = ag.GalaxiesToInversion(
                dataset=self.dataset,
                galaxies=galaxies,
                adapt_images=self.adapt_images,
                settings=self.settings,
                xp=self._xp,
            )

            image_plane_mesh_grid_list = to_inversion.image_plane_mesh_grid_list
            image_plane_mesh_grid_list_of_planes.append(image_plane_mesh_grid_list)

        return image_plane_mesh_grid_list_of_planes

    @cached_property
    def traced_mesh_grid_pg_list(self) -> List[List]:
        """
        Returns a list of lists of traced mesh grids, where each inner list corresponds to a single plane.

        Certain pixelizations (e.g. the `VoronoiMagnification`) begin by placing what will become its the
        source-pixel centres in the image-plane. This is done by calculating the centres in the image-plane
        using an `image_mesh` object, and then ray-tracing these centres to the source-plane.

        This function then uses the tracer to ray-trace these image-plane mesh grids to the source-plane, returning
        their grid of coordinates in the source-plane (e.g. after they have been lensed). These ray traced grids
        are input into the inversion to ensure the source reconstruction occurs in the source-plane.

        By grouping the traced mesh grids by plane, it makes it straight forward to pair them with the appropriate
        ray-traced grid.

        The notation `_pg_` stands for `plane galaxy`, and indicates that the objects are grouped by plane

        Returns
        -------
            The list of lists of traced mesh grids grouped by plane.
        """
        image_plane_mesh_grid_pg_list = self.image_plane_mesh_grid_pg_list

        traced_mesh_grid_pg_list = []

        for plane_index, galaxies in enumerate(self.planes):
            if image_plane_mesh_grid_pg_list[plane_index] is None:
                traced_mesh_grid_pg_list.append(None)
            else:
                traced_mesh_grids_list = []

                for image_plane_mesh_grid in image_plane_mesh_grid_pg_list[plane_index]:
                    try:
                        traced_mesh_grids_list.append(
                            self.tracer.traced_grid_2d_list_from(
                                grid=image_plane_mesh_grid, xp=self._xp
                            )[plane_index]
                        )
                    except AttributeError:
                        traced_mesh_grids_list.append(None)

                traced_mesh_grid_pg_list.append(traced_mesh_grids_list)

        return traced_mesh_grid_pg_list

    @cached_property
    def mapper_galaxy_dict(self) -> Dict[aa.Mapper, ag.Galaxy]:
        """
        Returns a dictionary associating each `Mapper` object with the galaxy it belongs to.

        The docstring of the function `mapper_from` in PyAutoGalaxy describes the `Mapper` object in detail, and is used
        in this function to create the `Mapper` objects which are associated with the galaxies.

        This function begins by extracting all galaxies with pixelizations, determining which have an image mesh
        (see `image_plane_mesh_grid_pg_list`), which require an adapt image (see `adapt_galaxy_image_pg_list`), and
        ray tracing these image-plane mesh grids to the source-plane (see `traced_mesh_grid_pg_list`).

        The tag `_pg_` stands for `plane galaxy`, and indicates that the objects are grouped by plane after being
        extracted from galaxies in the tracer. Because all of these objects are grouped by plane, it makes it
        straight forward for this function to pair them with the appropriate ray-traced grid and input them into
        the mapper of that plane.

        This function essentially finds all galaxies with pixelizations, performs all necessary calculations to
        set up the `Mapper` objects (e.g. compute the `image_plane_mesh_grid`), and then associates each `Mapper`
        with the galaxy it belongs to.

        Returns
        -------
        A dictionary associating each `Mapper` object with the galaxy it belongs to.
        """
        if not self.has_mapper:
            return {}

        mapper_galaxy_dict = {}

        traced_grids_of_planes_list = self.traced_grid_2d_list_of_inversion

        traced_mesh_grids_list_of_planes = self.traced_mesh_grid_pg_list
        image_plane_mesh_grid_list = self.image_plane_mesh_grid_pg_list

        for plane_index, galaxies in enumerate(self.planes):
            if galaxies.has(cls=aa.Pixelization):
                to_inversion = ag.GalaxiesToInversion(
                    dataset=self.dataset,
                    galaxies=galaxies,
                    adapt_images=self.adapt_images,
                    settings=self.settings,
                    xp=self._xp,
                )

                galaxies_with_pixelization_list = galaxies.galaxies_with_cls_list_from(
                    cls=aa.Pixelization
                )

                for mapper_index in range(
                    len(traced_mesh_grids_list_of_planes[plane_index])
                ):
                    pixelization_list = self.cls_pg_list_from(cls=aa.Pixelization)

                    try:
                        adapt_galaxy_image = self.adapt_galaxy_image_pg_list[
                            plane_index
                        ][mapper_index]
                    except AttributeError:
                        adapt_galaxy_image = None

                    mapper = to_inversion.mapper_from(
                        mesh=pixelization_list[plane_index][mapper_index].mesh,
                        regularization=pixelization_list[plane_index][
                            mapper_index
                        ].regularization,
                        source_plane_data_grid=traced_grids_of_planes_list[plane_index],
                        source_plane_mesh_grid=traced_mesh_grids_list_of_planes[
                            plane_index
                        ][mapper_index],
                        image_plane_mesh_grid=image_plane_mesh_grid_list[plane_index][
                            mapper_index
                        ],
                        adapt_galaxy_image=adapt_galaxy_image,
                    )

                    galaxy = galaxies_with_pixelization_list[mapper_index]

                    mapper_galaxy_dict[mapper] = galaxy

        return mapper_galaxy_dict

    @cached_property
    def inversion(self):
        """
        Returns an inversion object from the dataset, galaxies and inversion settings.

        The inversion uses all linear light profiles and pixelizations of the galaxies in the tracer to fit the data,
        fully accounting for ray tracing.

        It solves for the linear light profile intensities and pixelization mesh pixel values via linear algebra,
        finding the solution which best fits the data after regularization is applied.

        The `TracerToInversion` object acts as an interface between the dataset and tracer and the inversion module,
        with many of its functions required to set up the inputs to the inversion object, primarily
        the `linear_obj_list` and `linear_obj_galaxy_dict` properties.

        Returns
        -------
        The inversion object which fits the dataset using the tracer.
        """

        inversion = inversion_from(
            dataset=self.dataset,
            linear_obj_list=self.linear_obj_list,
            settings=self.settings,
            xp=self._xp,
        )

        inversion.linear_obj_galaxy_dict = self.linear_obj_galaxy_dict

        return inversion
