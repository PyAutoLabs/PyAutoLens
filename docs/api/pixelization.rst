Pixelizations
=============

Reconstruct the extended background source on a mesh when a small set of analytic light profiles cannot describe its structure. The lens model maps image-plane coordinates to the source plane, where the reconstruction is defined.

The reconstruction values are solved by a linear inversion for each proposed model. Regularization constrains their variation, balancing the fit to the data against the smoothness of the reconstruction.

.. _pixelization:

Build a pixelization
--------------------

A ``Pixelization`` combines a ``mesh`` with a ``regularization`` scheme. This example constructs an adaptive rectangular mesh with constant regularization; it does not run a fit.

.. code-block:: python

    import autolens as al

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularBilinearAdaptDensity(shape=(20, 20)),
        regularization=al.reg.Constant(coefficient=1.0),
    )

The shape and regularization coefficient are illustrative choices. Their suitable values depend on the data and model. Attach the pixelization to the source galaxy and use the `worked pixelization examples <https://github.com/PyAutoLabs/autolens_workspace/blob/main/notebooks/imaging/features/pixelization/start_here.ipynb>`_ for the complete fitting workflow.

The detailed ``Pixelization`` reference below explains the coordinate mappings, masking and image-plane versus source-plane reconstruction with diagrams.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Pixelization

.. _mesh-ag-mesh:

Choose a mesh
-------------

Access these classes through ``al.mesh``. The generated pages show their implementation in **PyAutoArray**, which supplies the mesh classes to both libraries.

- **Density-adaptive rectangular meshes** allocate resolution according to the spatial density of the coordinates used for reconstruction, after ray tracing in a lens model.
- **Image-adaptive rectangular meshes** use an adapt image to concentrate resolution according to the estimated light distribution. Supply the adapt-image information required by the fitting workflow.
- **Delaunay** uses a triangular mesh and barycentric interpolation between its vertices. Its adaptive setup requires an image-plane mesh grid.

The Bilinear and RTU rectangular families use different coordinate transforms. RTU uses a smooth kernel-based transform; Bilinear uses ranks of the sampled coordinates. This distinction matters for gradient-based fitting. Consult the class-specific gradient and over-sampling notes before choosing a mesh for a JAX analysis.

.. currentmodule:: autoarray.inversion.mesh.mesh

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   RectangularBilinearAdaptDensity
   RectangularBilinearAdaptImage
   RectangularRTUAdaptDensity
   RectangularRTUAdaptImage
   Delaunay

.. _regularization-ag-reg:

Choose regularization
---------------------

Access these classes through ``al.reg``. Regularization controls how strongly the reconstruction is smoothed; it does not set the mesh resolution.

``Constant`` applies a shared coefficient, while adaptive schemes vary the smoothing using estimated signal. The ``Split`` variants use a different construction of the regularization constraints. Choose a scheme compatible with the mesh, as described in the detailed references.

``AdaptPower`` and ``AdaptSplitPower`` expose a coefficient-power convention that differs from the legacy adaptive classes. Coefficients and their priors are therefore not interchangeable across these families; check the definitions when changing scheme.

.. currentmodule:: autoarray.inversion.regularization

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Constant
   ConstantSplit
   Adapt
   AdaptSplit
   AdaptPower
   AdaptSplitPower

.. _image-mesh-ag-image-mesh:

Image-mesh utilities
--------------------

These lower-level utilities place sampling points in the image plane for workflows that require an image-plane mesh. They are not a third argument to ``Pixelization``: its constructor takes ``mesh`` and ``regularization``. Follow the selected mesh’s setup and adapt-image requirements.

.. currentmodule:: autoarray.inversion.mesh.image_mesh

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Overlay
   Hilbert
   KMeans

Settings
--------

``Settings`` controls numerical choices used when constructing and solving an inversion. Keep the defaults unless your analysis needs a particular option, and consult the parameter descriptions before changing them.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Settings

Mapper
------

A ``Mapper`` describes how reconstruction values contribute at the sampled coordinates. For lensing, those coordinates have been traced to the source plane. Mappers are normally built by the fitting machinery; their reference is useful when inspecting mappings or developing a custom calculation.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Mapper
