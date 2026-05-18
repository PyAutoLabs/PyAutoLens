=============
Mass Profiles
=============

Total [ag.mp]
-------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   PowerLawCore
   PowerLawCoreSph
   PowerLawBroken
   PowerLawBrokenSph
   IsothermalCore
   IsothermalCoreSph
   PowerLaw
   PowerLawSph
   Isothermal
   IsothermalSph
   dPIEMass
   dPIEMassSph
   PIEMass
   dPIEPotential
   dPIEPotentialSph

Mass Sheets [ag.mp]
-------------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ExternalShear
   ExternalPotential
   MassSheet

Multipoles [ag.mp]
------------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   PowerLawMultipole

Point Mass [ag.mp]
------------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   PointMass
   SMBH
   SMBHBinary

Stellar [ag.mp]
---------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Gaussian
   GaussianGradient
   Sersic
   SersicSph
   SersicCore
   SersicCoreSph
   Exponential
   ExponentialSph
   DevVaucouleurs
   DevVaucouleursSph
   SersicGradient
   SersicGradientSph
   Chameleon
   ChameleonSph

Dark [ag.mp]
------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   gNFW
   gNFWSph
   gNFWMCRLudlow
   gNFWVirialMassConcSph
   gNFWVirialMassgNFWConcSph
   NFW
   NFWSph
   NFWMCRDuffySph
   NFWMCRLudlow
   NFWMCRLudlowSph
   NFWMCRScatterLudlow
   NFWMCRScatterLudlowSph
   NFWVirialMassConcSph
   NFWTruncatedSph
   NFWTruncatedMCRDuffySph
   NFWTruncatedMCRLudlowSph
   NFWTruncatedMCRScatterLudlowSph
   cNFW
   cNFWSph
   cNFWMCRLudlow
   cNFWMCRLudlowSph
   cNFWMCRScatterLudlow
   cNFWMCRScatterLudlowSph

Stellar Light+Mass [ag.lmp]
---------------------------

Combined light-and-mass profiles whose ``image_2d_from`` and ``convergence_2d_from`` share
a single parametric shape via a ``mass_to_light_ratio`` parameter.

.. currentmodule:: autogalaxy.profiles.light_and_mass_profiles

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Gaussian
   GaussianGradient
   Sersic
   SersicSph
   SersicCore
   SersicCoreSph
   SersicGradient
   SersicGradientSph
   Exponential
   ExponentialSph
   ExponentialGradient
   ExponentialGradientSph
   DevVaucouleurs
   DevVaucouleursSph
   Chameleon
   ChameleonSph

Linear Light+Mass [ag.lmp_linear]
---------------------------------

The inversion-aware variants of ``ag.lmp.*`` — same parametric shapes and
``mass_to_light_ratio`` semantics, with ``intensity`` solved analytically via the linear
inversion during each likelihood evaluation rather than as a free non-linear parameter.

.. currentmodule:: autogalaxy.profiles.light_linear_and_mass_profiles

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Gaussian
   GaussianGradient
   Sersic
   SersicSph
   SersicCore
   SersicCoreSph
   SersicGradient
   SersicGradientSph
   Exponential
   ExponentialSph
   ExponentialGradient
   ExponentialGradientSph
   DevVaucouleurs
   DevVaucouleursSph
   Chameleon
   ChameleonSph
