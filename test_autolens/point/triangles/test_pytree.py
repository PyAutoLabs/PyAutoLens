import autofit as af
from autogalaxy.profiles.mass import Isothermal


def test_isothermal_pytree():
    model = af.Model(Isothermal)

    children, aux = model.instance_flatten(Isothermal())
    instance = model.instance_unflatten(aux, children)

    assert isinstance(instance, Isothermal)
    assert instance.centre == (0.0, 0.0)
