(source)=

<span id="building-from-source"></span>

<span id="packages"></span>

# Install from source

Use this route when you want to edit PyAutoLens. For running examples, use [conda](conda.md) or [pip](pip.md).

<span id="building-only-pyautolens"></span>

## Edit PyAutoLens

In an active Python environment:

```bash
git clone https://github.com/PyAutoLabs/PyAutoLens.git
cd PyAutoLens
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
python -c "import autolens; print(autolens.__file__)"
```

An editable install uses the code in your checkout. Dependencies are installed from PyPI; you do not need `conda-develop` or a manual `PYTHONPATH` entry. The printed path should point into your checkout.

<span id="building-all-projects"></span>

## Edit several PyAuto libraries

If your change also touches a parent library, install those checkouts together. From a folder containing none of these repositories yet:

```bash
git clone https://github.com/PyAutoLabs/PyAutoNerves.git
git clone https://github.com/PyAutoLabs/PyAutoFit.git
git clone https://github.com/PyAutoLabs/PyAutoArray.git
git clone https://github.com/PyAutoLabs/PyAutoGalaxy.git
git clone https://github.com/PyAutoLabs/PyAutoLens.git
python -m pip install -e ./PyAutoNerves -e ./PyAutoFit -e ./PyAutoArray -e ./PyAutoGalaxy -e ./PyAutoLens
python -m pip check
```

Use compatible branches or release tags across the repositories. A development branch can depend on changes that are not yet published.

## Run tests

From the PyAutoLens checkout:

```bash
python -m pip install -e ".[dev]"
python -m pytest test_autolens/
```

Some tests need additional packages or test data; follow the repository’s contributor instructions for the full test environment.

## JAX & GPU

See [JAX and GPU setup](acceleration.md#jax-and-gpus).
