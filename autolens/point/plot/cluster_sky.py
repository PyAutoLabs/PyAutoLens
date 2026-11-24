"""Shared sky coordinates, family markers and intensity stretch for cluster figures."""

import numpy as np
from matplotlib.colors import AsinhNorm, to_hex

from autoarray.plot.utils import _arcsec_labels, apply_extent, apply_labels


def family_style(record):
    """Bright and dark colours and marker for one family on image and white backgrounds."""
    styles = [("#ffbc53", "#a15b0c", "o"), ("#5fe3ee", "#00778a", "D")]
    return (
        styles[record.k]
        if record.k < 2
        else (to_hex(record.color), to_hex(record.color), record.marker)
    )


def draw_cluster_image(ax, image):
    """Display the native raster with row zero at the largest y coordinate."""
    pixels = np.asarray(image.native)
    finite = pixels[np.isfinite(pixels)]
    high = max(float(np.percentile(finite, 99.95)), 1.0e-9) if finite.size else 1.0
    ax.imshow(
        pixels,
        origin="upper",
        extent=image.geometry.extent,
        cmap="gray",
        norm=AsinhNorm(linear_width=high * 0.01, vmin=0, vmax=high),
    )
    ax.set_facecolor("#23282d")


def cluster_field(extent, positions):
    """A square sky field around the cutout centre, expanded to include the supplied positions."""
    xmin, xmax, ymin, ymax = extent
    centre = np.array([(ymin + ymax) / 2, (xmin + xmax) / 2])
    half = max(
        0.375 * min(xmax - xmin, ymax - ymin),
        1.12
        * np.max(
            np.abs(positions[np.isfinite(positions).all(axis=1)] - centre), initial=0
        ),
    )
    return centre, half


def angular_axes(ax, centre, half_width, half_height=None, small=False):
    """Keep angular pixels square and retain precision around the physical centre."""
    half_height = half_width if half_height is None else half_height
    y, x = centre
    extent = (x - half_width, x + half_width, y - half_height, y + half_height)
    apply_labels(ax, is_subplot=True)
    apply_extent(ax, extent)
    for axis, value, radius in ((ax.xaxis, x, half_width), (ax.yaxis, y, half_height)):
        ticks = value + radius * np.array([-0.75, 0, 0.75])
        digits = max(1, int(np.ceil(-np.log10(radius))) + 1)
        labels = _arcsec_labels(np.round(ticks, digits))
        if len(set(labels)) < len(ticks):
            # Very small fields far from zero exceed the shared formatter's six significant digits.
            labels = [f"{t:.{digits}f}″".replace("-", "−") for t in ticks]
        axis.set_ticks(ticks, labels)
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8 if small else 11)
    ax.tick_params(axis="y", labelrotation=90)
    for label in ax.get_yticklabels():
        label.set_verticalalignment("center")
