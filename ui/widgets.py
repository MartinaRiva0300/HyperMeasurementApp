"""
Shared pyqtgraph helpers — a configured ImageView used by both the camera panel
and the measurement panel, matching the colormap from hyperspectral_measure.py.
"""

import numpy as np
import pyqtgraph as pg


# Same colormap as hyperspectral_measure.py.setup_figure()
_CMAP_COLORS = [
    (0, 0, 0),
    (45, 5, 61),
    (84, 42, 55),
    (150, 87, 60),
    (208, 171, 141),
    (255, 255, 255),
]


def make_image_view() -> pg.ImageView:
    """Return a pg.ImageView with the hyperspectral colormap applied.

    pyqtgraph's built-in ROI and normalization buttons are hidden: they operate
    on a time-series (not our single 2-D frames), and dragging the ROI on a large
    live-updating image saturates the GUI thread (the app appears to freeze). The
    camera ROI/subarray is set explicitly in the Camera panel instead. The
    histogram/levels widget on the right is kept.
    """
    imv = pg.ImageView()
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(_CMAP_COLORS)), color=_CMAP_COLORS)
    imv.setColorMap(cmap)
    try:
        imv.ui.roiBtn.hide()
        imv.ui.menuBtn.hide()
    except Exception as e:
        print(f"[widgets] could not hide ImageView ROI/menu buttons: {e}")
    return imv


def show_frame(imv: pg.ImageView, frame: np.ndarray,
               auto_levels: bool = True, auto_range: bool = False,
               level_min=None, level_max=None):
    """
    Display a 2-D frame in an ImageView, transposed like the original code
    (imv.setImage(img.T)). When auto_levels is False and explicit levels are
    given, applies them.
    """
    imv.setImage(frame.T, autoLevels=auto_levels, autoRange=auto_range,
                 levelMode='mono')
    if not auto_levels and level_min is not None and level_max is not None:
        imv.setLevels(min=level_min, max=level_max)
