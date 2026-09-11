# Vendored wheels

## `smaract_ctl-1.5.3-py3-none-any.whl` — SmarAct MCS2 Python API

The TWINS wedge stage ([instruments/twins_stage.py](../instruments/twins_stage.py))
talks to the SmarAct MCS2 controller through this package (`import smaract.ctl`).

Install it (in your activated environment):

```
pip install wheels/smaract_ctl-1.5.3-py3-none-any.whl
```

It's a pure-Python `py3-none-any` wheel, so this works on any OS and any Python 3
with no compilation.

### Why a wheel instead of the SmarAct SDK `.zip`?

SmarAct ships the package as a source `.zip` and the manual tells you to run
`pip install smaract.ctl-1.5.3.zip`. That **fails on modern setuptools**: the
zip's `setup.py` does `from setuptools.command.upload import upload` to build a
no-op "you may not upload this" stub, but the `upload` command was removed from
setuptools (deprecated in 40.0, removed later). Build then dies with:

```
ModuleNotFoundError: No module named 'setuptools.command.upload'
```

The manual (Copyright 2022) predates that setuptools change. This wheel was built
from SmarAct's own `smaract.ctl-1.5.3.zip` with only that dead upload stub removed
from `setup.py` — the installed `smaract.ctl` module is identical. Shipping the
wheel means `pip install` just works everywhere, with no build and no setuptools
version to pin.

### Important: the DLL is not in the wheel

The wheel is the Python binding only. `SmarActCTL.dll` (and the rest of the MCS2
runtime) is installed by the **SmarAct MCS2 software installer**, which must be run
on any machine that actually drives the stage. Without the controller present the
driver falls back cleanly (see `TwinsStage.connect` / `--mode mock`).
