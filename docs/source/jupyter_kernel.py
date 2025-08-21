"""
Jupyter Kernel Sphinx Extension
===============================

This Sphinx extension creates and manages Jupyter kernels for documentation builds.
It automatically sets up a kernel at the beginning of the build process and tears it down at the end.

Installation
-----------
1. Place this file in your Sphinx project directory or extension path
2. Add 'jupyter_kernel_sphinx' to your extensions list in conf.py:

   extensions = [
       'jupyter_kernel', # must be placed before myst_nb
       ...,
       'myst_nb'
   ]

Configuration
------------
In your conf.py file, configure:

- kernel_name: The kernel name used in the documentation notebook.
  This MUST be set to the kernel name specified in your notebook files.

  # Example: Use a specific kernel name
  kernel_name = "portfolio_plan"

Requirements
-----------
- ipykernel
- jupyter_core
- sphinx
- myst-nb

"""

import logging
import os
import re
import shutil
from uuid import uuid4

from ipykernel.kernelspec import install
from jupyter_core.paths import jupyter_data_dir
from sphinx.application import Sphinx

logger = logging.getLogger(__name__)


def jupyter_kernel(name: str):
    jupyter_path = jupyter_data_dir()
    kernels_path = os.path.join(jupyter_path, "kernels")
    kernel_path = os.path.join(kernels_path, name)
    return kernel_path


def jupyter_kernel_is_installed(name: str) -> bool:
    kernel_path = jupyter_kernel(name)
    return os.path.exists(kernel_path)


def setup_jupyter_kernel(name: str):
    if jupyter_kernel_is_installed(name):
        raise RuntimeError(f"ipykernel with name {name} exists. Interrupting")

    install(
        user=True,
        kernel_name=name,
        display_name=name,
    )


def teardown_jupyter_kernel(name: str):
    kernel_path = jupyter_kernel(name)
    if os.path.islink(kernel_path):
        os.remove(kernel_path)
        return
    if os.path.exists(kernel_path):
        shutil.rmtree(kernel_path)
        return
    raise RuntimeError(f"kernel {name} not found at {kernel_path}. Failed to delete")


def setup(app: Sphinx):
    app.add_config_value("kernel_name", None, "env")

    def on_builder_init(app):
        kernel_name = app.config.kernel_name
        if kernel_name is None:
            raise ValueError("kernel_name must be set in conf.py")

        temp_kernel_name = str(uuid4())

        app.env.jupyter_kernel_name = temp_kernel_name

        setup_jupyter_kernel(temp_kernel_name)

        escaped_kernel_name = re.escape(kernel_name)

        try:
            app.config.nb_kernel_rgx_aliases = {
                f"^{escaped_kernel_name}$": temp_kernel_name
            }
            logger.info(f"Set up kernel alias: '{kernel_name}' -> '{temp_kernel_name}'")
        except AttributeError:
            logger.warning("Could not set nb_kernel_rgx_aliases. Is myst-nb installed?")

    def on_build_finished(app, exception):
        kernel_name = getattr(app.env, "jupyter_kernel_name", None)
        if kernel_name:
            try:
                teardown_jupyter_kernel(kernel_name)
                logger.info(f"Removed temporary kernel: {kernel_name}")
            except Exception as e:
                logger.error(f"Failed to remove kernel {kernel_name}: {e}")

    app.connect("builder-inited", on_builder_init)
    app.connect("build-finished", on_build_finished)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
