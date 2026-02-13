"""
Copyright 2025, the SparseDiffPy developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import builtins
import glob
import platform

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def not_on_windows(s: str) -> str:
    return s if platform.system().lower() != "windows" else ""


class build_ext_numpy(build_ext):
    """Custom build_ext that injects NumPy headers."""

    def finalize_options(self) -> None:
        build_ext.finalize_options(self)
        builtins.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


# Collect all C source files from SparseDiffEngine
diff_engine_sources = [
    s for s in glob.glob("SparseDiffEngine/src/**/*.c", recursive=True)
    if "dnlp_diff_engine" not in s
] + ["sparsediffpy/_bindings/bindings.c"]

# Define _POSIX_C_SOURCE on Linux for clock_gettime and struct timespec
defines = []
if platform.system().lower() == "linux":
    defines.append(("_POSIX_C_SOURCE", "200809L"))

sparsediffengine = Extension(
    "sparsediffpy._sparsediffengine",
    sources=diff_engine_sources,
    include_dirs=[
        "SparseDiffEngine/include/",
        "SparseDiffEngine/src/",
        "sparsediffpy/_bindings/",
    ],
    define_macros=defines,
    extra_compile_args=[
        #"-O3",
        #"-std=c99",
        #"-Wall",
        #not_on_windows("-Wextra"),
        #'-DDIFF_ENGINE_VERSION="0.1.0"',
    ],
    extra_link_args=["-lm"] if platform.system().lower() != "windows" else [],
)

setup(
    cmdclass={"build_ext": build_ext_numpy},
    ext_modules=[sparsediffengine],
)
