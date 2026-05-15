"""Local Lynn Engine Triton kernels.

This package marker prevents an installed third-party ``triton_kernels``
package from shadowing the repository-local kernels when benchmarks are run
from conda environments that already ship a package with the same name.
"""
