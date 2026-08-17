"""Table-tennis video intelligence.

The package deliberately keeps research-model integrations optional.  The
core schemas, geometry, event logic, storage, and tests run without model
weights or a GPU.
"""

from .schemas import SCHEMA_VERSION

__version__ = "0.1.0"
__all__ = ["SCHEMA_VERSION", "__version__"]
