"""Chorus: batched learning of many sparse additive models."""
from .model import Chorus, lambda_factors
from .design import InMemoryDesign, DiskDesign, write_disk_design
from .basis import SplineDesign
