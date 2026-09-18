"""Firebrand detection and tracking for single-view security footage.

Pipeline:  frames -> residual -> streak candidates -> tracks -> accepted tracks
The acceptance step is the detector; see track.py.
"""
from .config import Config, DetectConfig, TrackConfig, SynthConfig, TrainConfig  # noqa: F401
from . import frames, detect, track, synth, evaluate, gcsio  # noqa: F401

__version__ = "0.1.0"
