"""
Player Influence Analysis Package

This package provides tools for analyzing individual player contributions 
to pitch control changes during football sequences.

Modules:
- config: Configuration constants and parameters
- data_loader: Data loading and preprocessing utilities
- pitch_control: Pitch control calculation functions
- influence_calculator: Player influence analysis logic
- visualization: Plotting and animation utilities
- main: Main entry point for running the analysis
"""

from .config import Config
from .data_loader import DataLoader
from .pitch_control import PitchControlCalculator
from .influence_calculator import InfluenceCalculator
from .visualization import Visualizer

__version__ = "1.0.0"
__author__ = "LaurieOnTracking"
