"""
EPV-Weighted Influence Calculation Module

Extends the player influence calculation with Expected Possession Value (EPV)
spatial weighting. Instead of uniform integration over the pitch, each grid cell
is weighted by its EPV value, reflecting the spatial value of pitch control.

Key Formula:
    Original: influence = sum(ΔPC)
    EPV-Weighted: influence = sum(EPV × ΔPC)

Which simplifies to:
    influence = sum(EPV × (PC_counterfactual - PC_baseline))

This module provides:
- EPV grid loading and normalization
- Weighted influence computation
- Backward-compatible utilities for comparison and validation
"""

import numpy as np
import os
from pathlib import Path


class EPVWeighting:
    """Handles EPV grid loading, normalization, and weighted influence calculations."""
    
    # Default EPV grid filename
    DEFAULT_EPV_GRID_FILE = 'EPV_grid.csv'
    
    # Expected grid dimensions to match pitch control grid
    EXPECTED_GRID_DIM = (32, 50)  # (ny, nx) = (height, width)
    
    def __init__(self, epv_grid_path=None, normalize=True):
        """
        Initialize EPV weighting.
        
        Parameters
        ----------
        epv_grid_path : str, optional
            Path to EPV grid CSV file. If None, looks for 'EPV_grid.csv' in 
            the project root directory.
        normalize : bool, optional
            If True, normalize EPV grid to [0, 1] range (default: True).
            This prevents dramatic scale changes while preserving relative weights.
        """
        self.epv_grid_path = epv_grid_path
        self.epv_grid = None
        self.epv_grid_normalized = None
        self.is_normalized = False
        
        # Try to load EPV grid
        self._load_epv_grid(normalize=normalize)
    
    def _load_epv_grid(self, normalize=True):
        """
        Load EPV grid from file.
        
        Parameters
        ----------
        normalize : bool
            Whether to normalize to [0, 1] range
        
        Raises
        ------
        FileNotFoundError
            If EPV grid file cannot be found
        ValueError
            If loaded grid has unexpected dimensions
        """
        # Determine path to EPV grid file
        if self.epv_grid_path is None:
            # Try to find EPV_grid.csv in project root
            possible_paths = [
                self.DEFAULT_EPV_GRID_FILE,
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            self.DEFAULT_EPV_GRID_FILE),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.epv_grid_path = path
                    break
        
        if self.epv_grid_path is None:
            raise FileNotFoundError(
                f"Could not find EPV grid file '{self.DEFAULT_EPV_GRID_FILE}'. "
                f"Please specify epv_grid_path or ensure the file exists in the project root."
            )
        
        # Load the grid
        try:
            self.epv_grid = np.loadtxt(self.epv_grid_path, delimiter=',')
        except Exception as e:
            raise FileNotFoundError(f"Failed to load EPV grid from {self.epv_grid_path}: {e}")
        
        # Validate dimensions
        if self.epv_grid.shape != self.EXPECTED_GRID_DIM:
            raise ValueError(
                f"EPV grid has unexpected dimensions {self.epv_grid.shape}. "
                f"Expected {self.EXPECTED_GRID_DIM}."
            )
        
        # Normalize if requested
        if normalize:
            self._normalize_epv_grid()
    
    def _normalize_epv_grid(self):
        """
        Normalize EPV grid to [0, 1] range using min-max scaling.
        
        This prevents the EPV weighting from dramatically changing the scale
        of influence scores while preserving relative spatial weights.
        
        The normalization is: (grid - min) / (max - min)
        """
        epv_min = np.nanmin(self.epv_grid)
        epv_max = np.nanmax(self.epv_grid)
        
        if epv_max == epv_min:
            # All values are the same, set to 0.5 (neutral)
            self.epv_grid_normalized = np.full_like(self.epv_grid, 0.5)
        else:
            self.epv_grid_normalized = (self.epv_grid - epv_min) / (epv_max - epv_min)
        
        self.is_normalized = True
    
    def get_epv_grid(self, normalized=False):
        """
        Get the EPV grid.
        
        Parameters
        ----------
        normalized : bool, optional
            If True, return normalized grid (default: False).
            Normalized grid is recommended for weighting to maintain
            scale consistency.
        
        Returns
        -------
        ndarray
            EPV grid of shape (32, 50)
        """
        if normalized:
            if self.epv_grid_normalized is None and self.is_normalized:
                self._normalize_epv_grid()
            return self.epv_grid_normalized
        else:
            return self.epv_grid
    
    def apply_epv_weighting(self, delta_pc, normalized=True):
        """
        Apply EPV weighting to a pitch control difference surface.
        
        Parameters
        ----------
        delta_pc : ndarray
            Pitch control difference surface of shape (32, 50).
            Typically: PC_counterfactual - PC_baseline
        normalized : bool, optional
            If True, use normalized EPV grid (recommended, default: True).
            Normalized grids maintain consistent scale across applications.
        
        Returns
        -------
        float
            EPV-weighted influence score = sum(EPV × ΔPC)
        
        Raises
        ------
        ValueError
            If delta_pc shape doesn't match EPV grid shape
        """
        if self.epv_grid is None:
            raise RuntimeError("EPV grid not loaded. Call load_epv_grid() first.")
        
        # Validate shapes match
        if delta_pc.shape != self.epv_grid.shape:
            raise ValueError(
                f"delta_pc shape {delta_pc.shape} doesn't match "
                f"EPV grid shape {self.epv_grid.shape}"
            )
        
        # Get EPV grid (normalized or original)
        epv = self.get_epv_grid(normalized=normalized)
        
        # Compute element-wise product and sum
        # This gives us: sum(EPV × ΔPC)
        weighted_influence = np.nansum(epv * delta_pc)
        
        return weighted_influence
    
    def apply_epv_weighting_vectorized(self, delta_pc_dict, normalized=True):
        """
        Apply EPV weighting to multiple pitch control surfaces at once.
        
        Optimized for bulk processing (e.g., all players in a single frame).
        
        Parameters
        ----------
        delta_pc_dict : dict
            Dictionary mapping player_id -> delta_pc (ndarray).
            Each delta_pc should be shape (32, 50).
        normalized : bool, optional
            If True, use normalized EPV grid (default: True).
        
        Returns
        -------
        dict
            Dictionary mapping player_id -> weighted_influence_score (float)
        
        Raises
        ------
        ValueError
            If any delta_pc shape doesn't match EPV grid shape
        """
        if self.epv_grid is None:
            raise RuntimeError("EPV grid not loaded.")
        
        epv = self.get_epv_grid(normalized=normalized)
        weighted_results = {}
        
        for player_id, delta_pc in delta_pc_dict.items():
            if delta_pc.shape != self.epv_grid.shape:
                raise ValueError(
                    f"delta_pc[{player_id}] shape {delta_pc.shape} doesn't match "
                    f"EPV grid shape {self.epv_grid.shape}"
                )
            weighted_results[player_id] = np.nansum(epv * delta_pc)
        
        return weighted_results
    
    @staticmethod
    def compute_influence_metrics(delta_pc, epv_grid=None, normalized=True):
        """
        Compute multiple influence metrics from a pitch control surface.
        
        Returns both original (uniform) and EPV-weighted metrics for comparison.
        
        Parameters
        ----------
        delta_pc : ndarray
            Pitch control difference surface
        epv_grid : ndarray, optional
            EPV grid. If None, only uniform metrics are computed.
        normalized : bool, optional
            If True and epv_grid provided, normalize EPV (default: True).
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'uniform_total': sum(|ΔPC|)
            - 'uniform_positive': sum(max(ΔPC, 0))
            - 'uniform_negative': sum(min(ΔPC, 0))
            - 'uniform_net': sum(ΔPC)
            - 'epv_weighted_total': sum(|EPV × ΔPC|) if epv_grid provided
            - 'epv_weighted_positive': sum(max(EPV × ΔPC, 0)) if epv_grid provided
            - 'epv_weighted_negative': sum(min(EPV × ΔPC, 0)) if epv_grid provided
            - 'epv_weighted_net': sum(EPV × ΔPC) if epv_grid provided
        """
        metrics = {
            'uniform_total': np.nansum(np.abs(delta_pc)),
            'uniform_positive': np.nansum(np.where(delta_pc > 0, delta_pc, 0)),
            'uniform_negative': np.nansum(np.where(delta_pc < 0, delta_pc, 0)),
            'uniform_net': np.nansum(delta_pc),
        }
        
        if epv_grid is not None:
            # Normalize EPV if requested
            if normalized:
                epv_min = np.nanmin(epv_grid)
                epv_max = np.nanmax(epv_grid)
                if epv_max > epv_min:
                    epv_grid_norm = (epv_grid - epv_min) / (epv_max - epv_min)
                else:
                    epv_grid_norm = np.full_like(epv_grid, 0.5)
            else:
                epv_grid_norm = epv_grid
            
            # Compute EPV-weighted metrics
            epv_weighted_pc = epv_grid_norm * delta_pc
            metrics['epv_weighted_total'] = np.nansum(np.abs(epv_weighted_pc))
            metrics['epv_weighted_positive'] = np.nansum(
                np.where(epv_weighted_pc > 0, epv_weighted_pc, 0)
            )
            metrics['epv_weighted_negative'] = np.nansum(
                np.where(epv_weighted_pc < 0, epv_weighted_pc, 0)
            )
            metrics['epv_weighted_net'] = np.nansum(epv_weighted_pc)
        
        return metrics
    
    @staticmethod
    def generate_placeholder_epv_grid(grid_shape=(32, 50), field_dimen=(106, 68)):
        """
        Generate a placeholder EPV grid using distance-to-goal heuristic.
        
        This is a fallback for when no EPV grid file exists.
        Values increase towards the attacking goal based on Euclidean distance.
        
        Parameters
        ----------
        grid_shape : tuple, optional
            Grid shape (ny, nx), default (32, 50)
        field_dimen : tuple, optional
            Pitch dimensions in meters (length, width), default (106, 68)
        
        Returns
        -------
        ndarray
            Placeholder EPV grid of specified shape, values in [0, 1]
        """
        ny, nx = grid_shape
        field_length, field_width = field_dimen
        
        # Create coordinate grids (in field coordinates, origin at center)
        dy = field_width / ny
        dx = field_length / nx
        y_coords = np.arange(ny) * dy - field_width / 2
        x_coords = np.arange(nx) * dx - field_length / 2
        
        # Attacking direction: towards right (positive x)
        goal_x = field_length / 2  # Right goal
        goal_y = 0  # Center line
        
        # Compute distance from each cell to goal
        xx, yy = np.meshgrid(x_coords, y_coords)
        distances = np.sqrt((xx - goal_x) ** 2 + (yy - goal_y) ** 2)
        
        # Convert distance to "goal-proximity" (closer = higher value)
        # Use exponential decay to emphasize near-goal areas
        max_dist = np.sqrt((field_length / 2) ** 2 + (field_width / 2) ** 2)
        epv_placeholder = np.exp(-3 * distances / max_dist)
        
        # Normalize to [0, 1]
        epv_placeholder = (epv_placeholder - epv_placeholder.min()) / \
                         (epv_placeholder.max() - epv_placeholder.min())
        
        return epv_placeholder


def create_epv_weighting_instance(epv_grid_path=None):
    """
    Convenience function to create an EPV weighting instance.
    
    Parameters
    ----------
    epv_grid_path : str, optional
        Path to EPV grid file
    
    Returns
    -------
    EPVWeighting
        Initialized EPV weighting instance
    """
    return EPVWeighting(epv_grid_path=epv_grid_path, normalize=True)
