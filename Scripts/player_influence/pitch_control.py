"""
Pitch Control calculation module.

Provides functions for calculating pitch control surfaces.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import Metrica_PitchControl as mpc

from .config import Config


class PitchControlCalculator:
    """Handles pitch control calculations."""
    
    def __init__(self, gk_numbers, params=None):
        """
        Initialize the pitch control calculator.
        
        Parameters
        ----------
        gk_numbers : list
            Goalkeeper jersey numbers [home_GK, away_GK]
        params : dict, optional
            Pitch control model parameters (default: mpc.default_model_params())
        """
        self.gk_numbers = gk_numbers
        self.params = params or mpc.default_model_params()
        
        # Pre-calculate grid
        self._setup_grid()
    
    def _setup_grid(self):
        """Set up the pitch control grid."""
        field_dimen = Config.FIELD_DIMEN
        n_grid_cells_x = Config.N_GRID_CELLS_X
        n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
        
        dx = field_dimen[0] / n_grid_cells_x
        dy = field_dimen[1] / n_grid_cells_y
        
        self.xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0]/2. + dx/2.
        self.ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1]/2. + dy/2.
        self.grid_shape = (len(self.ygrid), len(self.xgrid))
    
    def calculate_surface(self, home_row, away_row, attacking_half_only=True, attacking_team='Home'):
        """
        Calculate pitch control surface for given player positions.
        
        Parameters
        ----------
        home_row : Series
            Home team player position data
        away_row : Series
            Away team player position data
        attacking_half_only : bool
            If True, only calculate PC in the attacking half
        attacking_team : str
            'Home' or 'Away' - determines which half is the attacking half
            
        Returns
        -------
        tuple
            (PPCF_attacking, xgrid, ygrid)
        """
        ball_pos = np.array([home_row['ball_x'], home_row['ball_y']])
        
        PPCFa = np.zeros(shape=self.grid_shape)
        PPCFd = np.zeros(shape=self.grid_shape)
        
        # Initialize players
        attacking_players = mpc.initialise_players(
            home_row, 'Home', self.params, self.gk_numbers[0], is_attacking=True
        )
        defending_players = mpc.initialise_players(
            away_row, 'Away', self.params, self.gk_numbers[1], is_attacking=False
        )
        
        # Calculate pitch control at each grid location
        for ii in range(len(self.ygrid)):
            for jj in range(len(self.xgrid)):
                target_position = np.array([self.xgrid[jj], self.ygrid[ii]])
                
                # Only calculate in the attacking half
                if attacking_half_only:
                    if attacking_team == 'Home' and target_position[0] <= 0:
                        PPCFa[ii, jj] = np.nan
                        PPCFd[ii, jj] = np.nan
                        continue
                    elif attacking_team == 'Away' and target_position[0] >= 0:
                        PPCFa[ii, jj] = np.nan
                        PPCFd[ii, jj] = np.nan
                        continue
                
                PPCFa[ii, jj], PPCFd[ii, jj] = mpc.calculate_pitch_control_at_target(
                    target_position, attacking_players, defending_players, ball_pos, self.params
                )
        
        return PPCFa, self.xgrid, self.ygrid
    
    def calculate_surface_with_hybrid(self, attack_row, defend_row, attacking_team='Home',
                                       attacking_half_only=True):
        """
        Calculate pitch control with specified attacking/defending rows.
        
        Parameters
        ----------
        attack_row : Series
            Attacking team player position data
        defend_row : Series
            Defending team player position data
        attacking_team : str
            'Home' or 'Away'
        attacking_half_only : bool
            If True, only calculate in attacking half
            
        Returns
        -------
        tuple
            (PPCF, xgrid, ygrid)
        """
        if attacking_team == 'Home':
            return self.calculate_surface(attack_row, defend_row, attacking_half_only, attacking_team)
        else:
            # For Away team, swap the order
            PPCF, xgrid, ygrid = self.calculate_surface(defend_row, attack_row, attacking_half_only, attacking_team)
            # Invert for away team perspective
            return 1 - PPCF, xgrid, ygrid
    
    @staticmethod
    def get_backfilled_row(tracking_data, frame):
        """
        Get row with backfilled velocities for pitch control calculation.
        
        Parameters
        ----------
        tracking_data : DataFrame
            Tracking data
        frame : int
            Frame number
            
        Returns
        -------
        Series
            Row with backfilled velocity data
        """
        return mpc._row_with_backfilled_velocities(tracking_data, frame)
