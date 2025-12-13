"""
Configuration module for Player Influence Analysis.

Contains all constants, default parameters, and configuration settings.
"""

import os

# Get project root directory (two levels up from this file)
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_PACKAGE_DIR)
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)


class Config:
    """Configuration settings for the player influence analysis."""
    
    # Directory settings (absolute paths)
    PROJECT_ROOT = _PROJECT_ROOT
    DATA_DIR = os.path.join(_PROJECT_ROOT, 'Sample Data')
    OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'Metrica_Output')
    
    # Analysis parameters
    TARGET_FPS = 5  # Frames per second for movie
    FRAME_TOLERANCE = 5  # Tolerance for matching event frames
    
    # Pitch dimensions (in meters)
    FIELD_LENGTH = 106.0
    FIELD_WIDTH = 68.0
    FIELD_DIMEN = (FIELD_LENGTH, FIELD_WIDTH)
    
    # Grid settings for pitch control calculation
    N_GRID_CELLS_X = 50
    
    # Visualization settings
    FIGURE_SIZE = (18, 10)
    DPI = 150
    BITRATE = 5000
    
    # Colors
    HOME_COLOR_TOP5 = "#FF0000"  # Crimson for top 5 home players
    HOME_COLOR_OTHERS = "#F14B4B"  # Light red for other home players
    AWAY_COLOR_TOP5 = "#0D00FF"  # Dodger blue for top 5 away players
    AWAY_COLOR_OTHERS = "#596EFA"  # Light blue for other away players
    
    # Marker sizes
    PLAYER_MARKER_SIZE = 12
    TOP5_MARKER_SIZE = 14
    BALL_MARKER_SIZE = 8
    
    @classmethod
    def get_output_path(cls, filename):
        """Get full output path for a file."""
        if not os.path.exists(cls.OUTPUT_DIR):
            os.makedirs(cls.OUTPUT_DIR)
        return os.path.join(cls.OUTPUT_DIR, filename)
    
    @classmethod
    def get_movie_filename(cls, sequence_number):
        """Generate movie filename for a sequence."""
        return f'sequence_{int(sequence_number)}_player_influence.mp4'
