"""
Data loading and preprocessing module.

Handles loading tracking data, event data, and preprocessing for analysis.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import Metrica_IO as mio
import Metrica_Velocities as mvel

from .config import Config


class DataLoader:
    """Handles loading and preprocessing of tracking and event data."""
    
    def __init__(self, game_id, data_dir=None):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        game_id : int
            The match ID to load
        data_dir : str, optional
            Directory containing the data (default: Config.DATA_DIR)
        """
        self.game_id = game_id
        self.data_dir = data_dir or Config.DATA_DIR
        
        self.events = None
        self.tracking_home = None
        self.tracking_away = None
        self.player_mapping = None
        self.gk_numbers = None
        
    def load_all(self, verbose=True):
        """
        Load all data for the match.
        
        Parameters
        ----------
        verbose : bool
            Print progress messages
            
        Returns
        -------
        tuple
            (events, tracking_home, tracking_away)
        """
        if verbose:
            print("Loading data...")
        
        # Load event data
        self.events = mio.read_event_data(self.data_dir, self.game_id)
        
        # Load tracking data
        self.tracking_home = mio.tracking_data(self.data_dir, self.game_id, 'Home')
        self.tracking_away = mio.tracking_data(self.data_dir, self.game_id, 'Away')
        
        # Convert to metric coordinates
        self.tracking_home = mio.to_metric_coordinates(self.tracking_home)
        self.tracking_away = mio.to_metric_coordinates(self.tracking_away)
        self.events = mio.to_metric_coordinates(self.events)
        
        # Normalize playing direction
        self.tracking_home, self.tracking_away, self.events = mio.to_single_playing_direction(
            self.tracking_home, self.tracking_away, self.events
        )
        
        if verbose:
            print(f"Data loaded: {len(self.events)} events, {len(self.tracking_home):,} tracking frames")
        
        # Calculate velocities
        self._calculate_velocities(verbose)
        
        # Find goalkeepers
        self.gk_numbers = [
            mio.find_goalkeeper(self.tracking_home),
            mio.find_goalkeeper(self.tracking_away)
        ]
        
        if verbose:
            print(f"Goalkeepers: Home #{self.gk_numbers[0]}, Away #{self.gk_numbers[1]}")
        
        # Try to load player mapping
        try:
            self.player_mapping = mio.load_player_mapping(self.data_dir, self.game_id)
        except:
            self.player_mapping = None
        
        return self.events, self.tracking_home, self.tracking_away
    
    def _calculate_velocities(self, verbose=True):
        """Calculate player velocities."""
        pff_speed_cols = [c for c in self.tracking_home.columns if c.endswith('_pff_speed')]
        
        if pff_speed_cols:
            if verbose:
                print("Calculating player velocities using HYBRID method (PFF speed + calculated direction)...")
            self.tracking_home = mvel.calc_player_velocities_hybrid(
                self.tracking_home, smoothing=True, use_pff_speed=True
            )
            self.tracking_away = mvel.calc_player_velocities_hybrid(
                self.tracking_away, smoothing=True, use_pff_speed=True
            )
            if verbose:
                print("  -> Using PFF's raw speed values for more accurate velocities")
        else:
            if verbose:
                print("Calculating player velocities from position differences...")
            self.tracking_home = mvel.calc_player_velocities(self.tracking_home, smoothing=True)
            self.tracking_away = mvel.calc_player_velocities(self.tracking_away, smoothing=True)
    
    def get_sequence_events(self, sequence_number):
        """
        Get events for a specific sequence.
        
        Parameters
        ----------
        sequence_number : float
            The sequence number to filter
            
        Returns
        -------
        DataFrame
            Events in the specified sequence
        """
        return self.events[self.events['Sequence'] == sequence_number].copy()
    
    def get_available_sequences(self):
        """
        Get list of available sequence numbers.
        
        Returns
        -------
        list
            Sorted list of available sequence numbers
        """
        available = self.events['Sequence'].dropna().unique()
        return sorted([int(s) for s in available if not pd.isna(s)])
    
    def get_sequence_frame_range(self, sequence_events):
        """
        Get frame range for a sequence.
        
        Parameters
        ----------
        sequence_events : DataFrame
            Events in the sequence
            
        Returns
        -------
        tuple
            (start_frame, end_frame, start_time, end_time)
        """
        start_frame = int(sequence_events['Start Frame'].min())
        end_frame = int(sequence_events['End Frame'].max())
        
        if start_frame in self.tracking_home.index and end_frame in self.tracking_home.index:
            # Use iloc[0] to ensure we get a scalar value (handles duplicate indices)
            start_time_val = self.tracking_home.loc[start_frame, 'Time [s]']
            end_time_val = self.tracking_home.loc[end_frame, 'Time [s]']
            # Convert to float if it's a Series
            start_time = float(start_time_val.iloc[0]) if hasattr(start_time_val, 'iloc') else float(start_time_val)
            end_time = float(end_time_val.iloc[0]) if hasattr(end_time_val, 'iloc') else float(end_time_val)
        else:
            start_time = float(sequence_events['Start Time [s]'].min())
            end_time = float(sequence_events['End Time [s]'].max())
            print("Warning: Using time-based lookup (frames not found in tracking data)")
        
        return start_frame, end_frame, start_time, end_time
    
    def get_event_frames(self, sequence_events):
        """
        Get unique event frames from sequence events.
        
        Parameters
        ----------
        sequence_events : DataFrame
            Events in the sequence
            
        Returns
        -------
        list
            Sorted list of unique event frame numbers
        """
        event_frames = pd.concat([
            sequence_events['Start Frame'],
            sequence_events['End Frame']
        ]).dropna().unique()
        
        return sorted([int(f) for f in event_frames])
    
    def get_frames_for_analysis(self, event_frames):
        """
        Filter event frames to only include those in tracking data.
        
        Parameters
        ----------
        event_frames : list
            List of event frame numbers
            
        Returns
        -------
        list
            Frames that exist in both home and away tracking data
        """
        valid_frames = []
        for frame in event_frames:
            if frame in self.tracking_home.index and frame in self.tracking_away.index:
                valid_frames.append(frame)
        return valid_frames
    
    def get_movie_frames(self, start_time, end_time, fps=None):
        """
        Get frames for movie generation at specified FPS.
        
        Parameters
        ----------
        start_time : float
            Start time in seconds
        end_time : float
            End time in seconds
        fps : int, optional
            Target frames per second (default: Config.TARGET_FPS)
            
        Returns
        -------
        list
            Frame numbers for movie
        """
        fps = fps or Config.TARGET_FPS
        time_interval = 1.0 / fps
        sample_times = np.arange(start_time, end_time + time_interval/2, time_interval)
        
        frame_times = self.tracking_home['Time [s]'].values
        movie_frames = []
        last_frame = None
        
        for sample_time in sample_times:
            time_diffs = np.abs(frame_times - sample_time)
            closest_idx = np.argmin(time_diffs)
            frame = self.tracking_home.index[closest_idx]
            
            if frame in self.tracking_away.index and frame != last_frame:
                movie_frames.append(frame)
                last_frame = frame
        
        return movie_frames
    
    def get_player_display_name(self, player_id):
        """
        Get display name for a player.
        
        Parameters
        ----------
        player_id : str
            Player identifier (e.g., 'Home_10')
            
        Returns
        -------
        str
            Display name (e.g., '#10')
        """
        if '_' in player_id:
            team, jersey = player_id.split('_', 1)
            
            # Try to get name from mapping
            if self.player_mapping:
                try:
                    name = mio.get_player_name(self.player_mapping, team, jersey)
                    if name and not name.startswith('Player'):
                        return f"#{jersey}"
                except:
                    pass
            
            return f"#{jersey}"
        return player_id
