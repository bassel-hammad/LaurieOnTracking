#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pitch Control Sequence Similarity Analysis

This module provides tools for finding similar possession sequences based on 
pitch control surface evolution using computer vision techniques (SSIM + DTW).

The approach treats pitch control sequences like video clips:
- Each pitch control surface = a frame (50x34 "image")
- SSIM (Structural Similarity Index) = frame-to-frame comparison
- DTW (Dynamic Time Warping) = sequence alignment and matching

Author: Sequence Matching System
Date: November 2025
"""

import numpy as np
import pandas as pd
import json
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim
from typing import List, Tuple, Dict, Optional
import Metrica_PitchControl as mpc


class PitchControlSequence:
    """
    Container for a possession sequence with pitch control data.
    
    Attributes:
    -----------
    sequence_id : int
        PFF sequence identifier
    event_ids : list[int]
        DataFrame indices of events in this sequence
    ppcf_surfaces : list[ndarray]
        Pitch control surfaces (50x34 grids) for attacking team at each event
    event_metadata : list[dict]
        Metadata for each event (type, team, player, time, frame)
    xgrid : ndarray
        X-coordinates of pitch control grid
    ygrid : ndarray
        Y-coordinates of pitch control grid
    duration : float
        Total duration of sequence in seconds
    team : str
        Team in possession ('Home' or 'Away')
    game_id : int
        Game identifier
    """
    
    def __init__(self, sequence_id, event_ids, team, game_id):
        self.sequence_id = sequence_id
        self.event_ids = event_ids
        self.team = team
        self.game_id = game_id
        self.ppcf_surfaces = []
        self.event_metadata = []
        self.xgrid = None
        self.ygrid = None
        self.duration = 0.0
        
    def add_event_pitch_control(self, event_id, ppcf, xgrid, ygrid, metadata):
        """Add pitch control data for one event in the sequence"""
        self.ppcf_surfaces.append(ppcf)
        self.event_metadata.append(metadata)
        if self.xgrid is None:
            self.xgrid = xgrid
            self.ygrid = ygrid
            
    def finalize(self, events_df):
        """Calculate duration and validate sequence"""
        if len(self.event_ids) > 0:
            start_time = events_df.loc[self.event_ids[0]]['Start Time [s]']
            end_time = events_df.loc[self.event_ids[-1]]['Start Time [s]']
            self.duration = end_time - start_time
            
    def __len__(self):
        """Return number of events in sequence"""
        return len(self.event_ids)
    
    def __repr__(self):
        return (f"PitchControlSequence(id={self.sequence_id}, team={self.team}, "
                f"events={len(self.event_ids)}, duration={self.duration:.1f}s)")


def load_pff_sequences(pff_data_dir, game_id):
    """
    Load sequence IDs from raw PFF JSON data.
    
    Parameters:
    -----------
    pff_data_dir : str
        Path to PFF data directory
    game_id : int
        Game identifier
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: gameEventId, sequence, startTime, teamName, period
    """
    events_file = f"{pff_data_dir}/Event Data/{game_id}.json"
    
    with open(events_file, 'r', encoding='utf-8') as f:
        pff_data = json.load(f)
    
    # Extract relevant fields
    sequence_data = []
    for event in pff_data:
        sequence_data.append({
            'gameEventId': event.get('gameEventId'),
            'sequence': event.get('sequence'),
            'startTime': event.get('startTime'),
            'teamName': event['gameEvents'].get('teamName'),
            'period': event['gameEvents'].get('period'),
            'eventType': event['possessionEvents'].get('possessionEventType')
        })
    
    return pd.DataFrame(sequence_data)


def extract_possession_sequences(events_df, pff_sequences_df, min_events=4, min_duration=5.0):
    """
    Extract possession sequences that meet minimum criteria.
    
    Uses the 'sequence' field from PFF data to identify possession chains.
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        Metrica-format events dataframe (from read_event_data)
    pff_sequences_df : pd.DataFrame
        PFF sequence information (from load_pff_sequences)
    min_events : int
        Minimum number of events in a valid sequence (default: 4)
    min_duration : float
        Minimum duration in seconds for a valid sequence (default: 5.0)
        
    Returns:
    --------
    list[dict]
        List of sequence dictionaries with keys: 
        'sequence_id', 'event_indices', 'team', 'duration', 'n_events'
    """
    sequences = []
    
    # Group by sequence ID
    for seq_id, group in pff_sequences_df.groupby('sequence'):
        if pd.isna(seq_id):
            continue
            
        # Only include first 2 periods (no extra time)
        group = group[group['period'] <= 2]
        if len(group) == 0:
            continue
        
        # Get team and time info
        team = group.iloc[0]['teamName']
        start_time = group['startTime'].min()
        end_time = group['startTime'].max()
        duration = end_time - start_time
        n_events = len(group)
        
        # Apply filters: min events OR min duration
        if n_events >= min_events or duration >= min_duration:
            # Find corresponding indices in events_df
            # Match by Start Time since gameEventId might not be in converted data
            event_indices = []
            for _, row in group.iterrows():
                # Find matching event in events_df by time (with tolerance)
                matches = events_df[
                    np.abs(events_df['Start Time [s]'] - row['startTime']) < 0.01
                ]
                if len(matches) > 0:
                    event_indices.append(matches.index[0])
            
            if len(event_indices) >= min_events or duration >= min_duration:
                sequences.append({
                    'sequence_id': int(seq_id),
                    'event_indices': event_indices,
                    'team': team,
                    'duration': duration,
                    'n_events': len(event_indices)
                })
    
    print(f"Extracted {len(sequences)} possession sequences meeting criteria:")
    print(f"  - Min events: {min_events} OR min duration: {min_duration}s")
    print(f"  - Sequences range from {min([s['n_events'] for s in sequences])} to "
          f"{max([s['n_events'] for s in sequences])} events")
    
    return sequences


def calculate_sequence_pitch_control(sequence_dict, events, tracking_home, tracking_away, 
                                    params, GK_numbers, game_id, field_dimen=(106., 68.), 
                                    n_grid_cells_x=50):
    """
    Calculate pitch control surfaces for all events in a sequence.
    
    Parameters:
    -----------
    sequence_dict : dict
        Sequence information from extract_possession_sequences
    events : pd.DataFrame
        Events dataframe
    tracking_home : pd.DataFrame
        Home team tracking data
    tracking_away : pd.DataFrame
        Away team tracking data
    params : dict
        Pitch control model parameters
    GK_numbers : tuple
        (home_GK_number, away_GK_number)
    game_id : int
        Game identifier
    field_dimen : tuple
        (length, width) in meters
    n_grid_cells_x : int
        Grid resolution
        
    Returns:
    --------
    PitchControlSequence or None
        Populated sequence object, or None if insufficient data
    """
    # Determine team from Metrica event data (most reliable)
    first_event = events.loc[sequence_dict['event_indices'][0]]
    metrica_team = first_event['Team']
    
    # Create sequence object
    pc_sequence = PitchControlSequence(
        sequence_id=sequence_dict['sequence_id'],
        event_ids=sequence_dict['event_indices'],
        team=metrica_team,
        game_id=game_id
    )
    
    # Calculate pitch control for each event
    successful_events = 0
    for event_idx in sequence_dict['event_indices']:
        try:
            # Check if tracking data exists for this event
            event = events.loc[event_idx]
            event_frame = int(event['Start Frame'])
            
            if event_frame not in tracking_home.index or event_frame not in tracking_away.index:
                continue
            
            # Generate pitch control surface
            PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
                event_idx, events, tracking_home, tracking_away, params, GK_numbers,
                field_dimen=field_dimen, n_grid_cells_x=n_grid_cells_x
            )
            
            # Store metadata
            metadata = {
                'event_idx': event_idx,
                'type': event['Type'],
                'subtype': event.get('Subtype', ''),
                'player': event.get('From', 'Unknown'),
                'time': event['Start Time [s]'],
                'frame': event_frame,
                'period': event['Period']
            }
            
            pc_sequence.add_event_pitch_control(event_idx, PPCF, xgrid, ygrid, metadata)
            successful_events += 1
            
        except Exception as e:
            print(f"  Warning: Could not calculate pitch control for event {event_idx}: {e}")
            continue
    
    # Finalize sequence
    if successful_events >= 3:  # Need at least 3 events for meaningful comparison
        pc_sequence.finalize(events)
        return pc_sequence
    else:
        return None


def compute_frame_distance(ppcf1, ppcf2, method='ssim'):
    """
    Calculate distance between two pitch control surfaces.
    
    Uses SSIM (Structural Similarity Index) by default - treats pitch control
    surfaces like images and measures perceptual similarity.
    
    Parameters:
    -----------
    ppcf1 : ndarray
        First pitch control surface (shape: n_y x n_x)
    ppcf2 : ndarray
        Second pitch control surface (same shape as ppcf1)
    method : str
        Distance method: 'ssim' (default) or 'mse'
        
    Returns:
    --------
    float
        Distance value (0 = identical, higher = more different)
        For SSIM: range [0, 2] (since SSIM in [-1, 1], distance = 1 - SSIM)
        For MSE: range [0, ∞)
    """
    if ppcf1.shape != ppcf2.shape:
        raise ValueError(f"Shape mismatch: {ppcf1.shape} vs {ppcf2.shape}")
    
    if method == 'ssim':
        # SSIM returns similarity in range [-1, 1], where 1 = identical
        # Convert to distance: distance = 1 - similarity
        # data_range=1.0 because PPCF values are probabilities in [0, 1]
        similarity = ssim(ppcf1, ppcf2, data_range=1.0)
        distance = 1.0 - similarity
        return distance
    
    elif method == 'mse':
        # Mean Squared Error
        mse = np.mean((ppcf1 - ppcf2) ** 2)
        return mse
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ssim' or 'mse'")


def dtw_distance(seq1_frames, seq2_frames, distance_func=compute_frame_distance):
    """
    Compute Dynamic Time Warping distance between two sequences.
    
    DTW finds the optimal alignment between two sequences of different lengths
    by minimizing the cumulative distance between aligned frames.
    
    Parameters:
    -----------
    seq1_frames : list[ndarray]
        First sequence of pitch control surfaces
    seq2_frames : list[ndarray]
        Second sequence of pitch control surfaces
    distance_func : callable
        Function to compute distance between two frames
        
    Returns:
    --------
    tuple: (distance, path)
        distance : float
            DTW distance (normalized by path length)
        path : list[tuple]
            Optimal alignment path as list of (i, j) index pairs
    """
    n, m = len(seq1_frames), len(seq2_frames)
    
    # Initialize DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = distance_func(seq1_frames[i-1], seq2_frames[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    # Backtrack to find optimal path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        
        # Find which direction we came from
        candidates = [
            (dtw_matrix[i-1, j-1], (i-1, j-1)),  # diagonal
            (dtw_matrix[i-1, j], (i-1, j)),      # up
            (dtw_matrix[i, j-1], (i, j-1))       # left
        ]
        _, (i, j) = min(candidates, key=lambda x: x[0])
    
    path.reverse()
    
    # Normalize distance by path length to compare sequences of different lengths
    normalized_distance = dtw_matrix[n, m] / len(path) if len(path) > 0 else dtw_matrix[n, m]
    
    return normalized_distance, path


def compare_sequences_dtw(seq1, seq2, distance_method='ssim'):
    """
    Compare two PitchControlSequence objects using DTW.
    
    Parameters:
    -----------
    seq1 : PitchControlSequence
        First sequence
    seq2 : PitchControlSequence
        Second sequence
    distance_method : str
        Method for frame distance ('ssim' or 'mse')
        
    Returns:
    --------
    tuple: (distance, path, alignment_quality)
        distance : float
            DTW distance between sequences
        path : list[tuple]
            Optimal alignment path
        alignment_quality : dict
            Statistics about the alignment
    """
    # Create distance function with specified method
    def frame_dist(f1, f2):
        return compute_frame_distance(f1, f2, method=distance_method)
    
    # Compute DTW
    distance, path = dtw_distance(seq1.ppcf_surfaces, seq2.ppcf_surfaces, frame_dist)
    
    # Calculate alignment quality metrics
    alignment_quality = {
        'path_length': len(path),
        'seq1_length': len(seq1),
        'seq2_length': len(seq2),
        'compression_ratio': len(path) / max(len(seq1), len(seq2)),
        'avg_frame_distance': distance
    }
    
    return distance, path, alignment_quality


def find_similar_sequences(query_sequence, sequence_database, top_k=5, 
                          distance_method='ssim', filter_team=None, 
                          filter_game=None):
    """
    Find the most similar sequences to a query sequence.
    
    Parameters:
    -----------
    query_sequence : PitchControlSequence
        Query sequence to match
    sequence_database : list[PitchControlSequence]
        Database of sequences to search
    top_k : int
        Number of top matches to return
    distance_method : str
        Method for frame distance ('ssim' or 'mse')
    filter_team : str or None
        Filter by team ('Home', 'Away', or None for no filter)
    filter_game : int or None
        Filter by game_id (or None for no filter)
        
    Returns:
    --------
    list[dict]
        Top K similar sequences with keys:
        'sequence', 'distance', 'path', 'alignment_quality'
    """
    results = []
    
    # Filter database
    filtered_db = sequence_database
    if filter_team:
        filtered_db = [s for s in filtered_db if s.team == filter_team]
    if filter_game:
        filtered_db = [s for s in filtered_db if s.game_id == filter_game]
    
    # Exclude the query itself if it's in the database
    filtered_db = [s for s in filtered_db if s.sequence_id != query_sequence.sequence_id]
    
    print(f"Searching {len(filtered_db)} sequences for matches to sequence {query_sequence.sequence_id}...")
    
    # Compare query to each sequence in database
    for seq in filtered_db:
        try:
            distance, path, quality = compare_sequences_dtw(
                query_sequence, seq, distance_method=distance_method
            )
            
            results.append({
                'sequence': seq,
                'distance': distance,
                'path': path,
                'alignment_quality': quality
            })
        except Exception as e:
            print(f"  Warning: Could not compare with sequence {seq.sequence_id}: {e}")
            continue
    
    # Sort by distance (lower = more similar)
    results.sort(key=lambda x: x['distance'])
    
    # Return top K
    return results[:top_k]


def build_sequence_database(events, tracking_home, tracking_away, params, GK_numbers,
                           game_id, pff_data_dir, min_events=4, min_duration=5.0,
                           field_dimen=(106., 68.), n_grid_cells_x=50):
    """
    Build a database of all possession sequences with pitch control data.
    
    This is a convenience function that:
    1. Extracts possession sequences from PFF data
    2. Calculates pitch control for each sequence
    3. Returns a list of PitchControlSequence objects
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events dataframe
    tracking_home : pd.DataFrame
        Home team tracking data
    tracking_away : pd.DataFrame
        Away team tracking data
    params : dict
        Pitch control model parameters
    GK_numbers : tuple
        (home_GK_number, away_GK_number)
    game_id : int
        Game identifier
    pff_data_dir : str
        Path to PFF data directory
    min_events : int
        Minimum events per sequence
    min_duration : float
        Minimum duration per sequence
    field_dimen : tuple
        Field dimensions
    n_grid_cells_x : int
        Grid resolution
        
    Returns:
    --------
    list[PitchControlSequence]
        Database of sequences with pitch control data
    """
    print("=" * 70)
    print("BUILDING PITCH CONTROL SEQUENCE DATABASE")
    print("=" * 70)
    print()
    
    # Load PFF sequences
    print("Loading PFF sequence data...")
    pff_sequences = load_pff_sequences(pff_data_dir, game_id)
    print(f"Loaded {len(pff_sequences)} events with sequence IDs")
    print()
    
    # Extract possession sequences
    print("Extracting possession sequences...")
    sequences = extract_possession_sequences(
        events, pff_sequences, min_events=min_events, min_duration=min_duration
    )
    print()
    
    # Calculate pitch control for each sequence
    print("Calculating pitch control for sequences...")
    database = []
    for i, seq_dict in enumerate(sequences):
        print(f"Processing sequence {i+1}/{len(sequences)} "
              f"(ID: {seq_dict['sequence_id']}, {seq_dict['n_events']} events)...")
        
        pc_seq = calculate_sequence_pitch_control(
            seq_dict, events, tracking_home, tracking_away, params, GK_numbers,
            game_id, field_dimen=field_dimen, n_grid_cells_x=n_grid_cells_x
        )
        
        if pc_seq is not None:
            database.append(pc_seq)
            print(f"  ✓ Successfully processed {len(pc_seq)} events")
        else:
            print(f"  ✗ Insufficient tracking data")
    
    print()
    print("=" * 70)
    print(f"DATABASE COMPLETE: {len(database)} sequences with pitch control data")
    print("=" * 70)
    print()
    
    return database
