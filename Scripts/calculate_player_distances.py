#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate total distance covered by each player in a match

Based on the approach from Tutorial2_DelvingDeeper.py
This script calculates distance by summing player speeds over time.

@author: Generated Script
"""

import sys
import os
# Add parent directory to path to import Metrica modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Metrica_IO as mio
import Metrica_Velocities as mvel
import numpy as np
import pandas as pd
import json


def load_player_name_to_jersey_mapping(game_id, data_dir='PFF Data'):
    """
    Load player name to jersey number mapping from PFF tracking data
    
    Parameters:
    -----------
    game_id : int
        The game ID
    data_dir : str
        Path to PFF data directory
        
    Returns:
    --------
    dict : Mapping of player names to jersey numbers for both teams
    """
    tracking_file = f"{data_dir}/Tracking Data/{game_id}.jsonl"
    name_to_jersey = {'Home': {}, 'Away': {}}
    
    try:
        import unicodedata
        
        def normalize_name(name):
            """Normalize name for matching - remove accents, lowercase, strip"""
            # Normalize unicode
            normalized = unicodedata.normalize('NFKD', name)
            # Remove accents
            ascii_name = ''.join(c for c in normalized if not unicodedata.combining(c))
            return ascii_name.lower().strip()
        
        # First pass: collect all name->jersey mappings from game_event
        jersey_map = {'Home': {}, 'Away': {}}
        with open(tracking_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    frame = json.loads(line.strip())
                    
                    if 'game_event' in frame and frame['game_event']:
                        ge = frame['game_event']
                        shirt_number = ge.get('shirt_number')
                        player_name = ge.get('player_name')
                        is_home = ge.get('home_team', 0)
                        
                        if shirt_number and player_name:
                            team = 'Home' if is_home else 'Away'
                            normalized = normalize_name(player_name)
                            if normalized not in jersey_map[team]:
                                jersey_map[team][normalized] = str(shirt_number)
                                jersey_map[team][player_name] = str(shirt_number)  # Also store original
                                
                except json.JSONDecodeError:
                    continue
        
        print(f"Found {len(jersey_map['Home'])} home player mappings, {len(jersey_map['Away'])} away player mappings")
        
        # Now map these to the actual names used in tracking data columns
        # Return the jersey_map which will be used with normalized matching
        name_to_jersey = jersey_map
                    
    except Exception as e:
        print(f"Warning: Could not load jersey mappings: {e}")
        import traceback
        traceback.print_exc()
        
    return name_to_jersey


def calculate_player_distances(tracking_data, team_name='Home', name_to_jersey=None):
    """
    Calculate total distance covered by each player
    
    Parameters:
    -----------
    tracking_data : DataFrame
        Tracking data for the team
    team_name : str
        Name of team ('Home' or 'Away')
    name_to_jersey : dict, optional
        Mapping of player names to jersey numbers
    
    Returns:
    --------
    summary : DataFrame
        Summary with player distances and minutes played
    """
    # Restrict calculations to regulation time only.
    # The source files include extra-time periods, but we only want first and second half here.
    tracking_data = tracking_data[tracking_data['Period'] <= 2].copy()

    # Get list of unique players
    players = np.unique([c.split('_')[1] for c in tracking_data.columns if c[:len(team_name)] == team_name])
    
    # Create summary dataframe
    summary = pd.DataFrame(index=players)
    
    # Use the actual time deltas in the tracking file instead of assuming a fixed frame rate.
    # This keeps the distance estimate correct when the data is irregularly sampled or has gaps.
    time_deltas = tracking_data['Time [s]'].diff().fillna(0)

    # Calculate minutes played for each player
    print(f"\nCalculating statistics for {team_name} team...")
    player_ids = []
    minutes = []
    distance = []
    
    for player in players:
        # Determine which raw source columns actually exist for this player.
        # Minutes and distance should only be counted when the player has real tracking data.
        source_columns = [
            f'{team_name}_{player}_x',
            f'{team_name}_{player}_y',
            f'{team_name}_{player}_visibility',
            f'{team_name}_{player}_pff_speed',
        ]
        active_mask = pd.Series(False, index=tracking_data.index)
        for source_column in source_columns:
            if source_column in tracking_data.columns:
                active_mask = active_mask | tracking_data[source_column].notna()

        # Get jersey number if mapping is available, otherwise use player identifier
        if name_to_jersey and team_name in name_to_jersey:
            jersey_num = None
            
            # Try exact match first
            if player in name_to_jersey[team_name]:
                jersey_num = name_to_jersey[team_name][player]
            else:
                # Try normalized match
                import unicodedata
                import re
                
                def normalize_name(name):
                    # Fix common encoding issues
                    try:
                        # Try to fix double-encoding
                        fixed = name.encode('latin-1').decode('utf-8')
                    except:
                        fixed = name
                    # Normalize unicode
                    normalized = unicodedata.normalize('NFKD', fixed)
                    # Remove accents and soft hyphens
                    ascii_name = ''.join(c for c in normalized if not unicodedata.combining(c))
                    ascii_name = ascii_name.replace('\xad', '')  # Remove soft hyphen
                    return ascii_name.lower().strip()
                
                normalized_player = normalize_name(player)
                if normalized_player in name_to_jersey[team_name]:
                    jersey_num = name_to_jersey[team_name][normalized_player]
                else:
                    # Try fuzzy match on last name
                    player_last = normalized_player.split()[-1] if ' ' in normalized_player else normalized_player
                    for mapped_name in name_to_jersey[team_name]:
                        mapped_last = mapped_name.split()[-1] if ' ' in mapped_name else mapped_name
                        if player_last == mapped_last:
                            jersey_num = name_to_jersey[team_name][mapped_name]
                            break
            
            if jersey_num:
                player_ids.append(f"{team_name}_{jersey_num}")
            else:
                # No match found, use player name
                player_ids.append(f"{team_name}_{player}")
        else:
            # Assume player is already a jersey number (Metrica format) or use as-is
            player_ids.append(f"{team_name}_{player}")
        # Calculate minutes played
        x_column = f'{team_name}_{player}_x'
        if active_mask.any():
            last_idx = active_mask[active_mask].index[-1]
            first_idx = active_mask[active_mask].index[0]
            if last_idx is not None and first_idx is not None:
                first_time = tracking_data.loc[first_idx, 'Time [s]']
                last_time = tracking_data.loc[last_idx, 'Time [s]']
                player_minutes = (last_time - first_time) / 60.0
            else:
                player_minutes = 0
        else:
            player_minutes = 0
        minutes.append(player_minutes)
        
        # Calculate distance covered
        speed_column = f'{team_name}_{player}_speed'
        if speed_column in tracking_data.columns:
            # Sum of distance = sum(speed * delta_time)
            # Speed is in m/s and delta_time is in seconds, so the result is meters.
            player_distance = ((tracking_data[speed_column].fillna(0) * active_mask.astype(float)) * time_deltas).sum() / 1000.0
        else:
            player_distance = 0
        distance.append(player_distance)
    
    summary['Player ID'] = player_ids
    summary['Minutes Played'] = minutes
    summary['Distance [km]'] = distance
    
    # Reorder columns to have Player ID first
    summary = summary[['Player ID', 'Minutes Played', 'Distance [km]']]
    
    # Sort by distance covered
    summary = summary.sort_values(['Distance [km]'], ascending=False)
    
    # Remove players who never appeared in the tracked regulation-time data.
    summary = summary[summary['Minutes Played'] > 0].copy()
    
    return summary


def print_summary_stats(summary, team_name='Home'):
    """
    Print summary statistics
    
    Parameters:
    -----------
    summary : DataFrame
        Player summary with distance and minutes data
    team_name : str
        Name of team
    """
    print(f"\n{'='*60}")
    print(f"{team_name} Team - Distance Summary")
    print(f"{'='*60}")
    print(f"{'Player':<10} {'Minutes Played':<20} {'Distance [km]':<15}")
    print(f"{'-'*60}")
    
    for player in summary.index:
        mins = summary.loc[player, 'Minutes Played']
        dist = summary.loc[player, 'Distance [km]']
        print(f"{player:<10} {mins:>15.1f} {dist:>18.2f}")
    
    print(f"{'-'*60}")
    print(f"{'Total':<10} {'':<20} {summary['Distance [km]'].sum():>15.2f}")
    print(f"{'Average':<10} {summary['Minutes Played'].mean():>15.1f} {summary['Distance [km]'].mean():>15.2f}")
    print(f"{'='*60}\n")


def main():
    """
    Main function to run the distance calculation
    """
    # Configuration
    DATADIR = 'Sample Data'
    
    # Get game_id from user input
    if len(sys.argv) > 1:
        try:
            game_id = int(sys.argv[1])
        except ValueError:
            print("Error: Game ID must be an integer")
            sys.exit(1)
    else:
        game_id_input = input("Enter game ID (Ex: 10517): ").strip()
        if game_id_input:
            try:
                game_id = int(game_id_input)
            except ValueError:
                print("Invalid input. Using default game ID = 2")
                game_id = 2
        else:
            game_id = 2
    
    print(f"\nLoading match data for game {game_id}...")
    
    # Read in tracking data
    tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
    tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')
    
    # Convert positions from metrica units to meters
    tracking_home = mio.to_metric_coordinates(tracking_home)
    tracking_away = mio.to_metric_coordinates(tracking_away)
    
    print("Calculating player velocities...")
    # Calculate player velocities (required for distance calculation)
    # Prefer PFF raw speed when present because it preserves the higher-fidelity speed signal.
    tracking_home = mvel.calc_player_velocities_hybrid(tracking_home, smoothing=True, use_pff_speed=True)
    tracking_away = mvel.calc_player_velocities_hybrid(tracking_away, smoothing=True, use_pff_speed=True)
    
    # Check if data is PFF format (has player names) or Metrica format (has jersey numbers)
    sample_cols = [c for c in tracking_home.columns if c.startswith('Home_') and c.endswith('_x')]
    if sample_cols:
        sample_player = sample_cols[0].split('_')[1]
        # If it's a digit, it's already jersey numbers (Metrica format)
        if sample_player.isdigit():
            print("Detected Metrica format data (jersey numbers already in columns)")
            name_to_jersey = None
        else:
            # It's player names (PFF format) - need to load jersey mapping
            print("Detected PFF format data - loading player jersey number mappings...")
            name_to_jersey = load_player_name_to_jersey_mapping(game_id)
    else:
        name_to_jersey = None
    
    # Calculate distances for both teams
    home_summary = calculate_player_distances(tracking_home, 'Home', name_to_jersey)
    away_summary = calculate_player_distances(tracking_away, 'Away', name_to_jersey)
    
    # Print summaries
    print_summary_stats(home_summary, 'Home')
    print_summary_stats(away_summary, 'Away')
    
    # Save results to Excel file with two sheets
    output_dir = os.path.join('Metrica_Output', 'Distances')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f'{output_dir}/game_{game_id}_player_distances.xlsx'
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        home_summary.to_excel(writer, sheet_name='Home Team')
        away_summary.to_excel(writer, sheet_name='Away Team')
    
    print(f"\nResults saved to: {output_file}")
    
    return home_summary, away_summary


if __name__ == '__main__':
    home_summary, away_summary = main()
