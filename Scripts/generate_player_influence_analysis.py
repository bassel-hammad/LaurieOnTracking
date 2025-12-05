#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Player Influence on Pitch Control - Sequential Perturbation Analysis

This script calculates each player's individual contribution to pitch control changes
by isolating their movement while keeping all other players frozen.

Method: For each frame transition (t → t+1):
1. Calculate baseline PC at time t (all players at t)
2. For each attacking player, move ONLY that player to t+1 position
3. Calculate ΔPC caused by that player's movement alone
4. Compare sum of individual contributions to actual total change

Optimizations:
- 5 FPS sampling (reduces frames)
- Only analyze players in attacking half (reduces players per frame)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import Metrica_Viz as mviz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print("=" * 70)
print("PLAYER INFLUENCE ANALYSIS")
print("Individual Contribution to Pitch Control Changes")
print("=" * 70)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================
DATADIR = 'Sample Data'
OUTPUT_DIR = 'Metrica_Output'

# Get match ID from user
game_id = input("Enter match ID (e.g., 10517): ").strip()
if not game_id:
    print("ERROR: Match ID is required!")
    sys.exit(1)

try:
    game_id = int(game_id)
except ValueError:
    print("ERROR: Match ID must be a number!")
    sys.exit(1)

print(f"\nSelected match ID: {game_id}")
print()

print("Loading data...")
events = mio.read_event_data(DATADIR, game_id)
tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')

tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

print(f"Data loaded: {len(events)} events, {len(tracking_home):,} tracking frames")
print()

# Check if PFF speed columns exist
pff_speed_cols = [c for c in tracking_home.columns if c.endswith('_pff_speed')]
if pff_speed_cols:
    print("Calculating player velocities using HYBRID method (PFF speed + calculated direction)...")
    tracking_home = mvel.calc_player_velocities_hybrid(tracking_home, smoothing=True, use_pff_speed=True)
    tracking_away = mvel.calc_player_velocities_hybrid(tracking_away, smoothing=True, use_pff_speed=True)
    print("  -> Using PFF's raw speed values (calculated at ~15 FPS) for more accurate velocities")
else:
    print("Calculating player velocities from position differences...")
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

# Get sequence number from user
available_sequences = events['Sequence'].dropna().unique()
available_sequences = sorted([int(s) for s in available_sequences if not pd.isna(s)])
print(f"\nAvailable sequences: {available_sequences}")
sequence_input = input("Enter sequence number (e.g., 1): ").strip()

if not sequence_input:
    print("ERROR: Sequence number is required!")
    sys.exit(1)

try:
    sequence_number = float(sequence_input)
except ValueError:
    print("ERROR: Sequence must be a number!")
    sys.exit(1)

# Filter events for this sequence
sequence_events = events[events['Sequence'] == sequence_number].copy()

if len(sequence_events) == 0:
    print(f"ERROR: No events found for sequence {sequence_number}!")
    sys.exit(1)

# Get frame window from sequence events (use frames instead of times for accuracy)
start_frame = sequence_events['Start Frame'].min()
end_frame = sequence_events['End Frame'].max()

# Get corresponding times from tracking data
if start_frame in tracking_home.index and end_frame in tracking_home.index:
    start_time = tracking_home.loc[start_frame, 'Time [s]']
    end_time = tracking_home.loc[end_frame, 'Time [s]']
else:
    # Fallback to time-based if frames not found
    start_time = sequence_events['Start Time [s]'].min()
    end_time = sequence_events['End Time [s]'].max()
    print("Warning: Using time-based lookup (frames not found in tracking data)")

print(f"\nAnalyzing sequence {sequence_number}")
print(f"  Events in sequence: {len(sequence_events)}")
print(f"  Frame range: {int(start_frame)} to {int(end_frame)}")
print(f"  Time window: {start_time:.1f}s to {end_time:.1f}s")
print()
print("First few events in this sequence:")
print(sequence_events[['Team', 'Type', 'From', 'To', 'Start Frame', 'Start Time [s]']].head(10).to_string(index=False))
print()

# Determine which team is attacking in this sequence (majority of events)
team_counts = sequence_events['Team'].value_counts()
default_attacking_team = team_counts.idxmax()
print(f"Detected attacking team in this sequence: {default_attacking_team}")
print()

# Ask user which team to analyze
print("Which team do you want to analyze?")
print("  1. Attacking team (space creation)")
print("  2. Defending team (space concession)")
team_choice = input(f"Enter choice (1-2, default: 1 for {default_attacking_team}): ").strip()

if team_choice == '2':
    # Analyze defending team
    attacking_team = 'Away' if default_attacking_team == 'Home' else 'Home'
    analyze_mode = 'defending'
    print(f"\nAnalyzing DEFENDING team: {attacking_team}")
    print("Focus: Players who LOST the most pitch control")
else:
    # Analyze attacking team (default)
    attacking_team = default_attacking_team
    analyze_mode = 'attacking'
    print(f"\nAnalyzing ATTACKING team: {attacking_team}")
    print("Focus: Players who CREATED the most space")
print()

params = mpc.default_model_params()
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
print(f"Goalkeepers: Home #{GK_numbers[0]}, Away #{GK_numbers[1]}")
print()

# =============================================================================
# LOAD PLAYER NAME MAPPING FROM JSON
# =============================================================================

print("Loading player name mapping...")
player_mapping = mio.load_player_mapping(DATADIR, game_id)

if player_mapping:
    print(f"  Home team ({player_mapping.get('teams', {}).get('Home', {}).get('name', 'Unknown')}): {len(player_mapping.get('players', {}).get('Home', {}))} players")
    print(f"  Away team ({player_mapping.get('teams', {}).get('Away', {}).get('name', 'Unknown')}): {len(player_mapping.get('players', {}).get('Away', {}))} players")
    
    # Convert to player_name_map format: {'Home_10': 'Lionel Messi', ...}
    player_name_map = {}
    for team in ['Home', 'Away']:
        team_players = player_mapping.get('players', {}).get(team, {})
        for jersey, name in team_players.items():
            player_name_map[f'{team}_{jersey}'] = name
else:
    print("  Warning: No player mapping found, using jersey numbers only")
    player_name_map = {}

print(f"  Total mapped: {len(player_name_map)} players")
print()

# =============================================================================
# PREPARE FRAMES TO ANALYZE - USE EVENT FRAMES
# =============================================================================

print(f"Frame range: {int(start_frame)} to {int(end_frame)}")
print(f"Time window: {start_time:.1f}s to {end_time:.1f}s")

# Get unique event frames from the sequence (both start and end frames)
event_frames = pd.concat([
    sequence_events['Start Frame'], 
    sequence_events['End Frame']
]).dropna().unique()
event_frames = sorted([int(f) for f in event_frames])

print(f"Events have {len(event_frames)} unique frame points")

# Use only frames that exist in tracking data
frames_to_analyze = []
for frame in event_frames:
    if frame in tracking_home.index and frame in tracking_away.index:
        frames_to_analyze.append(frame)

print(f"Analyzing {len(frames_to_analyze)} frames (at event frames)")
print()

# =============================================================================
# PITCH CONTROL CALCULATION FUNCTION
# =============================================================================

def calculate_pitch_control_surface(home_row, away_row, params, GK_numbers, attacking_half_only=True, attacking_team='Home'):
    """
    Calculate pitch control surface for given player positions
    
    Parameters:
    -----------
    home_row, away_row : Series
        Player position data for both teams
    params : dict
        Pitch control model parameters
    GK_numbers : list
        Goalkeeper jersey numbers [home_GK, away_GK]
    attacking_half_only : bool
        If True, only calculate PC in the attacking half (x > 0)
    attacking_team : str
        'Home' or 'Away' - determines which half is the attacking half
    """
    
    ball_pos = np.array([home_row['ball_x'], home_row['ball_y']])
    
    field_dimen = (106., 68.)
    n_grid_cells_x = 50
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0]/2. + dx/2.
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1]/2. + dy/2.
    
    PPCFa = np.zeros(shape=(len(ygrid), len(xgrid)))
    PPCFd = np.zeros(shape=(len(ygrid), len(xgrid)))
    
    # Initialize players
    attacking_players = mpc.initialise_players(home_row, 'Home', params, GK_numbers[0], is_attacking=True)
    defending_players = mpc.initialise_players(away_row, 'Away', params, GK_numbers[1], is_attacking=False)
    
    # Calculate pitch control at each grid location
    for ii in range(len(ygrid)):
        for jj in range(len(xgrid)):
            target_position = np.array([xgrid[jj], ygrid[ii]])
            
            # Only calculate in attacking half if flag is set
            if attacking_half_only:
                # Home team attacks towards positive x (right side)
                # Away team attacks towards negative x (left side)
                if attacking_team == 'Home' and target_position[0] < 0:
                    # Skip defensive half for Home team
                    PPCFa[ii, jj] = np.nan
                    PPCFd[ii, jj] = np.nan
                    continue
                elif attacking_team == 'Away' and target_position[0] > 0:
                    # Skip defensive half for Away team
                    PPCFa[ii, jj] = np.nan
                    PPCFd[ii, jj] = np.nan
                    continue
            
            PPCFa[ii, jj], PPCFd[ii, jj] = mpc.calculate_pitch_control_at_target(
                target_position, attacking_players, defending_players, ball_pos, params
            )
    
    return PPCFa, xgrid, ygrid

# =============================================================================
# PLAYER INFLUENCE ANALYSIS
# =============================================================================

print("=" * 70)
print("CALCULATING PLAYER INFLUENCES")
print("=" * 70)
print()
print(f"NOTE: Only calculating pitch control changes in the ATTACKING HALF")
print(f"      (the half where {default_attacking_team} team is attacking)")
print()

# Store results
influence_results = []

# Analyze consecutive frame pairs
for i in range(len(frames_to_analyze) - 1):
    frame_t = frames_to_analyze[i]
    frame_t1 = frames_to_analyze[i + 1]
    
    if i % 10 == 0:
        print(f"  Processing frame {i+1}/{len(frames_to_analyze)-1}...")
    
    home_t = tracking_home.loc[frame_t]
    away_t = tracking_away.loc[frame_t]
    home_t1 = tracking_home.loc[frame_t1]
    away_t1 = tracking_away.loc[frame_t1]
    
    time_t = home_t['Time [s]']
    time_t1 = home_t1['Time [s]']
    
    # Get backfilled rows for pitch control calculation
    home_t_backfilled = mpc._row_with_backfilled_velocities(tracking_home, frame_t)
    away_t_backfilled = mpc._row_with_backfilled_velocities(tracking_away, frame_t)
    home_t1_backfilled = mpc._row_with_backfilled_velocities(tracking_home, frame_t1)
    away_t1_backfilled = mpc._row_with_backfilled_velocities(tracking_away, frame_t1)
    
    # 1. Baseline: PC at time t (all players at t)
    # Only calculate in attacking half of the team controlling the ball
    PC_baseline, xgrid, ygrid = calculate_pitch_control_surface(
        home_t_backfilled, away_t_backfilled, params, GK_numbers,
        attacking_half_only=True, attacking_team=default_attacking_team
    )
    
    # 2. Actual: PC at time t+1 (all players moved)
    PC_actual, _, _ = calculate_pitch_control_surface(
        home_t1_backfilled, away_t1_backfilled, params, GK_numbers,
        attacking_half_only=True, attacking_team=default_attacking_team
    )
    
    ΔPC_actual = PC_actual - PC_baseline
    
    # 3. Identify all attacking team players based on sequence
    if attacking_team == 'Home':
        attacking_tracking = tracking_home
        defending_tracking = tracking_away
        attack_t = home_t
        attack_t1 = home_t1
        attack_t_backfilled = home_t_backfilled
        attack_t1_backfilled = home_t1_backfilled
        defend_t_backfilled = away_t_backfilled
    else:  # Away
        attacking_tracking = tracking_away
        defending_tracking = tracking_home
        attack_t = away_t
        attack_t1 = away_t1
        attack_t_backfilled = away_t_backfilled
        attack_t1_backfilled = away_t1_backfilled
        defend_t_backfilled = home_t_backfilled
    
    attack_player_cols = [c for c in attack_t.keys() 
                        if c[-2:].lower()=='_x' and c!='ball_x' 
                        and 'visibility' not in c.lower()]
    
    attacking_players = []
    for col in attack_player_cols:
        player_id = col.replace('_x', '')
        x_pos = attack_t[col]
        
        # Include all players
        if not pd.isna(x_pos):
            attacking_players.append(player_id)
    
    # 4. Calculate influence for each attacking player
    player_influences = {}
    
    for player_id in attacking_players:
        # Create hybrid frame: only this player moves to t+1
        attack_hybrid = attack_t_backfilled.copy()
        
        # Update only this player's position to t+1
        attack_hybrid[f'{player_id}_x'] = attack_t1_backfilled[f'{player_id}_x']
        attack_hybrid[f'{player_id}_y'] = attack_t1_backfilled[f'{player_id}_y']
        
        # Update velocities if they exist
        if f'{player_id}_vx' in attack_t1_backfilled.keys():
            attack_hybrid[f'{player_id}_vx'] = attack_t1_backfilled[f'{player_id}_vx']
            attack_hybrid[f'{player_id}_vy'] = attack_t1_backfilled[f'{player_id}_vy']
            attack_hybrid[f'{player_id}_speed'] = attack_t1_backfilled[f'{player_id}_speed']
        
        # Calculate PC with only this player moved (team order matters for pitch control)
        if attacking_team == 'Home':
            PC_only_this_player, _, _ = calculate_pitch_control_surface(
                attack_hybrid, defend_t_backfilled, params, GK_numbers,
                attacking_half_only=True, attacking_team=default_attacking_team
            )
        else:  # Away team attacking
            PC_only_this_player, _, _ = calculate_pitch_control_surface(
                defend_t_backfilled, attack_hybrid, params, GK_numbers,
                attacking_half_only=True, attacking_team=default_attacking_team
            )
            # For away team, invert the pitch control (1 - PPCF)
            PC_only_this_player = 1 - PC_only_this_player
            PC_baseline_away = 1 - PC_baseline
        
        # Calculate influence
        if attacking_team == 'Home':
            ΔPC_this_player = PC_only_this_player - PC_baseline
        else:
            ΔPC_this_player = PC_only_this_player - PC_baseline_away
        
        # Store influence (use nansum to handle NaN values in defensive half)
        player_influences[player_id] = {
            'delta_PC': ΔPC_this_player,
            'total_influence': np.nansum(np.abs(ΔPC_this_player)),
            'positive_influence': np.nansum(np.where(ΔPC_this_player > 0, ΔPC_this_player, 0)),
            'negative_influence': np.nansum(np.where(ΔPC_this_player < 0, ΔPC_this_player, 0)),
            'position_t': (attack_t[f'{player_id}_x'], attack_t[f'{player_id}_y']),
            'position_t1': (attack_t1[f'{player_id}_x'], attack_t1[f'{player_id}_y']),
        }
    
    # 5. Calculate sum of attributions (use nansum for NaN-safe calculation)
    if len(player_influences) > 0:
        ΔPC_sum = np.nansum([p['delta_PC'] for p in player_influences.values()], axis=0)
    else:
        ΔPC_sum = np.zeros_like(PC_baseline)
    
    # 6. Calculate interaction term
    interaction = ΔPC_actual - ΔPC_sum
    
    # Store results
    influence_results.append({
        'frame_t': frame_t,
        'frame_t1': frame_t1,
        'time_t': time_t,
        'time_t1': time_t1,
        'PC_baseline': PC_baseline,
        'PC_actual': PC_actual,
        'ΔPC_actual': ΔPC_actual,
        'ΔPC_sum': ΔPC_sum,
        'interaction': interaction,
        'player_influences': player_influences,
        'num_attacking_players': len(attacking_players),
        'xgrid': xgrid,
        'ygrid': ygrid,
    })

print()
print(f"Analysis complete: {len(influence_results)} frame transitions analyzed")
print()

# =============================================================================
# SUMMARY STATISTICS - PRINT TO TERMINAL
# =============================================================================

print("=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print()

# Aggregate player influences across all frames
player_total_influences = {}

for result in influence_results:
    for player_id, influence_data in result['player_influences'].items():
        if player_id not in player_total_influences:
            player_total_influences[player_id] = {
                'total': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'net': 0.0,
                'frames': 0
            }
        
        player_total_influences[player_id]['total'] += influence_data['total_influence']
        player_total_influences[player_id]['positive'] += influence_data['positive_influence']
        player_total_influences[player_id]['negative'] += influence_data['negative_influence']
        player_total_influences[player_id]['net'] = player_total_influences[player_id]['positive'] + player_total_influences[player_id]['negative']
        player_total_influences[player_id]['frames'] += 1

# ===== TABLE 1: TOTAL INFLUENCE RANKINGS =====
sorted_by_total = sorted(player_total_influences.items(), 
                         key=lambda x: x[1]['total'], 
                         reverse=True)

print("TABLE 1: TOTAL INFLUENCE RANKINGS (Magnitude of Impact)")
print("=" * 80)
print(f"{'Rank':<6} {'Player':<25} {'Total':<12} {'Frames':<8}")
print("-" * 80)

for rank, (player_id, stats) in enumerate(sorted_by_total, 1):
    player_name = player_name_map.get(player_id, player_id)
    print(f"{rank:<6} {player_name:<25} {stats['total']:>11.3f} {stats['frames']:>7}")

print()
print(f"Total players analyzed: {len(sorted_by_total)}")
print()
print()

# ===== TABLE 2: POSITIVE INFLUENCE RANKINGS =====
sorted_by_positive = sorted(player_total_influences.items(), 
                            key=lambda x: x[1]['positive'], 
                            reverse=True)

print("TABLE 2: POSITIVE INFLUENCE RANKINGS (Space Creation)")
print("=" * 80)
print(f"{'Rank':<6} {'Player':<25} {'Positive':<12} {'Frames':<8}")
print("-" * 80)

for rank, (player_id, stats) in enumerate(sorted_by_positive, 1):
    player_name = player_name_map.get(player_id, player_id)
    print(f"{rank:<6} {player_name:<25} {stats['positive']:>11.3f} {stats['frames']:>7}")

print()
print()

# ===== TABLE 3: NEGATIVE INFLUENCE RANKINGS =====
sorted_by_negative = sorted(player_total_influences.items(), 
                            key=lambda x: x[1]['negative'], 
                            reverse=False)  # Ascending (most negative first)

print("TABLE 3: NEGATIVE INFLUENCE RANKINGS (Space Concession)")
print("=" * 80)
print(f"{'Rank':<6} {'Player':<25} {'Negative':<12} {'Frames':<8}")
print("-" * 80)

for rank, (player_id, stats) in enumerate(sorted_by_negative, 1):
    player_name = player_name_map.get(player_id, player_id)
    print(f"{rank:<6} {player_name:<25} {stats['negative']:>11.3f} {stats['frames']:>7}")

print()

# Store the player with most negative influence for highlighting in movie
most_negative_player = sorted_by_negative[0][0] if sorted_by_negative else None
most_negative_value = sorted_by_negative[0][1]['negative'] if sorted_by_negative else 0
print(f"*** Player with MOST NEGATIVE influence: {player_name_map.get(most_negative_player, most_negative_player)} ({most_negative_value:.3f})")
print()

# ===== TABLE 4: NET INFLUENCE RANKINGS =====
sorted_by_net = sorted(player_total_influences.items(), 
                       key=lambda x: x[1]['net'], 
                       reverse=True)

print("TABLE 4: NET INFLUENCE RANKINGS (Positive + Negative)")
print("=" * 80)
print(f"{'Rank':<6} {'Player':<25} {'Net Gain':<12} {'Positive':<12} {'Negative':<12} {'Frames':<8}")
print("-" * 80)

for rank, (player_id, stats) in enumerate(sorted_by_net, 1):
    player_name = player_name_map.get(player_id, player_id)
    print(f"{rank:<6} {player_name:<25} {stats['net']:>11.3f} {stats['positive']:>11.3f} "
          f"{stats['negative']:>11.3f} {stats['frames']:>7}")

print()
print(f"Total players analyzed: {len(sorted_by_net)}")
print()

# Interaction analysis (use nansum for NaN-safe calculation)
total_interaction = np.nansum([np.nansum(np.abs(r['interaction'])) for r in influence_results])
total_actual_change = np.nansum([np.nansum(np.abs(r['ΔPC_actual'])) for r in influence_results])
interaction_percentage = (total_interaction / total_actual_change * 100) if total_actual_change > 0 else 0

print(f"Total actual dPC (attacking half only): {total_actual_change:.3f}")
print(f"Total interaction term:  {total_interaction:.3f}")
print(f"Interaction percentage:  {interaction_percentage:.1f}%")
print()

if interaction_percentage < 10:
    print("Low interaction - linear approximation is good!")
elif interaction_percentage < 20:
    print("~ Moderate interaction - some nonlinear effects")
else:
    print("! High interaction - significant player coupling")

print()

print("=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print()
print(f"Match ID: {game_id}")
print(f"Sequence: {sequence_number}")
print(f"Time window: {start_time:.1f}s to {end_time:.1f}s")
print(f"Events in sequence: {len(sequence_events)}")
print(f"Frames analyzed: {len(influence_results)}")
print()

# =============================================================================
# GENERATE PITCH CONTROL MOVIE
# =============================================================================

print("=" * 70)
print("GENERATING PITCH CONTROL MOVIE")
print("=" * 70)
print()

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Generate filename based on analysis mode
if analyze_mode == 'defending':
    movie_filename = f'sequence_{int(sequence_number)}_defending_influence.mp4'
else:
    movie_filename = f'sequence_{int(sequence_number)}_attacking_influence.mp4'
output_path = os.path.join(OUTPUT_DIR, movie_filename)

# Generate frames for movie using time-based sampling (like the tutorial)
TARGET_FPS = 5  # Frames per second for movie
time_interval = 1.0 / TARGET_FPS  # Time between frames
sample_times = np.arange(start_time, end_time + time_interval/2, time_interval)

print(f"Calculating pitch control for movie at {TARGET_FPS} FPS...")
print(f"  Time range: {start_time:.1f}s to {end_time:.1f}s")
print(f"  Sequence duration: {(end_time - start_time):.1f}s")
print(f"  Sampling at {TARGET_FPS} FPS: {len(sample_times)} frames")

# Find tracking frames closest to each sample time
frame_times = tracking_home['Time [s]'].values
movie_frames = []

for sample_time in sample_times:
    time_diffs = np.abs(frame_times - sample_time)
    closest_idx = np.argmin(time_diffs)
    frame = tracking_home.index[closest_idx]
    
    if frame in tracking_away.index:
        movie_frames.append(frame)

print(f"  Found {len(movie_frames)} valid frames to render")

# Calculate pitch control for each frame
pitch_control_data = []

for i, frame in enumerate(movie_frames):
    if i % 10 == 0:
        print(f"  Frame {i+1}/{len(movie_frames)}...")
    
    if frame not in tracking_home.index or frame not in tracking_away.index:
        continue
    
    home_row = tracking_home.loc[frame].copy()
    away_row = tracking_away.loc[frame].copy()
    
    pitch_control_data.append({
        'home_row': home_row,
        'away_row': away_row,
        'ball_pos': np.array([home_row['ball_x'], home_row['ball_y']]),
        'time': home_row['Time [s]'],
        'frame': frame
    })

print()
print("Creating animation...")

# Get all 11 players for bar chart - use different ranking based on mode
if analyze_mode == 'defending':
    # For defending team, show players ranked by NEGATIVE influence (space lost)
    all_players = sorted_by_negative[:11]
    all_player_names = [player_name_map.get(p[0], p[0]) for p in all_players]
    all_player_influences = [abs(p[1]['negative']) for p in all_players]  # Use absolute value for bar height
    all_player_ids = [p[0] for p in all_players]
    
    # Create player ranking map for annotations (by negative influence)
    player_rankings = {p[0]: rank for rank, p in enumerate(sorted_by_negative, 1)}
    
    print(f"Bar chart: Showing top 11 defenders by NEGATIVE influence (space conceded)")
    print(f"Highlighting: {player_name_map.get(most_negative_player, most_negative_player)} with {most_negative_value:.3f}")
else:
    # For attacking team, show players ranked by POSITIVE influence (space created)
    all_players = sorted_by_positive[:11]
    all_player_names = [player_name_map.get(p[0], p[0]) for p in all_players]
    all_player_influences = [p[1]['positive'] for p in all_players]
    all_player_ids = [p[0] for p in all_players]
    
    # Create player ranking map for annotations (by positive influence)
    player_rankings = {p[0]: rank for rank, p in enumerate(sorted_by_positive, 1)}
    
    print(f"Bar chart: Showing top 11 attackers by POSITIVE influence (space created)")
    print(f"Highlighting: Top 5 players by positive influence")

# Create figure with pitch on left and bar chart on right
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[20, 1], hspace=0.1, wspace=0.3)

# Pitch subplot (left side)
ax_pitch = fig.add_subplot(gs[0, 0])
ax_pitch.set_xlim(-55, 55)
ax_pitch.set_ylim(-36, 36)
ax_pitch.set_aspect('equal')
ax_pitch.axis('off')

# Draw green pitch rectangle
from matplotlib.patches import Rectangle, Arc, Circle
pitch_rect = Rectangle((-52.5, -34), 105, 68, facecolor='#2E8B57', edgecolor='white', linewidth=2, zorder=0)
ax_pitch.add_patch(pitch_rect)

# Draw pitch lines in white
ax_pitch.plot([0, 0], [-34, 34], 'white', linewidth=2)  # Halfway line
ax_pitch.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor='white', linewidth=2))  # Center circle

# Penalty areas
# Left penalty area (16.5m)
ax_pitch.plot([-52.5, -36], [-20.16, -20.16], 'white', linewidth=2)
ax_pitch.plot([-36, -36], [-20.16, 20.16], 'white', linewidth=2)
ax_pitch.plot([-52.5, -36], [20.16, 20.16], 'white', linewidth=2)

# Right penalty area
ax_pitch.plot([52.5, 36], [-20.16, -20.16], 'white', linewidth=2)
ax_pitch.plot([36, 36], [-20.16, 20.16], 'white', linewidth=2)
ax_pitch.plot([52.5, 36], [20.16, 20.16], 'white', linewidth=2)

# Goal areas (6 yard box)
# Left goal area
ax_pitch.plot([-52.5, -47], [-9.16, -9.16], 'white', linewidth=2)
ax_pitch.plot([-47, -47], [-9.16, 9.16], 'white', linewidth=2)
ax_pitch.plot([-52.5, -47], [9.16, 9.16], 'white', linewidth=2)

# Right goal area
ax_pitch.plot([52.5, 47], [-9.16, -9.16], 'white', linewidth=2)
ax_pitch.plot([47, 47], [-9.16, 9.16], 'white', linewidth=2)
ax_pitch.plot([52.5, 47], [9.16, 9.16], 'white', linewidth=2)

# Penalty spots
ax_pitch.plot(-41.5, 0, 'o', color='white', markersize=4)
ax_pitch.plot(41.5, 0, 'o', color='white', markersize=4)

# Corner arcs
corner_arc1 = Arc((-52.5, -34), 2, 2, angle=0, theta1=0, theta2=90, color='white', linewidth=2)
corner_arc2 = Arc((-52.5, 34), 2, 2, angle=0, theta1=270, theta2=360, color='white', linewidth=2)
corner_arc3 = Arc((52.5, -34), 2, 2, angle=0, theta1=90, theta2=180, color='white', linewidth=2)
corner_arc4 = Arc((52.5, 34), 2, 2, angle=0, theta1=180, theta2=270, color='white', linewidth=2)
ax_pitch.add_patch(corner_arc1)
ax_pitch.add_patch(corner_arc2)
ax_pitch.add_patch(corner_arc3)
ax_pitch.add_patch(corner_arc4)

# Bar chart subplot (right side) - vertical bars
ax_bar = fig.add_subplot(gs[0, 1])
# Initialize bars with zero height - they'll be updated in animation
bar_color = '#8B0000' if analyze_mode == 'defending' else 'crimson'  # Dark red for defending
bars = ax_bar.bar(range(len(all_player_names)), [0] * len(all_player_names), color=bar_color, alpha=0.7, width=0.7)
ax_bar.set_xticks(range(len(all_player_names)))
ax_bar.set_xticklabels([f"{i+1}" for i in range(len(all_player_names))], fontsize=9)

if analyze_mode == 'defending':
    ax_bar.set_ylabel('Negative Influence (Space Lost)', fontsize=11, fontweight='bold')
    ax_bar.set_xlabel('Player Rank (by space lost)', fontsize=10)
    ax_bar.set_title(f'Defending Team - Space Concession', fontsize=12, fontweight='bold')
else:
    ax_bar.set_ylabel('Positive Influence (Space Created)', fontsize=11, fontweight='bold')
    ax_bar.set_xlabel('Player Rank (by space created)', fontsize=10)
    ax_bar.set_title(f'Attacking Team - Space Creation', fontsize=12, fontweight='bold')

ax_bar.grid(axis='y', alpha=0.3)

# Set y-axis to maximum value from the start (final values)
max_influence = max(all_player_influences) if all_player_influences else 1
ax_bar.set_ylim(0, max_influence * 1.15)

# Store player name text objects (will be positioned below the frame)
player_name_texts = []
for i, name in enumerate(all_player_names):
    # Calculate position in figure coordinates
    x_pos = 0.69 + (i / len(all_player_names)) * 0.24  # Right half of figure
    text_obj = fig.text(x_pos, 0.13, name, 
                       rotation=45, ha='right', va='top', fontsize=7)
    player_name_texts.append(text_obj)

# Generate distinct colors for each event (using a colormap)
import matplotlib.cm as cm
num_events = len(influence_results)
event_colors = cm.rainbow(np.linspace(0, 1, num_events))

# Add values on top of bars
for i, (bar, val) in enumerate(zip(bars, all_player_influences)):
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)

ax = ax_pitch

# No pitch control visualization - just green pitch
data = pitch_control_data[0]

# Initialize time text
time_text = ax.text(
    0.02, 0.98, '',
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# Store references to player and ball artists that will be updated
player_artists = []

def animate(frame_idx):
    """Animation function"""
    global player_artists
    data = pitch_control_data[frame_idx]
    
    # Clear previous player positions only (not pitch lines)
    for artist in player_artists:
        artist.remove()
    player_artists = []
    
    # Clear text annotations (ranking numbers)
    for txt in ax.texts[:]:
        if txt != time_text:  # Don't remove the time text
            txt.remove()
    
    home_row = data['home_row']
    away_row = data['away_row']
    
    # Determine which team is attacking and which is defending
    if attacking_team == 'Home':
        attack_row = home_row
        defend_row = away_row
        attack_color_top5 = "#FF0033"  # Bright red for top 5
        attack_color_others = "#CC2F2F"  # Darker red for others
        defend_color = '#1E90FF'  # Blue
    else:  # Away
        attack_row = away_row
        defend_row = home_row
        attack_color_top5 = "#0033FF"  # Bright blue for top 5
        attack_color_others = "#2F2FCC"  # Darker blue for others
        defend_color = '#FF6B6B'  # Light red
    
    # Get player column names for the attacking team
    x_cols_attack = [c for c in attack_row.keys() if c[-2:].lower()=='_x' and c!='ball_x' and 'visibility' not in c.lower()]
    y_cols_attack = [c for c in attack_row.keys() if c[-2:].lower()=='_y' and c!='ball_y' and 'visibility' not in c.lower()]
    
    # Determine which players to highlight based on analysis mode
    if analyze_mode == 'defending':
        # Highlight top 5 players with most NEGATIVE influence (lost most space)
        top_5_x = []
        top_5_y = []
        other_x = []
        other_y = []
        
        for x_col, y_col in zip(x_cols_attack, y_cols_attack):
            player_id = x_col.replace('_x', '')
            x_pos = attack_row[x_col]
            y_pos = attack_row[y_col]
            
            if not pd.isna(x_pos) and not pd.isna(y_pos):
                if player_id in player_rankings and player_rankings[player_id] <= 5:
                    top_5_x.append(x_pos)
                    top_5_y.append(y_pos)
                else:
                    other_x.append(x_pos)
                    other_y.append(y_pos)
        
        # Plot other players (with black stroke)
        if other_x:
            line = ax.plot(other_x, other_y, 'o', color=attack_color_others, markersize=12, alpha=0.6, 
                    markeredgecolor='black', markeredgewidth=1.5)[0]
            player_artists.append(line)
        
        # Plot top 5 players with most negative influence (with black stroke)
        if top_5_x:
            line = ax.plot(top_5_x, top_5_y, 'o', color=attack_color_top5, markersize=14, alpha=0.9,
                    markeredgecolor='black', markeredgewidth=1.5)[0]
            player_artists.append(line)
        
        # Add ranking numbers to top 5 players
        for x_col, y_col in zip(x_cols_attack, y_cols_attack):
            player_id = x_col.replace('_x', '')
            if player_id in player_rankings and player_rankings[player_id] <= 5:
                x_pos = attack_row[x_col]
                y_pos = attack_row[y_col]
                if not pd.isna(x_pos) and not pd.isna(y_pos):
                    rank = player_rankings[player_id]
                    txt = ax.text(x_pos, y_pos, str(rank), 
                           fontsize=9, fontweight='bold', color='white',
                           ha='center', va='center', zorder=11)
                    player_artists.append(txt)
    else:
        # Attacking mode: highlight top 5 by positive influence
        top_5_x = []
        top_5_y = []
        other_x = []
        other_y = []
        
        for x_col, y_col in zip(x_cols_attack, y_cols_attack):
            player_id = x_col.replace('_x', '')
            x_pos = attack_row[x_col]
            y_pos = attack_row[y_col]
            
            if not pd.isna(x_pos) and not pd.isna(y_pos):
                if player_id in player_rankings and player_rankings[player_id] <= 5:
                    top_5_x.append(x_pos)
                    top_5_y.append(y_pos)
                else:
                    other_x.append(x_pos)
                    other_y.append(y_pos)
        
        # Plot other attacking players (with black stroke)
        if other_x:
            line = ax.plot(other_x, other_y, 'o', color=attack_color_others, markersize=12, alpha=0.6, 
                    markeredgecolor='black', markeredgewidth=1.5)[0]
            player_artists.append(line)
        
        # Plot top 5 attacking players (with black stroke)
        if top_5_x:
            line = ax.plot(top_5_x, top_5_y, 'o', color=attack_color_top5, markersize=14, alpha=0.9,
                    markeredgecolor='black', markeredgewidth=1.5)[0]
            player_artists.append(line)
        
        # Add ranking numbers to top 5 players only
        for x_col, y_col in zip(x_cols_attack, y_cols_attack):
            player_id = x_col.replace('_x', '')
            if player_id in player_rankings and player_rankings[player_id] <= 5:  # Only top 5
                x_pos = attack_row[x_col]
                y_pos = attack_row[y_col]
                if not pd.isna(x_pos) and not pd.isna(y_pos):
                    rank = player_rankings[player_id]
                    txt = ax.text(x_pos, y_pos, str(rank), 
                           fontsize=9, fontweight='bold', color='white',
                           ha='center', va='center', zorder=11)
                    player_artists.append(txt)
    
    # Defending team with black stroke
    x_cols_defend = [c for c in defend_row.keys() if c[-2:].lower()=='_x' and c!='ball_x' and 'visibility' not in c.lower()]
    y_cols_defend = [c for c in defend_row.keys() if c[-2:].lower()=='_y' and c!='ball_y' and 'visibility' not in c.lower()]
    line = ax.plot(defend_row[x_cols_defend], defend_row[y_cols_defend], 'o', color=defend_color, markersize=12, alpha=0.7,
            markeredgecolor='black', markeredgewidth=1.5)[0]
    player_artists.append(line)
    
    # Ball
    ball_pos = data['ball_pos']
    ball_artist = ax.plot(ball_pos[0], ball_pos[1], 'ko', markersize=8, alpha=1.0, linewidth=0, zorder=10)[0]
    player_artists.append(ball_artist)
    
    # Update time text
    time_str = f"Time: {data['time']:.1f}s | Frame: {frame_idx+1}/{len(pitch_control_data)}"
    time_text.set_text(time_str)
    
    # Update bar chart heights dynamically based on current frame time
    # Build stacked bars with different colors for each event
    current_time = data['time']
    
    # Clear old bars and rebuild with stacked segments
    ax_bar.clear()
    ax_bar.set_xticks(range(len(all_player_names)))
    ax_bar.set_xticklabels([f"{i+1}" for i in range(len(all_player_names))], fontsize=9)
    
    if analyze_mode == 'defending':
        ax_bar.set_ylabel('Negative Influence (Space Lost)', fontsize=11, fontweight='bold')
        ax_bar.set_xlabel('Player Rank (by space lost)', fontsize=10)
        ax_bar.set_title(f'Defending Team - Space Concession', fontsize=12, fontweight='bold')
    else:
        ax_bar.set_ylabel('Positive Influence (Space Created)', fontsize=11, fontweight='bold')
        ax_bar.set_xlabel('Player Rank (by space created)', fontsize=10)
        ax_bar.set_title(f'Attacking Team - Space Creation', fontsize=12, fontweight='bold')
    
    ax_bar.grid(axis='y', alpha=0.3)
    ax_bar.set_ylim(0, max_influence * 1.15)
    
    # Build stacked bars - each event is a different segment
    bottoms = [0.0] * len(all_player_ids)
    
    for event_idx, result in enumerate(influence_results):
        if result['time_t1'] <= current_time:
            segment_heights = []
            for player_id in all_player_ids:
                if player_id in result['player_influences']:
                    if analyze_mode == 'defending':
                        # Use absolute value of negative influence for bar height
                        segment_heights.append(abs(result['player_influences'][player_id]['negative_influence']))
                    else:
                        # Use positive influence for attacking
                        segment_heights.append(result['player_influences'][player_id]['positive_influence'])
                else:
                    segment_heights.append(0.0)
            
            # Draw this event's segment with its unique color
            ax_bar.bar(range(len(all_player_names)), segment_heights, 
                      bottom=bottoms, color=event_colors[event_idx], 
                      alpha=0.8, width=0.7, edgecolor='white', linewidth=0.5)
            
            # Update bottoms for next segment
            bottoms = [b + h for b, h in zip(bottoms, segment_heights)]
    
    # Add value labels on top of bars
    for i, total in enumerate(bottoms):
        if total > 0:
            ax_bar.text(i, total, f'{total:.1f}', ha='center', va='bottom', fontsize=7)
    
    return [time_text]

# Create animation
print(f"Generating animation with {len(pitch_control_data)} frames...")
interval_ms = int(1000 / TARGET_FPS)  # milliseconds between frames
anim = animation.FuncAnimation(
    fig, 
    animate, 
    frames=len(pitch_control_data),
    interval=interval_ms,
    blit=False,
    repeat=True
)

# Add title
mode_text = "Defending Team - Space Concession" if analyze_mode == 'defending' else "Attacking Team - Space Creation"
fig.suptitle(f"Player Influence Analysis - Sequence {int(sequence_number)} ({mode_text})", fontsize=15, y=0.98)

# Save animation
print(f"Saving movie to: {output_path}")
print("Note: This may take a while...")
sys.stdout.flush()  # Flush output before animation save

writer = animation.FFMpegWriter(
    fps=TARGET_FPS,
    metadata=dict(artist='LaurieOnTracking'),
    bitrate=5000
)

try:
    # Suppress matplotlib animation progress output to reduce clutter
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anim.save(output_path, writer=writer, dpi=150)
    
    # Clear any duplicate prints and show final status
    print("\r" + " " * 80 + "\r", end='')  # Clear line
    print()
    print("=" * 70)
    print("MOVIE GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Movie saved to: {output_path}")
    print(f"Frames: {len(pitch_control_data)}")
    print(f"Game time duration: {end_time - start_time:.1f} seconds")
    print(f"Movie duration: {len(pitch_control_data) / TARGET_FPS:.1f} seconds")
    print(f"Playback FPS: {TARGET_FPS}")
    print()
    
except Exception as e:
    print()
    print(f"ERROR: Failed to save movie: {e}")
    print()
    print("Note: This script requires FFMpeg to be installed.")
    print("To install FFMpeg:")
    print("  - Windows: choco install ffmpeg")
    print("  - Mac: brew install ffmpeg")
    print("  - Linux: sudo apt-get install ffmpeg")
    print()
    
    # Try to save as GIF as fallback
    print("Attempting to save as GIF instead...")
    gif_path = output_path.replace('.mp4', '.gif')
    try:
        anim.save(gif_path, writer='pillow', fps=1, dpi=100)
        print(f"GIF saved to: {gif_path}")
    except Exception as gif_error:
        print(f"GIF save also failed: {gif_error}")

plt.close()
