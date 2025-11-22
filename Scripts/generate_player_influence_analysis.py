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
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("=" * 70)
print("PLAYER INFLUENCE ANALYSIS")
print("Individual Contribution to Pitch Control Changes")
print("=" * 70)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================
DATADIR = 'Sample Data'
game_id = 10517
OUTPUT_DIR = 'Metrica_Output'

# Time window configuration
TIME_WINDOW_BEFORE_GOAL = 10
TIME_WINDOW_AFTER_GOAL = 2
TARGET_FPS = 5

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

print("Calculating player velocities...")
tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

# Find Di María's goal
shots = events[events['Type']=='SHOT']
goals = shots[shots['Subtype'].str.contains('GOAL', na=False)].copy()

if len(goals) < 2:
    print("ERROR: Di María's goal not found!")
    sys.exit(1)

dimaria_goal = goals.iloc[1]
goal_frame = int(dimaria_goal['Start Frame'])
goal_scorer = dimaria_goal.get('From', 'Ángel Di María')

if goal_frame not in tracking_home.index:
    print(f"ERROR: Goal frame {goal_frame} not found!")
    sys.exit(1)

goal_time = tracking_home.loc[goal_frame, 'Time [s]']

print(f"Analyzing: {goal_scorer}'s goal")
print(f"  Frame: {goal_frame}")
print(f"  Time: {goal_time:.1f}s")
print()

params = mpc.default_model_params()
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
print(f"Goalkeepers: Home #{GK_numbers[0]}, Away #{GK_numbers[1]}")
print()

# =============================================================================
# CREATE PLAYER NAME MAPPING FROM EVENTS
# =============================================================================

print("Creating player name mapping...")
player_name_map = {}

# Get all unique players from event data
for idx, event in events.iterrows():
    if pd.notna(event.get('From')):
        player_name = event.get('From')
        team = event.get('Team')
        
        # Try to find matching player number in tracking data
        # This is approximate - we'll use jersey numbers from events if available
        if team == 'Home':
            # Look through home tracking columns for this player
            # Event data might have player numbers we can match
            player_id = f"Home_{player_name.split()[-1] if player_name.split()[-1].isdigit() else ''}"
            if player_id != "Home_":
                player_name_map[player_id] = player_name

# Manual mapping for known players (World Cup Final 2022)
# Argentina (Home) player numbers
known_players = {
    'Home_10': 'Lionel Messi',
    'Home_11': 'Ángel Di María',
    'Home_9': 'Julián Álvarez',
    'Home_7': 'Rodrigo De Paul',
    'Home_24': 'Enzo Fernández',
    'Home_20': 'Alexis Mac Allister',
    'Home_23': 'Emiliano Martínez',  # GK
    'Home_26': 'Nahuel Molina',
    'Home_3': 'Nicolás Tagliafico',
    'Home_13': 'Cristian Romero',
    'Home_19': 'Nicolás Otamendi',
    'Home_5': 'Leandro Paredes',
    'Home_8': 'Marcos Acuña',
    'Home_21': 'Paulo Dybala',
    'Home_22': 'Lautaro Martínez',
}

# Merge with any found from events
player_name_map.update(known_players)

print(f"  Mapped {len(player_name_map)} player names")
print()

# =============================================================================
# PREPARE FRAMES TO ANALYZE
# =============================================================================

start_time = max(0, goal_time - TIME_WINDOW_BEFORE_GOAL)
end_time = goal_time + TIME_WINDOW_AFTER_GOAL

print(f"Time window: {start_time:.1f}s to {end_time:.1f}s")

frame_times = tracking_home['Time [s]'].values
time_mask = (frame_times >= start_time) & (frame_times <= end_time)
all_frames = tracking_home.index[time_mask].tolist()

time_interval = 1.0 / TARGET_FPS
sample_times = np.arange(start_time, end_time + time_interval/2, time_interval)

frames_to_analyze = []
for sample_time in sample_times:
    time_diffs = np.abs(frame_times - sample_time)
    closest_idx = np.argmin(time_diffs)
    frame = tracking_home.index[closest_idx]
    
    if frame not in tracking_away.index:
        continue
    
    actual_time = tracking_home.loc[frame, 'Time [s]']
    frames_to_analyze.append(frame)

print(f"Analyzing {len(frames_to_analyze)} frames at {TARGET_FPS} FPS")
print()

# =============================================================================
# PITCH CONTROL CALCULATION FUNCTION
# =============================================================================

def calculate_pitch_control_surface(home_row, away_row, params, GK_numbers):
    """Calculate pitch control surface for given player positions"""
    
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
    PC_baseline, xgrid, ygrid = calculate_pitch_control_surface(
        home_t_backfilled, away_t_backfilled, params, GK_numbers
    )
    
    # 2. Actual: PC at time t+1 (all players moved)
    PC_actual, _, _ = calculate_pitch_control_surface(
        home_t1_backfilled, away_t1_backfilled, params, GK_numbers
    )
    
    ΔPC_actual = PC_actual - PC_baseline
    
    # 3. Identify players in attacking half
    # Home team attacks right (positive x)
    attacking_half_threshold = 0.0
    
    home_player_cols = [c for c in home_t.keys() 
                        if c[-2:].lower()=='_x' and c!='ball_x' 
                        and 'visibility' not in c.lower()]
    
    attacking_players = []
    for col in home_player_cols:
        player_id = col.replace('_x', '')
        x_pos = home_t[col]
        
        # Only include if in attacking half (x > 0)
        if x_pos > attacking_half_threshold:
            attacking_players.append(player_id)
    
    # 4. Calculate influence for each attacking player
    player_influences = {}
    
    for player_id in attacking_players:
        # Create hybrid frame: only this player moves to t+1
        home_hybrid = home_t_backfilled.copy()
        
        # Update only this player's position to t+1
        home_hybrid[f'{player_id}_x'] = home_t1_backfilled[f'{player_id}_x']
        home_hybrid[f'{player_id}_y'] = home_t1_backfilled[f'{player_id}_y']
        
        # Update velocities if they exist
        if f'{player_id}_vx' in home_t1_backfilled.keys():
            home_hybrid[f'{player_id}_vx'] = home_t1_backfilled[f'{player_id}_vx']
            home_hybrid[f'{player_id}_vy'] = home_t1_backfilled[f'{player_id}_vy']
            home_hybrid[f'{player_id}_speed'] = home_t1_backfilled[f'{player_id}_speed']
        
        # Calculate PC with only this player moved
        PC_only_this_player, _, _ = calculate_pitch_control_surface(
            home_hybrid, away_t_backfilled, params, GK_numbers
        )
        
        # Calculate influence
        ΔPC_this_player = PC_only_this_player - PC_baseline
        
        # Store influence
        player_influences[player_id] = {
            'delta_PC': ΔPC_this_player,
            'total_influence': np.sum(np.abs(ΔPC_this_player)),
            'positive_influence': np.sum(ΔPC_this_player[ΔPC_this_player > 0]),
            'negative_influence': np.sum(ΔPC_this_player[ΔPC_this_player < 0]),
            'position_t': (home_t[f'{player_id}_x'], home_t[f'{player_id}_y']),
            'position_t1': (home_t1[f'{player_id}_x'], home_t1[f'{player_id}_y']),
        }
    
    # 5. Calculate sum of attributions
    if len(player_influences) > 0:
        ΔPC_sum = np.sum([p['delta_PC'] for p in player_influences.values()], axis=0)
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
# SUMMARY STATISTICS
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
                'frames': 0
            }
        
        player_total_influences[player_id]['total'] += influence_data['total_influence']
        player_total_influences[player_id]['positive'] += influence_data['positive_influence']
        player_total_influences[player_id]['negative'] += influence_data['negative_influence']
        player_total_influences[player_id]['frames'] += 1

# Sort by total influence
sorted_players = sorted(player_total_influences.items(), 
                       key=lambda x: x[1]['total'], 
                       reverse=True)

print("PLAYER INFLUENCE RANKINGS (Attacking Half Only)")
print("=" * 70)
print(f"{'Player':<25} {'Total':<12} {'Positive':<12} {'Negative':<12} {'Frames':<8}")
print("-" * 70)

for player_id, stats in sorted_players:
    player_name = player_name_map.get(player_id, player_id)
    print(f"{player_name:<25} {stats['total']:>11.3f} {stats['positive']:>11.3f} "
          f"{stats['negative']:>11.3f} {stats['frames']:>7}")

print()

# Interaction analysis
total_interaction = np.sum([np.sum(np.abs(r['interaction'])) for r in influence_results])
total_actual_change = np.sum([np.sum(np.abs(r['ΔPC_actual'])) for r in influence_results])
interaction_percentage = (total_interaction / total_actual_change * 100) if total_actual_change > 0 else 0

print(f"Total actual ΔPC:        {total_actual_change:.3f}")
print(f"Total interaction term:  {total_interaction:.3f}")
print(f"Interaction percentage:  {interaction_percentage:.1f}%")
print()

if interaction_percentage < 10:
    print("✓ Low interaction - linear approximation is good!")
elif interaction_percentage < 20:
    print("~ Moderate interaction - some nonlinear effects")
else:
    print("! High interaction - significant player coupling")

print()

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("Saving results...")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Save detailed results as numpy file
results_file = os.path.join(OUTPUT_DIR, 'player_influence_results.npz')
np.savez(results_file, 
         influence_results=influence_results,
         player_total_influences=player_total_influences,
         sorted_players=sorted_players,
         player_name_map=player_name_map)

print(f"Results saved to: {results_file}")
print()

# =============================================================================
# GENERATE TOP PLAYER INFLUENCE HEATMAPS
# =============================================================================

print("Generating influence heatmaps for top 5 players...")

# Get top 5 players
top_5_players = sorted_players[:min(5, len(sorted_players))]

for player_id, stats in top_5_players:
    # Aggregate influence across all frames
    player_total_delta_PC = np.zeros_like(influence_results[0]['PC_baseline'])
    frame_count = 0
    
    for result in influence_results:
        if player_id in result['player_influences']:
            player_total_delta_PC += result['player_influences'][player_id]['delta_PC']
            frame_count += 1
    
    if frame_count == 0:
        continue
    
    # Average influence
    player_avg_delta_PC = player_total_delta_PC / frame_count
    
    # Create figure
    fig, ax = mviz.plot_pitch(field_dimen=(106., 68.))
    fig.set_size_inches(10, 6.5)
    
    # Plot influence heatmap
    vmax = max(abs(np.min(player_avg_delta_PC)), abs(np.max(player_avg_delta_PC)))
    im = ax.imshow(
        player_avg_delta_PC,
        extent=(-53, 53, -34, 34),
        interpolation='spline36',
        vmin=-vmax, vmax=vmax,
        cmap='seismic',
        alpha=0.8,
        origin='lower'
    )
    
    # Get player name
    player_name = player_name_map.get(player_id, player_id)
    
    # Title
    ax.set_title(f"{player_name} - Pitch Control Influence\n"
                f"Total: {stats['total']:.3f} | Positive: {stats['positive']:.3f} | "
                f"Negative: {stats['negative']:.3f}",
                fontsize=12, pad=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar.set_label('Average ΔPC (Red=Increased Control | Blue=Decreased Control)', fontsize=9)
    
    # Save
    output_file = os.path.join(OUTPUT_DIR, f'player_influence_{player_id}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

print()
print("=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print()
print("Files generated:")
print(f"  1. {results_file}")
print(f"  2. Top 5 player influence heatmaps in {OUTPUT_DIR}/")
print()
print("Next steps:")
print("  - Review player rankings to identify key contributors")
print("  - Examine influence heatmaps to see spatial patterns")
print("  - Analyze interaction term to understand player coordination")
