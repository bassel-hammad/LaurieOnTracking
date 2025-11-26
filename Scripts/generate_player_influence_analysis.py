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

print("Calculating player velocities...")
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
    
    # 3. Identify all home team players
    home_player_cols = [c for c in home_t.keys() 
                        if c[-2:].lower()=='_x' and c!='ball_x' 
                        and 'visibility' not in c.lower()]
    
    attacking_players = []
    for col in home_player_cols:
        player_id = col.replace('_x', '')
        x_pos = home_t[col]
        
        # Include all players (not just attacking half)
        if not pd.isna(x_pos):
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

print("PLAYER INFLUENCE RANKINGS (All Home Team Players)")
print("=" * 70)
print(f"{'Rank':<6} {'Player':<25} {'Total':<12} {'Positive':<12} {'Negative':<12} {'Frames':<8}")
print("-" * 70)

for rank, (player_id, stats) in enumerate(sorted_players, 1):
    player_name = player_name_map.get(player_id, player_id)
    print(f"{rank:<6} {player_name:<25} {stats['total']:>11.3f} {stats['positive']:>11.3f} "
          f"{stats['negative']:>11.3f} {stats['frames']:>7}")

print()
print(f"Total players analyzed: {len(sorted_players)}")
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

movie_filename = f'sequence_{int(sequence_number)}_pitch_control.mp4'
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
    
    home_row = mpc._row_with_backfilled_velocities(tracking_home, frame)
    away_row = mpc._row_with_backfilled_velocities(tracking_away, frame)
    
    PPCF, xgrid, ygrid = calculate_pitch_control_surface(home_row, away_row, params, GK_numbers)
    
    pitch_control_data.append({
        'PPCF': PPCF,
        'xgrid': xgrid,
        'ygrid': ygrid,
        'home_row': home_row,
        'away_row': away_row,
        'ball_pos': np.array([home_row['ball_x'], home_row['ball_y']]),
        'time': home_row['Time [s]'],
        'frame': frame
    })

print()
print("Creating animation...")

# Get top 5 players for bar chart
top_5_players = sorted_players[:5]
top_5_names = [player_name_map.get(p[0], p[0]) for p in top_5_players]
top_5_influences = [p[1]['total'] for p in top_5_players]
top_5_ids = [p[0] for p in top_5_players]

# Create player ranking map for annotations
player_rankings = {p[0]: rank for rank, p in enumerate(sorted_players, 1)}

# Create figure with pitch on top and bar chart below
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 0.1], hspace=0.3)

# Pitch subplot
plt.subplot(gs[0])
fig_pitch, ax_pitch = mviz.plot_pitch(field_dimen=(106., 68.))
plt.close(fig_pitch)  # Close the figure created by plot_pitch
ax_pitch = fig.add_subplot(gs[0])
# Manually draw pitch on ax_pitch
ax_pitch.set_xlim(-53, 53)
ax_pitch.set_ylim(-34, 34)
ax_pitch.set_aspect('equal')
ax_pitch.axis('off')
# Draw pitch lines
from matplotlib.patches import Rectangle, Arc, Circle
ax_pitch.plot([-52.5, 52.5], [-34, -34], 'k-', linewidth=2)
ax_pitch.plot([-52.5, 52.5], [34, 34], 'k-', linewidth=2)
ax_pitch.plot([-52.5, -52.5], [-34, 34], 'k-', linewidth=2)
ax_pitch.plot([52.5, 52.5], [-34, 34], 'k-', linewidth=2)
ax_pitch.plot([0, 0], [-34, 34], 'k-', linewidth=2)
ax_pitch.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor='k', linewidth=2))

# Bar chart subplot
ax_bar = fig.add_subplot(gs[1])
bars = ax_bar.barh(range(len(top_5_names)), top_5_influences, color='crimson', alpha=0.7)
ax_bar.set_yticks(range(len(top_5_names)))
ax_bar.set_yticklabels([f"{i+1}. {name}" for i, name in enumerate(top_5_names)])
ax_bar.set_xlabel('Total Influence (Pitch Control Change)', fontsize=10)
ax_bar.set_title('Top 5 Most Influential Players', fontsize=11, fontweight='bold')
ax_bar.invert_yaxis()
ax_bar.grid(axis='x', alpha=0.3)

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, top_5_influences)):
    ax_bar.text(val, i, f' {val:.2f}', va='center', fontsize=9)

ax = ax_pitch

# Initialize pitch control plot
data = pitch_control_data[0]
pitch_control_plot = ax.imshow(
    data['PPCF'],
    extent=(-53, 53, -34, 34),
    interpolation='spline36',
    vmin=0.0, vmax=1.0,
    cmap='bwr',
    alpha=0.6,
    origin='lower'
)

# Initialize time text
time_text = ax.text(
    0.02, 0.98, '',
    transform=ax.transAxes,
    fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

def animate(frame_idx):
    """Animation function"""
    data = pitch_control_data[frame_idx]
    
    # Update pitch control surface
    pitch_control_plot.set_data(data['PPCF'])
    
    # Clear previous player positions and text annotations
    for artist in ax.lines + ax.collections:
        artist.remove()
    
    # Clear text annotations (ranking numbers)
    for txt in ax.texts[:]:
        if txt != time_text:  # Don't remove the time text
            txt.remove()
    
    home_row = data['home_row']
    away_row = data['away_row']
    
    # Home team (red) with ranking annotations
    x_cols_home = [c for c in home_row.keys() if c[-2:].lower()=='_x' and c!='ball_x' and 'visibility' not in c.lower()]
    y_cols_home = [c for c in home_row.keys() if c[-2:].lower()=='_y' and c!='ball_y' and 'visibility' not in c.lower()]
    ax.plot(home_row[x_cols_home], home_row[y_cols_home], 'ro', markersize=10, alpha=0.7)
    
    # Add ranking numbers to players
    for x_col, y_col in zip(x_cols_home, y_cols_home):
        player_id = x_col.replace('_x', '')
        if player_id in player_rankings and player_rankings[player_id] <= 5:  # Only top 5
            x_pos = home_row[x_col]
            y_pos = home_row[y_col]
            if not pd.isna(x_pos) and not pd.isna(y_pos):
                rank = player_rankings[player_id]
                ax.text(x_pos, y_pos, str(rank), 
                       fontsize=8, fontweight='bold', color='white',
                       ha='center', va='center', zorder=11)
    
    # Away team (blue)
    x_cols_away = [c for c in away_row.keys() if c[-2:].lower()=='_x' and c!='ball_x' and 'visibility' not in c.lower()]
    y_cols_away = [c for c in away_row.keys() if c[-2:].lower()=='_y' and c!='ball_y' and 'visibility' not in c.lower()]
    ax.plot(away_row[x_cols_away], away_row[y_cols_away], 'bo', markersize=10, alpha=0.7)
    
    # Ball
    ball_pos = data['ball_pos']
    ax.plot(ball_pos[0], ball_pos[1], 'ko', markersize=8, alpha=1.0, linewidth=0, zorder=10)
    
    # Update time text
    time_str = f"Time: {data['time']:.1f}s | Frame: {frame_idx+1}/{len(pitch_control_data)}"
    time_text.set_text(time_str)
    
    return [pitch_control_plot, time_text]

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
fig.suptitle(f"Pitch Control & Player Influence - Sequence {int(sequence_number)}", fontsize=16, y=0.96)

# Add colorbar using the colorbar subplot area
cbar_ax = fig.add_subplot(gs[2])
cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=0, vmax=1)),
    cax=cbar_ax,
    orientation='horizontal'
)
cbar.set_label('Pitch Control Probability (Red=Home, Blue=Away)', fontsize=10)

# Save animation
print(f"Saving movie to: {output_path}")
print("Note: This may take a while...")

writer = animation.FFMpegWriter(
    fps=TARGET_FPS,
    metadata=dict(artist='LaurieOnTracking'),
    bitrate=5000
)

try:
    anim.save(output_path, writer=writer, dpi=150)
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
