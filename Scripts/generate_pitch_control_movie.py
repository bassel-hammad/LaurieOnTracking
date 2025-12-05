#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Pitch Control Movie

This script generates an animated movie showing pitch control evolution 
during a specific sequence.

Based on LaurieOnTracking's pitch control model.

Usage:
  - Run the script
  - Enter game ID (e.g., 10517 for World Cup Final 2022)
  - Enter sequence number to analyze
  - Movie is saved to Metrica_Output folder
"""

import sys
import os
# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Metrica_IO as mio
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

print("=" * 70)
print("PITCH CONTROL MOVIE GENERATOR")
print("Visualizing spatial control evolution during a play")
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

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data...")
# Read in the event data
events = mio.read_event_data(DATADIR, game_id)

# Read in tracking data
tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')

# Convert positions to meters
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

# Reverse direction of play in the second half
tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

print(f"Data loaded: {len(events)} events, {len(tracking_home):,} tracking frames")
print()

# Check if PFF speed columns exist and use hybrid velocity calculation
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
print()

# =============================================================================
# SELECT SEQUENCE
# =============================================================================

# Get sequence number from user
available_sequences = events['Sequence'].dropna().unique()
available_sequences = sorted([int(s) for s in available_sequences if not pd.isna(s)])
print(f"Available sequences: {available_sequences}")
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

# Get frame window from sequence events
start_frame = int(sequence_events['Start Frame'].min())
end_frame = int(sequence_events['End Frame'].max())

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
print(f"  Frame range: {start_frame} to {end_frame}")
print(f"  Time window: {start_time:.1f}s to {end_time:.1f}s")
print()

# Show first few events
print("First few events in this sequence:")
print(sequence_events[['Team', 'Type', 'From', 'To', 'Start Frame', 'Start Time [s]']].head(10).to_string(index=False))
print()

# Determine which team is attacking in this sequence (majority of events)
team_counts = sequence_events['Team'].value_counts()
attacking_team = team_counts.idxmax()
print(f"Detected attacking team: {attacking_team}")
print()

# =============================================================================
# SETUP FOR PITCH CONTROL
# =============================================================================

params = mpc.default_model_params()
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
print(f"Goalkeepers: Home #{GK_numbers[0]}, Away #{GK_numbers[1]}")
print()

# Movie filename based on game and sequence
MOVIE_FILENAME = f"pitch_control_game{game_id}_seq{int(sequence_number)}.mp4"

# =============================================================================
# GET FRAMES TO ANALYZE
# =============================================================================

# Get all frames in the sequence
sequence_tracking = tracking_home[(tracking_home.index >= start_frame) & 
                                   (tracking_home.index <= end_frame)]

print(f"Total frames in sequence: {len(sequence_tracking)}")

# Subsample to target FPS (tracking is ~4 FPS, target 5 FPS for smooth movie)
TARGET_FPS = 5
tracking_fps = 1.0 / (sequence_tracking['Time [s]'].diff().median())
print(f"Tracking data FPS: {tracking_fps:.2f}")

# Use every frame since tracking is already ~4 FPS
frames_to_analyze = []
for frame_idx in sequence_tracking.index:
    time_val = sequence_tracking.loc[frame_idx, 'Time [s]']
    frames_to_analyze.append({
        'frame': frame_idx,
        'actual_time': time_val
    })

print(f"Frames to analyze: {len(frames_to_analyze)}")
print()

# =============================================================================
# GENERATE PITCH CONTROL SURFACES
# =============================================================================

print("Generating pitch control surfaces...")
pitch_control_data = []

for i, frame_info in enumerate(frames_to_analyze):
    frame = frame_info['frame']
    actual_time = frame_info['actual_time']
    
    print(f"  Frame {i+1}/{len(frames_to_analyze)}: t={actual_time:.1f}s (frame {frame})", end='')
    
    try:
        # Get tracking data for this frame
        home_row = tracking_home.loc[frame]
        away_row = tracking_away.loc[frame]
        
        ball_pos = np.array([home_row['ball_x'], home_row['ball_y']])
        
        # Generate pitch control surface
        field_dimen = (106., 68.)
        n_grid_cells_x = 50
        n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
        dx = field_dimen[0] / n_grid_cells_x
        dy = field_dimen[1] / n_grid_cells_y
        xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0]/2. + dx/2.
        ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1]/2. + dy/2.
        
        # Initialize pitch control grids
        PPCFa = np.zeros(shape=(len(ygrid), len(xgrid)))
        PPCFd = np.zeros(shape=(len(ygrid), len(xgrid)))
        
        # Initialize players
        home_row_backfilled = mpc._row_with_backfilled_velocities(tracking_home, frame)
        away_row_backfilled = mpc._row_with_backfilled_velocities(tracking_away, frame)
        
        if attacking_team == 'Home':
            attacking_players = mpc.initialise_players(home_row_backfilled, 'Home', params, GK_numbers[0], is_attacking=True)
            defending_players = mpc.initialise_players(away_row_backfilled, 'Away', params, GK_numbers[1], is_attacking=False)
        else:
            defending_players = mpc.initialise_players(home_row_backfilled, 'Home', params, GK_numbers[0], is_attacking=False)
            attacking_players = mpc.initialise_players(away_row_backfilled, 'Away', params, GK_numbers[1], is_attacking=True)
        
        # Calculate pitch control at each grid location
        for ii in range(len(ygrid)):
            for jj in range(len(xgrid)):
                target_position = np.array([xgrid[jj], ygrid[ii]])
                PPCFa[ii, jj], PPCFd[ii, jj] = mpc.calculate_pitch_control_at_target(
                    target_position, attacking_players, defending_players, ball_pos, params
                )
        
        pitch_control_data.append({
            'frame': frame,
            'time': actual_time,
            'PPCF': PPCFa,
            'xgrid': xgrid,
            'ygrid': ygrid,
            'home_row': home_row,
            'away_row': away_row,
            'ball_pos': ball_pos
        })
        
        print(" ✓")
        
    except Exception as e:
        print(f" ✗ Error: {e}")
        continue

print()
print(f"Successfully generated {len(pitch_control_data)} pitch control frames")
print()

# =============================================================================
# CREATE MOVIE
# =============================================================================

if len(pitch_control_data) == 0:
    print("ERROR: No pitch control frames generated!")
    sys.exit(1)

print("Creating movie...")

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

output_path = os.path.join(OUTPUT_DIR, MOVIE_FILENAME)

# Set up the figure
fig, ax = mviz.plot_pitch(field_dimen=(106., 68.))
fig.set_size_inches(12, 8)

# Initialize plot elements
pitch_control_plot = None
time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                    fontsize=14, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def animate(frame_idx):
    """Animation function called for each frame"""
    global pitch_control_plot
    
    data = pitch_control_data[frame_idx]
    
    # Clear previous pitch control
    if pitch_control_plot is not None:
        pitch_control_plot.remove()
    
    # Plot pitch control surface
    pitch_control_plot = ax.imshow(
        data['PPCF'], 
        extent=(-53, 53, -34, 34),
        interpolation='spline36',
        vmin=0.0, vmax=1.0, 
        cmap='bwr',
        alpha=0.5,
        origin='lower'
    )
    
    # Remove old player markers
    for artist in ax.lines + ax.collections:
        artist.remove()
    
    # Plot players
    home_row = data['home_row']
    away_row = data['away_row']
    
    # Home team (red)
    x_cols_home = [c for c in home_row.keys() if c[-2:].lower()=='_x' and c!='ball_x' and 'visibility' not in c.lower()]
    y_cols_home = [c for c in home_row.keys() if c[-2:].lower()=='_y' and c!='ball_y' and 'visibility' not in c.lower()]
    ax.plot(home_row[x_cols_home], home_row[y_cols_home], 'ro', markersize=10, alpha=0.7)
    
    # Away team (blue)
    x_cols_away = [c for c in away_row.keys() if c[-2:].lower()=='_x' and c!='ball_x' and 'visibility' not in c.lower()]
    y_cols_away = [c for c in away_row.keys() if c[-2:].lower()=='_y' and c!='ball_y' and 'visibility' not in c.lower()]
    ax.plot(away_row[x_cols_away], away_row[y_cols_away], 'bo', markersize=10, alpha=0.7)
    
    # Ball
    ball_pos = data['ball_pos']
    ax.plot(ball_pos[0], ball_pos[1], 'ko', markersize=8, alpha=1.0, linewidth=0, zorder=10)
    
    # Update time text
    time_str = f"Time: {data['time']:.1f}s | Sequence {int(sequence_number)}"
    time_text.set_text(time_str)
    
    return [pitch_control_plot, time_text]

# Create animation
print(f"  Generating animation with {len(pitch_control_data)} frames...")
anim = animation.FuncAnimation(
    fig, 
    animate, 
    frames=len(pitch_control_data),
    interval=200,  # 200ms between frames (5 FPS)
    blit=False,
    repeat=True
)

# Add title
fig.suptitle(f"Pitch Control: Game {game_id} - Sequence {int(sequence_number)} ({attacking_team} attacking)", fontsize=16, y=0.98)

# Add colorbar
cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=0, vmax=1)),
    ax=ax,
    orientation='horizontal',
    pad=0.05,
    shrink=0.8
)
cbar.set_label(f'Pitch Control Probability (Red=Home, Blue=Away)', fontsize=10)

# Save animation
print(f"  Saving movie to: {output_path}")
print(f"  Note: This may take a while with {len(pitch_control_data)} frames...")

# Calculate FPS to match real-time playback
# Video duration should equal game time duration
game_duration = end_time - start_time
num_frames = len(pitch_control_data)
movie_fps = num_frames / game_duration if game_duration > 0 else 5
print(f"  Game duration: {game_duration:.1f}s, Frames: {num_frames}, Playback FPS: {movie_fps:.1f}")

writer = animation.FFMpegWriter(
    fps=movie_fps,
    metadata=dict(artist='LaurieOnTracking'),
    bitrate=5000
)

try:
    anim.save(output_path, writer=writer, dpi=150)
    print()
    print("=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print(f"Movie saved to: {output_path}")
    print(f"Frames: {len(pitch_control_data)} tracking frames")
    print(f"Duration: {end_time - start_time:.1f} seconds of game time")
    print(f"Playback FPS: {movie_fps}")
    print(f"Resolution: 1800x1200 (150 DPI)")
    print()
    print("The movie shows:")
    print("  - Red areas = Home team controls space")
    print("  - Blue areas = Away team controls space")
    print("  - Red dots = Home players")
    print("  - Blue dots = Away players")
    print("  - Black dot = Ball")
    print()
    
except Exception as e:
    print()
    print(f"ERROR: Failed to save movie: {e}")
    print()
    print("Note: This script requires FFMpeg to be installed.")
    print("To install FFMpeg:")
    print("  - Windows: Download from https://ffmpeg.org/ or use 'choco install ffmpeg'")
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
