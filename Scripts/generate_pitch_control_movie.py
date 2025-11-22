#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Pitch Control Movie for Di María's Goal

This script generates pitch control frames at 1-second intervals for the sequence
leading up to Di María's goal in the World Cup Final, then compiles them into a movie.

The movie shows how space control evolves during the attacking sequence.
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
from matplotlib.patches import Rectangle
import pandas as pd

print("=" * 70)
print("PITCH CONTROL MOVIE GENERATOR")
print("Di María's Goal - World Cup Final 2022")
print("=" * 70)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================
DATADIR = 'Sample Data'
game_id = 10517
OUTPUT_DIR = 'Metrica_Output'
MOVIE_FILENAME = 'dimaria_goal_pitch_control.mp4'

# Time window configuration
TIME_WINDOW_BEFORE_GOAL = 10  # seconds before the goal to start
TIME_WINDOW_AFTER_GOAL = 2    # seconds after the goal to end
USE_EVERY_FRAME = True        # Generate pitch control for EVERY tracking frame (not just 1 per second)

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

# Calculate player velocities
print("Calculating player velocities...")
tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

# Find Di María's goal (second goal)
shots = events[events['Type']=='SHOT']
goals = shots[shots['Subtype'].str.contains('GOAL', na=False)].copy()

if len(goals) < 2:
    print("ERROR: Di María's goal (second goal) not found in the data!")
    sys.exit(1)

dimaria_goal = goals.iloc[1]  # Second goal
dimaria_goal_idx = goals.index[1]
goal_frame = int(dimaria_goal['Start Frame'])
goal_scorer = dimaria_goal.get('From', 'Ángel Di María')

# Get actual goal time from tracking data using the frame number
if goal_frame not in tracking_home.index:
    print(f"ERROR: Goal frame {goal_frame} not found in tracking data!")
    sys.exit(1)

goal_time = tracking_home.loc[goal_frame, 'Time [s]']

print(f"Found Di María's goal:")
print(f"  Scorer: {goal_scorer}")
print(f"  Event frame: {goal_frame}")
print(f"  Actual time from tracking: {goal_time:.1f} seconds")
print()

# Get pitch control model parameters
params = mpc.default_model_params()

# Find goalkeepers
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
print(f"Goalkeepers: Home #{GK_numbers[0]}, Away #{GK_numbers[1]}")
print()

# =============================================================================
# GENERATE PITCH CONTROL FRAMES
# =============================================================================

# Calculate time range
start_time = max(0, goal_time - TIME_WINDOW_BEFORE_GOAL)
end_time = goal_time + TIME_WINDOW_AFTER_GOAL

print(f"Generating pitch control frames from {start_time:.1f}s to {end_time:.1f}s")

# Subsample frames to 5 FPS for pitch control generation
TARGET_FPS = 5
frame_times = tracking_home['Time [s]'].values
time_mask = (frame_times >= start_time) & (frame_times <= end_time)
all_frames_in_window = tracking_home.index[time_mask].tolist()

# Sample frames at 5 FPS intervals
time_interval = 1.0 / TARGET_FPS  # 0.2 seconds between frames
sample_times = np.arange(start_time, end_time + time_interval/2, time_interval)

print(f"Found {len(all_frames_in_window)} tracking frames in time window")
print(f"Subsampling to {TARGET_FPS} FPS: {len(sample_times)} frames")
print()

# Prepare frames to analyze
frames_to_analyze = []

for sample_time in sample_times:
    # Find closest frame to this sample time
    time_diffs = np.abs(frame_times - sample_time)
    closest_idx = np.argmin(time_diffs)
    frame = tracking_home.index[closest_idx]
    
    if frame not in tracking_away.index:
        continue
    
    actual_time = tracking_home.loc[frame, 'Time [s]']
    frames_to_analyze.append({
        'frame': frame,
        'actual_time': actual_time
    })

print(f"Found {len(frames_to_analyze)} valid frames to analyze")
print()

# Generate pitch control for each frame
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
        
        # Determine which team is attacking (assume Home = Argentina scored)
        pass_team = 'Home'  # Di María played for Argentina (Home)
        ball_pos = np.array([home_row['ball_x'], home_row['ball_y']])
        
        # Generate pitch control surface
        # We need to create pitch control similar to generate_pitch_control_for_event
        # but for an arbitrary frame
        
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
        
        if pass_team == 'Home':
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
    time_relative = data['time'] - goal_time
    if time_relative < 0:
        time_str = f"Time: {data['time']:.1f}s ({time_relative:.1f}s before goal)"
    elif time_relative == 0:
        time_str = f"Time: {data['time']:.1f}s (GOAL!)"
    else:
        time_str = f"Time: {data['time']:.1f}s (+{time_relative:.1f}s after goal)"
    
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
fig.suptitle(f"Pitch Control Evolution: {goal_scorer}'s Goal", fontsize=16, y=0.98)

# Add colorbar
cbar = plt.colorbar(
    plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=0, vmax=1)),
    ax=ax,
    orientation='horizontal',
    pad=0.05,
    shrink=0.8
)
cbar.set_label('Pitch Control Probability (Red=Home/Argentina, Blue=Away/France)', fontsize=10)

# Save animation
print(f"  Saving movie to: {output_path}")
print(f"  Note: This may take a while with {len(pitch_control_data)} frames...")

# Use 5 FPS for movie playback
movie_fps = 5

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
    print("  - Red areas = Home team (Argentina) controls space")
    print("  - Blue areas = Away team (France) controls space")
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
