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
# Use absolute paths based on project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(_PROJECT_ROOT, 'Sample Data')
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, 'Metrica_Output')

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
# IDENTIFY POSSESSION FRAMES
# =============================================================================

# Get all event frames in this sequence (frames where an event occurs)
event_frames_list = []
for _, event in sequence_events.iterrows():
    start_f = event.get('Start Frame', None)
    end_f = event.get('End Frame', None)
    if pd.notna(start_f):
        event_frames_list.append(int(start_f))
    if pd.notna(end_f) and end_f != start_f:
        event_frames_list.append(int(end_f))

event_frames_list = sorted(set(event_frames_list))
print(f"Found {len(event_frames_list)} frames with events in this sequence")
print(f"Event frames: {event_frames_list[:20]}..." if len(event_frames_list) > 20 else f"Event frames: {event_frames_list}")

def is_near_event_frame(frame, event_frames, tolerance=5):
    """
    Check if a frame is within tolerance of any event frame.
    This handles cases where our sampled frames don't exactly match event frames.
    """
    for ef in event_frames:
        if abs(frame - ef) <= tolerance:
            return True
    return False

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

# Sample at fixed time intervals to ensure consistent playback speed
TARGET_FPS = 5  # Frames per second for movie
time_interval = 1.0 / TARGET_FPS  # Fixed time between frames (0.2s)

# Generate sample times at exact intervals
sample_times = np.arange(start_time, end_time + time_interval/2, time_interval)
print(f"Target FPS: {TARGET_FPS}, Time interval: {time_interval:.3f}s")
print(f"Sample times: {len(sample_times)} frames at fixed intervals")

# Find the closest tracking frame for each sample time (no duplicates)
frame_times = tracking_home['Time [s]'].values
frames_to_analyze = []
last_frame = None

for sample_time in sample_times:
    # Find closest tracking frame to this sample time
    time_diffs = np.abs(frame_times - sample_time)
    closest_idx = np.argmin(time_diffs)
    frame = tracking_home.index[closest_idx]
    
    # Skip if this frame was already added (prevents duplicates causing slowdown)
    if frame != last_frame and frame in tracking_away.index:
        actual_time = tracking_home.loc[frame, 'Time [s]']
        frames_to_analyze.append({
            'frame': frame,
            'actual_time': actual_time
        })
        last_frame = frame

print(f"Frames to analyze: {len(frames_to_analyze)} (duplicates removed)")
print()

# =============================================================================
# GENERATE PITCH CONTROL SURFACES
# =============================================================================

print("Generating pitch control surfaces...")
pitch_control_data = []
ball_trajectory = []  # Store all ball positions for trajectory line
ball_possession = []  # Store whether ball is in possession at each frame

for i, frame_info in enumerate(frames_to_analyze):
    frame = frame_info['frame']
    actual_time = frame_info['actual_time']
    
    # Check if this frame is near an event frame (ball in possession)
    in_possession = is_near_event_frame(frame, event_frames_list, tolerance=5)
    
    print(f"  Frame {i+1}/{len(frames_to_analyze)}: t={actual_time:.1f}s (frame {frame})" + 
          (" [EVENT]" if in_possession else ""), end='')
    
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
        
        # Convert to Home team's pitch control for consistent visualization
        # PPCFa = attacking team's control, so:
        # - If Home is attacking: Home control = PPCFa
        # - If Away is attacking: Home control = PPCFd (or 1 - PPCFa)
        if attacking_team == 'Home':
            PPCF_home = PPCFa
        else:
            PPCF_home = PPCFd  # Home team is defending, so their control = PPCFd
        
        # Store ball position for trajectory
        ball_trajectory.append(ball_pos.copy())
        ball_possession.append(in_possession)
        
        pitch_control_data.append({
            'frame': frame,
            'time': actual_time,
            'PPCF': PPCF_home,  # Always show Home team's control probability
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
    
    # Draw ball trajectory: X markers for possession frames, lines between non-possession frames
    if frame_idx > 0:
        # Separate positions into possession and non-possession segments
        possession_x = []
        possession_y = []
        
        # Draw line segments only between consecutive non-possession frames
        for i in range(frame_idx + 1):
            if ball_possession[i]:
                # This is a possession frame - mark with X
                possession_x.append(ball_trajectory[i][0])
                possession_y.append(ball_trajectory[i][1])
            
            # Draw line segment from previous frame if both are non-possession
            if i > 0 and not ball_possession[i] and not ball_possession[i-1]:
                ax.plot([ball_trajectory[i-1][0], ball_trajectory[i][0]], 
                       [ball_trajectory[i-1][1], ball_trajectory[i][1]], 
                       'k-', linewidth=2, alpha=0.7, zorder=9)
        
        # Plot all possession markers as X symbols
        if possession_x:
            ax.plot(possession_x, possession_y, 'kx', markersize=10, 
                   markeredgewidth=2.5, alpha=0.9, zorder=9)
    
    # Also mark current frame with X if in possession
    if frame_idx < len(ball_possession) and ball_possession[frame_idx]:
        ball_pos = data['ball_pos']
        ax.plot(ball_pos[0], ball_pos[1], 'kx', markersize=12, 
               markeredgewidth=3, alpha=1.0, zorder=10)
    else:
        # Ball (current position) - regular dot for non-possession
        ball_pos = data['ball_pos']
        ax.plot(ball_pos[0], ball_pos[1], 'ko', markersize=8, alpha=1.0, linewidth=0, zorder=10)
    
    # Update time text
    time_str = f"Time: {data['time']:.1f}s | Sequence {int(sequence_number)}"
    time_text.set_text(time_str)
    
    return [pitch_control_plot, time_text]

# Create animation
print(f"  Generating animation with {len(pitch_control_data)} frames...")

# Use fixed interval matching TARGET_FPS for consistent playback
target_interval = int(1000 / TARGET_FPS)  # 200ms for 5 FPS
print(f"  Using fixed interval: {target_interval}ms between frames ({TARGET_FPS} FPS)")

anim = animation.FuncAnimation(
    fig, 
    animate, 
    frames=len(pitch_control_data),
    interval=target_interval,  # Fixed interval for consistent playback
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
cbar.set_label('Pitch Control: Red = Home Team | Blue = Away Team', fontsize=10)

# Save animation
print(f"  Saving movie to: {output_path}")
print(f"  Note: This may take a while with {len(pitch_control_data)} frames...")

# Use fixed FPS for consistent playback
movie_fps = TARGET_FPS
print(f"  Encoding at {movie_fps} FPS for consistent playback")

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
