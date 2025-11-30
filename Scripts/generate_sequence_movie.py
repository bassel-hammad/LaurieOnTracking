#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Movie for Any Sequence

This script generates a movie for a specified game ID and sequence number.
It shows player movements and ball position during the sequence.

Usage:
    python generate_sequence_movie.py
    (prompts for game_id and sequence_number)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Metrica_IO as mio
import Metrica_Viz as mviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

print("=" * 70)
print("SEQUENCE MOVIE GENERATOR")
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
events = mio.read_event_data(DATADIR, game_id)
tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')

tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

print(f"Data loaded: {len(events)} events, {len(tracking_home):,} tracking frames")
print()

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

# Get frame window from sequence events
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

# Determine which team is attacking in this sequence
team_counts = sequence_events['Team'].value_counts()
attacking_team = team_counts.idxmax()
print(f"Attacking team in this sequence: {attacking_team}")
print()

# =============================================================================
# PREPARE FRAMES FOR MOVIE
# =============================================================================

print(f"Preparing frames from {start_time:.1f}s to {end_time:.1f}s")

# Subsample frames to 5 FPS for smooth playback
TARGET_FPS = 5
frame_times = tracking_home['Time [s]'].values
time_interval = 1.0 / TARGET_FPS
sample_times = np.arange(start_time, end_time + time_interval/2, time_interval)

print(f"Sampling at {TARGET_FPS} FPS: {len(sample_times)} frames")
print()

# Find tracking frames closest to each sample time
movie_frames = []

for sample_time in sample_times:
    time_diffs = np.abs(frame_times - sample_time)
    closest_idx = np.argmin(time_diffs)
    frame = tracking_home.index[closest_idx]
    
    if frame not in tracking_away.index:
        continue
    
    home_row = tracking_home.loc[frame]
    away_row = tracking_away.loc[frame]
    actual_time = home_row['Time [s]']
    ball_pos = np.array([home_row['ball_x'], home_row['ball_y']])
    
    movie_frames.append({
        'frame': frame,
        'time': actual_time,
        'home_row': home_row,
        'away_row': away_row,
        'ball_pos': ball_pos
    })

print(f"Found {len(movie_frames)} valid frames")
print()

# =============================================================================
# CREATE MOVIE
# =============================================================================

if len(movie_frames) == 0:
    print("ERROR: No frames generated!")
    sys.exit(1)

print("Creating movie...")

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

movie_filename = f'sequence_{int(sequence_number)}_game_{game_id}.mp4'
output_path = os.path.join(OUTPUT_DIR, movie_filename)

# Set up the figure
fig, ax = mviz.plot_pitch(field_dimen=(106., 68.))
fig.set_size_inches(12, 8)

# Initialize plot elements
time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                    fontsize=14, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Store references to player artists
player_artists = []

def animate(frame_idx):
    """Animation function called for each frame"""
    global player_artists
    
    data = movie_frames[frame_idx]
    
    # Remove old player markers
    for artist in player_artists:
        artist.remove()
    player_artists = []
    
    # Plot players
    home_row = data['home_row']
    away_row = data['away_row']
    
    # Home team (red)
    x_cols_home = [c for c in home_row.keys() if c[-2:].lower()=='_x' and c!='ball_x' and 'visibility' not in c.lower()]
    y_cols_home = [c for c in home_row.keys() if c[-2:].lower()=='_y' and c!='ball_y' and 'visibility' not in c.lower()]
    line = ax.plot(home_row[x_cols_home], home_row[y_cols_home], 'ro', markersize=10, alpha=0.8,
                   markeredgecolor='black', markeredgewidth=1.5)[0]
    player_artists.append(line)
    
    # Away team (blue)
    x_cols_away = [c for c in away_row.keys() if c[-2:].lower()=='_x' and c!='ball_x' and 'visibility' not in c.lower()]
    y_cols_away = [c for c in away_row.keys() if c[-2:].lower()=='_y' and c!='ball_y' and 'visibility' not in c.lower()]
    line = ax.plot(away_row[x_cols_away], away_row[y_cols_away], 'bo', markersize=10, alpha=0.8,
                   markeredgecolor='black', markeredgewidth=1.5)[0]
    player_artists.append(line)
    
    # Ball
    ball_pos = data['ball_pos']
    ball_artist = ax.plot(ball_pos[0], ball_pos[1], 'o', color='white', markersize=6, 
                         markeredgecolor='black', markeredgewidth=1.5, zorder=10)[0]
    player_artists.append(ball_artist)
    
    # Update time text
    time_str = f"Time: {data['time']:.1f}s | Frame: {frame_idx+1}/{len(movie_frames)}"
    time_text.set_text(time_str)
    
    return player_artists + [time_text]

# Create animation
print(f"  Generating animation with {len(movie_frames)} frames...")
interval_ms = int(1000 / TARGET_FPS)

anim = animation.FuncAnimation(
    fig, 
    animate, 
    frames=len(movie_frames),
    interval=interval_ms,
    blit=False,
    repeat=True
)

# Add title
attacking_team_name = 'Home' if attacking_team == 'Home' else 'Away'
fig.suptitle(f"Sequence {int(sequence_number)} - Game {game_id} ({attacking_team_name} Attacking)", 
             fontsize=16, y=0.98)

# Save animation
print(f"  Saving movie to: {output_path}")
print(f"  Note: This may take a while...")

writer = animation.FFMpegWriter(
    fps=TARGET_FPS,
    metadata=dict(artist='LaurieOnTracking'),
    bitrate=5000
)

try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anim.save(output_path, writer=writer, dpi=150)
    
    print()
    print("=" * 70)
    print("MOVIE GENERATION COMPLETE!")
    print("=" * 70)
    print(f"Movie saved to: {output_path}")
    print(f"Frames: {len(movie_frames)}")
    print(f"Game time duration: {end_time - start_time:.1f} seconds")
    print(f"Movie duration: {len(movie_frames) / TARGET_FPS:.1f} seconds")
    print(f"Playback FPS: {TARGET_FPS}")
    print()
    print("The movie shows:")
    print("  - Red dots = Home players")
    print("  - Blue dots = Away players")
    print("  - White dot = Ball")
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
