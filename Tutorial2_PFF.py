#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 2: PFF Data Analysis - Player Velocities & Physical Performance
Based on "Friends of Tracking" Tutorial 2: Delving Deeper

Focuses on:
- Player velocities and physical performance 
- Video generation for goals
- Distance analysis and speed categories
- Sprint analysis

@author: Adapted for PFF Data
"""

import Metrica_IO as mio
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=" * 60)
print("***** PFF - TUTORIAL 2 *****")
print("*** PLAYER VELOCITIES & PHYSICAL PERFORMANCE ***")
print("=" * 60)

# =============================================================================
# CONFIGURATION - Change these values for different matches
# =============================================================================
DATADIR = 'Sample Data'
game_id = 10517  # Change this to your desired game ID
home_team_name = "Home Team"  # Will be determined from data
away_team_name = "Away Team"  # Will be determined from data

print(">> Loading PFF data...")

# Read in the event data
events = mio.read_event_data(DATADIR, game_id)

# Read in tracking data
print("   Loading tracking data...")
tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')

# Convert positions from metrica units to meters
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

# Determine team names from data
if 'From' in events.columns:
    home_events = events[events['Team'] == 'Home']
    away_events = events[events['Team'] == 'Away']
    home_players = home_events['From'].dropna().unique()
    away_players = away_events['From'].dropna().unique()
    if len(home_players) > 0:
        home_team_name = f"Home Team ({len(home_players)} players)"
    if len(away_players) > 0:
        away_team_name = f"Away Team ({len(away_players)} players)"

print(f"   Loaded {len(events)} events and {len(tracking_home)} tracking frames")

# Reverse direction of play in the second half so that home team is always attacking from right->left
tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

print("=" * 60)
print(">> VIDEO GENERATION")
print("=" * 60)

# Find goals for video generation
goals = events[events['Subtype'].str.contains('GOAL', na=False)]
print(f"Found {len(goals)} goals in PFF:")

for i, (goal_idx, goal) in enumerate(goals.iterrows(), 1):
    team = "[HOME]" if goal['Team'] == 'Home' else "[AWAY]"
    team_name = home_team_name if goal['Team'] == 'Home' else away_team_name
    player_name = goal.get('From', 'Unknown')
    # Convert tracking time to game minutes (tracking time is cumulative across periods)
    minute = goal.get('Start Time [s]', 0) / 60
    print(f"   Goal {i}: {team} {team_name} - {player_name} - {minute:.1f}'")

# Generate video for second goal (if available)
PLOTDIR = DATADIR

if len(goals) >= 2:
    second_goal = goals.iloc[1]
    goal_frame = int(second_goal['Start Frame'])
    player_name = second_goal.get('From', 'Unknown')
    
    print(f"\n>> Generating video for second goal ({player_name})...")
    print(f"   Goal frame: {goal_frame}")
    print(f"   Player: {player_name}")
    
    try:
        # Check if frame is available in tracking data (use actual frame numbers, not indices)
        if goal_frame in tracking_home.index:
            print("   Generating 20-second video around second goal...")
            
            # Get goal time and create 20-second time window
            goal_time = tracking_home.loc[goal_frame, 'Time [s]']
            start_time = goal_time - 10  # 10 seconds before
            end_time = goal_time + 10    # 10 seconds after
            
            print(f"   Time window: {start_time:.1f}s to {end_time:.1f}s (centered on goal at {goal_time:.1f}s)")
            
            # Get all frames in this time window
            home_window = tracking_home[(tracking_home['Time [s]'] >= start_time) & 
                                        (tracking_home['Time [s]'] <= end_time)].copy()
            away_window = tracking_away[(tracking_away['Time [s]'] >= start_time) & 
                                        (tracking_away['Time [s]'] <= end_time)].copy()
            
            print(f"   Found {len(home_window)} frames in time window (variable spacing)")
            
            # Resample to uniform 25 FPS for smooth video
            target_fps = 25
            uniform_times = np.linspace(start_time, end_time, int((end_time - start_time) * target_fps))
            
            # Interpolate positions to uniform time points
            home_clip = pd.DataFrame(index=range(len(uniform_times)))
            away_clip = pd.DataFrame(index=range(len(uniform_times)))
            
            home_clip['Time [s]'] = uniform_times
            away_clip['Time [s]'] = uniform_times
            home_clip['Period'] = home_window['Period'].iloc[0]
            away_clip['Period'] = away_window['Period'].iloc[0]
            
            # Interpolate player positions
            for col in home_window.columns:
                if col.endswith('_x') or col.endswith('_y') or col == 'ball_x' or col == 'ball_y':
                    home_clip[col] = np.interp(uniform_times, home_window['Time [s]'], home_window[col])
            
            for col in away_window.columns:
                if col.endswith('_x') or col.endswith('_y') or col == 'ball_x' or col == 'ball_y':
                    away_clip[col] = np.interp(uniform_times, away_window['Time [s]'], away_window[col])
            
            print(f"   Resampled to {len(home_clip)} frames at uniform {target_fps} FPS for smooth playback")
            
            # Fixed video generation (handles matplotlib plotting issues)
            try:
                mviz.save_match_clip(
                    home_clip, away_clip,
                    PLOTDIR, fname=f'pff_second_goal_{game_id}',
                    include_player_velocities=False,
                    team_colors=('b', 'r')  # Home blue, Away red
                )
                print(f"   ✅ Video saved: pff_second_goal_{game_id}.mp4")
            except Exception as plot_error:
                print(f"   ⚠️  Video generation had plotting issues but likely succeeded: {plot_error}")
                # Check if file was created despite the error
                import os
                if os.path.exists(f"{PLOTDIR}/pff_second_goal_{game_id}.mp4"):
                    print("   ✅ Video file found - generation successful!")
                else:
                    print("   ❌ Video file not created")
        else:
            print(f"   ❌ Goal frame {goal_frame} not available in tracking data")
            print(f"   Available frame range: {tracking_home.index.min()} to {tracking_home.index.max()}")
    except Exception as e:
        print(f"   ❌ Video generation failed: {e}")
else:
    print("   ❌ Second goal not found in events data")

print("=" * 60)
print(">> VELOCITY CALCULATIONS")
print("=" * 60)

print(">> Calculating player velocities...")
print("   [WARNING] Due to data sampling issues, velocity calculations may be affected by position jumps")

try:
    # Calculate player velocities with smoothing to handle data issues
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
    print("   ✅ Velocity calculations completed with smoothing")
    
    # Plot a frame with velocities (if available)
    try:
        sample_frame = min(10000, len(tracking_home) - 1)
        print(f"   Plotting frame {sample_frame} with player velocities...")
        mviz.plot_frame(tracking_home.loc[sample_frame], tracking_away.loc[sample_frame], 
                        include_player_velocities=True, annotate=True)
        plt.title("PFF - Player Velocities Sample Frame")
        plt.show()
    except Exception as e:
        print(f"   ❌ Frame plotting failed: {e}")
        
except Exception as e:
    print(f"   ❌ Velocity calculation failed: {e}")
    print("   Continuing without velocity data...")

print("=" * 60)
print(">> PHYSICAL PERFORMANCE ANALYSIS")
print("=" * 60)

# Create a Physical summary dataframe for Home team players
print(f">> Analyzing {home_team_name} physical performance...")

home_players = np.unique([c.split('_')[1] for c in tracking_home.columns if c[:4] == 'Home'])
print(f"   Found {len(home_players)} {home_team_name} players: {', '.join(home_players)}")

home_summary = pd.DataFrame(index=home_players)

# Calculate minutes played for each player
print("   Calculating minutes played...")
minutes = []
for player in home_players:
    try:
        column = 'Home_' + player + '_x'
        if column in tracking_home.columns:
            # Account for our data being ~7.5 FPS instead of 25 FPS
            first_valid = tracking_home[column].first_valid_index()
            last_valid = tracking_home[column].last_valid_index()
            if first_valid is not None and last_valid is not None:
                # Estimate FPS from time data
                time_range = tracking_home.loc[last_valid, 'Time [s]'] - tracking_home.loc[first_valid, 'Time [s]']
                frames_played = last_valid - first_valid + 1
                estimated_fps = frames_played / time_range if time_range > 0 else 7.5
                player_minutes = frames_played / estimated_fps / 60.0
            else:
                player_minutes = 0
        else:
            player_minutes = 0
    except Exception as e:
        print(f"   Warning: Could not calculate minutes for player {player}: {e}")
        player_minutes = 0
    minutes.append(player_minutes)

home_summary['Minutes Played'] = minutes
home_summary = home_summary.sort_values(['Minutes Played'], ascending=False)

print(f"   Top 5 by minutes played:")
for player in home_summary.head(5).index:
    mins = home_summary.loc[player, 'Minutes Played']
    print(f"      Player {player}: {mins:.1f} minutes")

# Calculate total distance covered (if speed data available)
if any('_speed' in col for col in tracking_home.columns):
    print("   Calculating distance covered...")
    distance = []
    for player in home_summary.index:
        try:
            speed_column = 'Home_' + player + '_speed'
            if speed_column in tracking_home.columns:
                # Use actual time intervals for distance calculation
                valid_speeds = tracking_home[speed_column].dropna()
                if len(valid_speeds) > 0:
                    # Estimate distance using time intervals
                    time_diffs = tracking_home.loc[valid_speeds.index, 'Time [s]'].diff().fillna(0.133)  # Default to ~7.5 FPS
                    player_distance = (valid_speeds * time_diffs).sum() / 1000  # Convert to km
                else:
                    player_distance = 0
            else:
                player_distance = 0
        except Exception as e:
            print(f"   Warning: Could not calculate distance for player {player}: {e}")
            player_distance = 0
        distance.append(player_distance)
    
    home_summary['Distance [km]'] = distance
    
    # Plot distance covered
    if home_summary['Distance [km]'].sum() > 0:
        print("   Plotting distance covered...")
        plt.figure(figsize=(12, 6))
        ax = home_summary['Distance [km]'].plot.bar(rot=45)
        ax.set_xlabel(f'{home_team_name} Player (Jersey Number)')
        ax.set_ylabel('Distance covered [km]')
        ax.set_title(f'PFF - Distance Covered by {home_team_name} Players')
        plt.tight_layout()
        plt.show()
        
        print(f"   Top 3 by distance:")
        top_distance = home_summary.nlargest(3, 'Distance [km]')
        for player in top_distance.index:
            dist = home_summary.loc[player, 'Distance [km]']
            print(f"      Player {player}: {dist:.2f} km")
    else:
        print("   ❌ No valid distance data calculated")
else:
    print("   ❌ No speed data available for distance calculations")

# Speed category analysis (if speed data available)
if any('_speed' in col for col in tracking_home.columns):
    print("   Analyzing speed categories...")
    
    walking = []
    jogging = []
    running = []
    sprinting = []
    
    for player in home_summary.index:
        try:
            speed_column = 'Home_' + player + '_speed'
            if speed_column in tracking_home.columns:
                valid_data = tracking_home[tracking_home[speed_column].notna()]
                
                # Walking (less than 2 m/s)
                walking_data = valid_data[valid_data[speed_column] < 2]
                walking_dist = (walking_data[speed_column] * walking_data['Time [s]'].diff().fillna(0.133)).sum() / 1000
                
                # Jogging (between 2 and 4 m/s)
                jogging_data = valid_data[(valid_data[speed_column] >= 2) & (valid_data[speed_column] < 4)]
                jogging_dist = (jogging_data[speed_column] * jogging_data['Time [s]'].diff().fillna(0.133)).sum() / 1000
                
                # Running (between 4 and 7 m/s)
                running_data = valid_data[(valid_data[speed_column] >= 4) & (valid_data[speed_column] < 7)]
                running_dist = (running_data[speed_column] * running_data['Time [s]'].diff().fillna(0.133)).sum() / 1000
                
                # Sprinting (greater than 7 m/s)
                sprinting_data = valid_data[valid_data[speed_column] >= 7]
                sprinting_dist = (sprinting_data[speed_column] * sprinting_data['Time [s]'].diff().fillna(0.133)).sum() / 1000
                
            else:
                walking_dist = jogging_dist = running_dist = sprinting_dist = 0
                
        except Exception as e:
            print(f"   Warning: Speed analysis failed for player {player}: {e}")
            walking_dist = jogging_dist = running_dist = sprinting_dist = 0
            
        walking.append(walking_dist)
        jogging.append(jogging_dist)
        running.append(running_dist)
        sprinting.append(sprinting_dist)
    
    home_summary['Walking [km]'] = walking
    home_summary['Jogging [km]'] = jogging
    home_summary['Running [km]'] = running
    home_summary['Sprinting [km]'] = sprinting
    
    # Plot speed categories if we have valid data
    total_movement = home_summary[['Walking [km]', 'Jogging [km]', 'Running [km]', 'Sprinting [km]']].sum().sum()
    if total_movement > 0:
        print("   Plotting speed category analysis...")
        plt.figure(figsize=(14, 8))
        ax = home_summary[['Walking [km]', 'Jogging [km]', 'Running [km]', 'Sprinting [km]']].plot.bar(
            stacked=True, colormap='RdYlBu_r')
        ax.set_xlabel(f'{home_team_name} Player (Jersey Number)')
        ax.set_ylabel('Distance covered [km]')
        ax.set_title(f'PFF - Distance by Speed Category ({home_team_name})')
        plt.xticks(rotation=45)
        plt.legend(title='Speed Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
        print("   ❌ No valid speed category data")

print("=" * 60)
print(">> SUMMARY")
print("=" * 60)

print("✅ PFF Tutorial 2 Analysis Complete!")
print()
print("📊 Results Summary:")
print(f"   • Analyzed {len(home_players)} {home_team_name} players")
print(f"   • Physical performance metrics calculated")
if 'Distance [km]' in home_summary.columns and home_summary['Distance [km]'].sum() > 0:
    avg_distance = home_summary['Distance [km]'].mean()
    print(f"   • Average distance covered: {avg_distance:.2f} km")
print()
print("⚠️  Note: Due to PFF data sampling artifacts, some velocity-based")
print("   calculations may be affected by position discontinuities.")
print()
print("🎯 Tutorial 2 successfully adapted for PFF data!")
print()

# Display final summary table
print("📋 FINAL PERFORMANCE SUMMARY:")
print(home_summary.round(2))

# END
