#!/usr/bin/env python3

"""
Tutorial 1: PFF Data Analysis (General)
======================================

This tutorial applies Tutorial 1's analysis to PFF data,
covering both event data and tracking data analysis:

EVENT DATA:
- Goal locations and analysis
- Shot maps  
- Passing sequences leading to goals
- Basic team statistics

TRACKING DATA:
- Player trajectory plotting
- Position analysis at key moments
- Frame-by-frame visualization
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Set interactive backend for Windows
import matplotlib.pyplot as plt
import Metrica_IO as mio
import Metrica_Viz as mviz

# =============================================================================
# CONFIGURATION - Change these values for different matches
# =============================================================================
game_id = 10517  # Change this to your desired game ID
DATADIR = 'Sample Data'
home_team_name = "Home"  # Will be determined from data
away_team_name = "Away"  # Will be determined from data

print("="*50)
print("***** PFF DATA ANALYSIS *****")
print("*** COMPLETE TUTORIAL 1 REPLICATION ***")
print("="*50)

#### EVENT DATA ####
print("\n>> Loading event data...")
events = mio.read_event_data(DATADIR, game_id)

# Convert positions from 0-1 coordinates to meters
events = mio.to_metric_coordinates(events)

# Note: Direction normalization requires tracking data, skipping for event-only analysis

print(f">> Events loaded: {len(events)}")
print(f">> Event types: {events['Type'].unique()}")

# Basic match statistics
home_events = events[events['Team'] == 'Home']
away_events = events[events['Team'] == 'Away']

# Determine team names from data (if available) or use defaults
if 'From' in events.columns:
    home_players = home_events['From'].dropna().unique()
    away_players = away_events['From'].dropna().unique()
    if len(home_players) > 0:
        home_team_name = f"Home Team ({len(home_players)} players)"
    if len(away_players) > 0:
        away_team_name = f"Away Team ({len(away_players)} players)"

print(f"\n>> Match Statistics:")
print(f"[HOME] {home_team_name} events: {len(home_events)}")
print(f"[AWAY] {away_team_name} events: {len(away_events)}")

# Shot analysis
shots = events[events['Type']=='SHOT']
home_shots = shots[shots['Team'] == 'Home']
away_shots = shots[shots['Team'] == 'Away']

print(f"\n>> Shot Analysis:")
print(f"[HOME] {home_team_name} shots: {len(home_shots)}")
print(f"[AWAY] {away_team_name} shots: {len(away_shots)}")

# Goal analysis
goals = shots[shots['Subtype'].str.contains('-GOAL', na=False)]
home_goals = goals[goals['Team'] == 'Home']
away_goals = goals[goals['Team'] == 'Away']

print(f"\n>> Goal Analysis:")
print(f"[HOME] {home_team_name} goals: {len(home_goals)}")
print(f"[AWAY] {away_team_name} goals: {len(away_goals)}")
print(f">> Total goals: {len(goals)}")

print("\n>>> GOAL DETAILS:")
for i, (idx, goal) in enumerate(goals.iterrows()):
    period_name = f"Period {goal['Period']}"
    goal_time_min = goal['Start Time [s]'] / 60
    team_flag = "[HOME]" if goal['Team'] == 'Home' else "[AWAY]"
    team_name = home_team_name if goal['Team'] == 'Home' else away_team_name
    print(f"Goal {i+1}: {team_flag} {team_name} - {goal['From']} - {period_name}, {goal_time_min:.1f}' - Location: ({goal['Start X']:.1f}, {goal['Start Y']:.1f})")

# Plot all goal locations
print(f"\n>> Plotting all goal locations...")
fig, ax = mviz.plot_pitch()

goal_number = 0
for goal_idx, goal in goals.iterrows():
    goal_number += 1
    team_color = 'lightblue' if goal['Team'] == 'Home' else 'blue'
    
    # Plot goal location
    ax.plot(events.loc[goal_idx]['Start X'], events.loc[goal_idx]['Start Y'], 'o', 
            color=team_color, markersize=12, markeredgecolor='black', markeredgewidth=2)
    ax.text(events.loc[goal_idx]['Start X'], events.loc[goal_idx]['Start Y'] + 4, 
            f"{goal_number}", ha='center', fontsize=10, fontweight='bold', color='black')

plt.title("PFF - All Goal Locations")
plt.legend([f'{home_team_name} Goals', f'{away_team_name} Goals'] if len(home_goals) > 0 and len(away_goals) > 0 else 
          ([f'{home_team_name} Goals'] if len(home_goals) > 0 else [f'{away_team_name} Goals']))
plt.show()

# Plot shot map for both teams  
fig,ax = mviz.plot_pitch()
if len(home_shots) > 0:
    ax.scatter(home_shots['Start X'], home_shots['Start Y'], c='lightblue', s=100, alpha=0.7, label=home_team_name)
if len(away_shots) > 0:
    ax.scatter(away_shots['Start X'], away_shots['Start Y'], c='blue', s=100, alpha=0.7, label=away_team_name)

plt.legend()
plt.title("PFF - Shot Map")
plt.show()

print("\n" + "="*50)
print(">> DETAILED GOAL ANALYSIS")
print("="*50)

# Analyze each goal individually
goal_number = 0
for goal_idx, goal in goals.iterrows():
    goal_number += 1
    
    # Goal info
    period_name = f"Period {goal['Period']}"
    goal_time_min = goal['Start Time [s]'] / 60
    team_flag = "[HOME]" if goal['Team'] == 'Home' else "[AWAY]"
    team_name = home_team_name if goal['Team'] == 'Home' else away_team_name
    
    print(f"\n>>> GOAL {goal_number}: {team_flag} {team_name} - {goal['From']}")
    print(f">> Time: {period_name}, {goal_time_min:.1f}'")
    print(f">> Location: ({goal['Start X']:.1f}, {goal['Start Y']:.1f})")
    
    # Plot individual goal
    fig, ax = mviz.plot_pitch()
    team_color = 'lightblue' if goal['Team'] == 'Home' else 'blue'
    
    # Plot goal location with large marker
    ax.plot(events.loc[goal_idx]['Start X'], events.loc[goal_idx]['Start Y'], 'o', 
            color=team_color, markersize=20, markeredgecolor='black', markeredgewidth=3)
    
    # Add goal label
    ax.text(events.loc[goal_idx]['Start X'], events.loc[goal_idx]['Start Y'] + 4, 
            f"Goal {goal_number}", ha='center', fontsize=12, fontweight='bold', 
            color='black', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.title(f"PFF - Goal {goal_number}: {team_flag} {team_name} - {goal['From']} ({period_name}, {goal_time_min:.1f}')")
    plt.show()

    # Show detailed passing sequence leading to goal (event data only)
    print(f">> Plotting passing sequence leading to Goal {goal_number}...")
    sequence_start = max(0, goal_idx - 8)  # 8 events before the goal
    sequence_events = events.loc[sequence_start:goal_idx]
    
    mviz.plot_events(sequence_events, indicators=['Marker','Arrow'], annotate=True)
    plt.title(f"PFF - Buildup to Goal {goal_number} ({goal['From']})")
    plt.show()
    print(f">> Goal {goal_number} analysis complete!\n")

#### TRACKING DATA ####

print("\n" + "="*50)
print(">> TRACKING DATA ANALYSIS")
print("="*50)

# READING IN TRACKING DATA
print(">> Loading tracking data...")
tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')  # World Cup Final
tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')  # World Cup Final

# Look at the column names
print(">> Tracking data columns:")
home_players_count = len([c for c in tracking_home.columns if c.startswith('Home_') and c.endswith('_x')])
away_players_count = len([c for c in tracking_away.columns if c.startswith('Away_') and c.endswith('_x')])
print(f"[HOME] {home_team_name} players: {home_players_count}")
print(f"[AWAY] {away_team_name} players: {away_players_count}")
print(f" ALL PLAYERS PRESERVED! (vs 14-player limit that would lose {max(0, home_players_count-14)} HOME + {max(0, away_players_count-14)} AWAY players)")

# Convert tracking positions from metrica units to meters 
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)

# Plot some player trajectories (first 15 seconds, with realistic movement filtering)
home_players = [c.split('_')[1] for c in tracking_home.columns if c.startswith('Home_') and c.endswith('_x')][:5]
colors = ['red', 'blue', 'green', 'orange', 'purple']

print(f">> Plotting player trajectories for first {len(home_players)} {home_team_name} players (10 seconds to 3 minutes)...")
fig,ax = mviz.plot_pitch()

# Filter to 10 seconds to 2 minutes (using actual time values)
time_window = tracking_home[(tracking_home['Time [s]'] >= 10) & (tracking_home['Time [s]'] <= 180)]
start_time = time_window.iloc[0]['Time [s]']
end_time = time_window.iloc[-1]['Time [s]']
num_frames = len(time_window)
print(f"   Using {num_frames} frames from {start_time:.1f}s to {end_time:.1f}s (170 second window)")

for i, player in enumerate(home_players):
    x_col = f'Home_{player}_x'
    y_col = f'Home_{player}_y'
    vis_col = f'Home_{player}_visibility'
    
    if x_col in time_window.columns and y_col in time_window.columns:
        # Get valid positions only
        player_data = time_window[[x_col, y_col, vis_col, 'Time [s]']].copy() if vis_col in time_window.columns else time_window[[x_col, y_col, 'Time [s]']].copy()
        
        # NO VISIBILITY FILTERING - include all positions (VISIBLE and ESTIMATED)
        # if vis_col in time_window.columns:
        #     player_data = player_data[player_data[vis_col] == 'VISIBLE']
        
        # Drop NaN positions
        player_data = player_data.dropna(subset=[x_col, y_col])
        
        if len(player_data) > 1:
            # Calculate time gaps between consecutive VISIBLE frames
            time_diff = player_data['Time [s]'].diff()
            
            # Calculate position changes
            x_diff = player_data[x_col].diff().abs()
            y_diff = player_data[y_col].diff().abs()
            distance = (x_diff**2 + y_diff**2)**0.5
            
            # Find where to break the trajectory line:
            # - Large time gap (>2 seconds between VISIBLE frames = player was off-camera)
            # - Large distance jump (>10m between consecutive VISIBLE frames = tracking error)
            breaks = (time_diff > 2.0) | (distance > 10.0)
            breaks.iloc[0] = True  # Start of first segment
            
            # Split into continuous segments
            segment_ids = breaks.cumsum()
            
            total_points = 0
            segments_plotted = 0
            
            for segment_id in segment_ids.unique():
                segment = player_data[segment_ids == segment_id]
                
                # Only plot segments with at least 3 points
                if len(segment) >= 3:
                    label = f'HOME {player}' if segments_plotted == 0 else None
                    ax.plot(segment[x_col], segment[y_col], color=colors[i], 
                           alpha=0.8, linewidth=1.5, label=label)
                    total_points += len(segment)
                    segments_plotted += 1
            
            if segments_plotted > 0:
                print(f"   Player {player}: {total_points}/{len(player_data)} positions ({segments_plotted} continuous segments)")

ax.legend(loc='upper right', fontsize=8)
plt.title(f"PFF - Player Movement Paths ({home_team_name}, 10s to 2min - ALL positions)")
plt.show()

# Plot player positions at kickoff (first event frame)
print(">> Plotting positions at kickoff...")
# Use the properly mapped Start Frame from first event
if 'Start Frame' in events.columns and pd.notna(events.loc[0, 'Start Frame']):
    KO_Frame = int(events.loc[0]['Start Frame'])
    print(f"   Using mapped event frame: {KO_Frame}")
else:
    KO_Frame = tracking_home.index[0]  # Use first available frame as fallback
    print(f"   Start Frame not mapped, using first tracking frame: {KO_Frame}")

if KO_Frame in tracking_home.index and KO_Frame in tracking_away.index:
    fig,ax = mviz.plot_frame(tracking_home.loc[KO_Frame], tracking_away.loc[KO_Frame])
    plt.title("PFF - Positions at Kickoff")
    plt.show()
    print("   Kickoff plot displayed successfully!")
else:
    print(f"   Kickoff frame {KO_Frame} not available in tracking data")
    print(f"   Available frames: {tracking_home.index[0]} to {tracking_home.index[-1]}")

# Plot positions at first goal (if available)
if len(goals) >= 1:
    print(">> Plotting positions at first goal...")
    first_goal = goals.iloc[0]  # First goal
    print(f"   First goal: {first_goal['From']} at {first_goal['Start Time [s]']:.1f}s")
    
    # Use the properly mapped Start Frame from event data (mapped via game_event_id)
    goal_frame = first_goal['Start Frame']
    
    if pd.notna(goal_frame):
        goal_frame = int(goal_frame)
        print(f"   Using mapped frame: {goal_frame}")
        
        if goal_frame in tracking_home.index and goal_frame in tracking_away.index:
            tracking_time = tracking_home.loc[goal_frame, 'Time [s]']
            print(f"   Tracking time at frame: {tracking_time:.1f}s")
            
            # First plot the goal event
            fig,ax = mviz.plot_events(events.loc[goals.index[0]:goals.index[0]], indicators=['Marker','Arrow'], annotate=True)
            # Then overlay player positions
            fig,ax = mviz.plot_frame(tracking_home.loc[goal_frame], tracking_away.loc[goal_frame], figax=(fig,ax))
            plt.title(f"PFF - Positions at First Goal ({first_goal['From']}) - Frame {goal_frame}")
            plt.show()
            print("   First goal plot displayed successfully!")
        else:
            print(f"   Goal frame {goal_frame} not available in tracking data")
    else:
        print(f"   Goal event not mapped to tracking frame (Start Frame is NULL)")
else:
    print(">> No goals found, skipping goal analysis")

print("\n" + "="*50)
print(">> COMPLETE PFF ANALYSIS FINISHED!")
print("="*50)
print(">> Event data analysis:  Complete")
print(">> Tracking data analysis:  Complete") 
print(f">> Total goals analyzed: {len(goals)}")
print(">> Tutorial 1 fully replicated with PFF data!")
print(">> Plots displayed (not saved to files)")
