#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 3: Pitch Control Analysis - PFF Data

This tutorial applies William Spearman's pitch control model to analyze space control 
and possession probability during key moments of PFF matches.

Adapted from Friends of Tracking Tutorial 3
Data: PFF Match Data
"""

import Metrica_IO as mio
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("TUTORIAL 3: PITCH CONTROL ANALYSIS")
print("PFF DATA")
print("=" * 70)
print()

# =============================================================================
# CONFIGURATION - Change these values for different matches
# =============================================================================
DATADIR = 'Sample Data'
game_id = 10517  # Change this to your desired game ID
home_team_name = "Home Team"  # Will be determined from data
away_team_name = "Away Team"  # Will be determined from data

print("Loading PFF data...")
# Read in the event data
events = mio.read_event_data(DATADIR, game_id)

# Read in tracking data
tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')

# Convert positions to meters
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

# Reverse direction of play in the second half so that home team always attacks left->right
tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)

print(f"Data loaded: {len(events)} events, {len(tracking_home):,} tracking frames")
print()

# Calculate player velocities
# Check if PFF speed columns exist for hybrid velocity calculation
pff_speed_cols = [c for c in tracking_home.columns if c.endswith('_pff_speed')]
if pff_speed_cols:
    print("Calculating player velocities using HYBRID method (PFF speed + calculated direction)...")
    print("  -> Using PFF's raw speed values (calculated at ~15 FPS) for more accurate velocities")
    tracking_home = mvel.calc_player_velocities_hybrid(tracking_home, smoothing=True, use_pff_speed=True)
    tracking_away = mvel.calc_player_velocities_hybrid(tracking_away, smoothing=True, use_pff_speed=True)
else:
    print("Calculating player velocities from position differences...")
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

print(">>> PITCH CONTROL ANALYSIS FOR WORLD CUP GOALS <<<")
print()

# Get all shots and goals in the match
shots = events[events['Type']=='SHOT']
goals = shots[shots['Subtype'].str.contains('GOAL', na=False)].copy()

print(f"Found {len(goals)} goals in the match:")
for i, goal in goals.iterrows():
    period = goal['Period']
    time = goal['Start Time [s]']
    team = goal['Team']
    player = goal['From']
    print(f"  Goal {len(goals[goals.index <= i])}: {team} - {player} at {time:.0f}s (Period {period})")

print()

# Get pitch control model parameters
params = mpc.default_model_params()

# Find goalkeepers for offside calculation
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
print(f"Goalkeepers: {home_team_name} #{GK_numbers[0]}, {away_team_name} #{GK_numbers[1]}")
print()

# Analyze pitch control for 3 passes before each goal
print("=" * 70)
print("ANALYZING PITCH CONTROL FOR 3 PASSES BEFORE EACH GOAL")
print("=" * 70)
print()

total_analyzed = 0
for goal_num, (goal_idx, goal) in enumerate(goals.iterrows(), start=1):
    goal_scorer = goal.get('From', 'Unknown')
    goal_team = goal.get('Team', 'Unknown')
    goal_time = goal.get('Start Time [s]', 0)
    
    print("=" * 70)
    print(f">>> GOAL {goal_num}: {goal_scorer} ({goal_team}) at {goal_time:.0f}s <<<")
    print("=" * 70)
    print()
    
    # Find the 3 passes/events leading up to this goal
    # Look back at most 10 events to find passes/carries
    lookback_start = max(0, goal_idx - 10)
    events_before_goal = events.loc[lookback_start:goal_idx-1]  # Exclude the goal shot itself
    
    # Filter for PASS and CARRY events from the same team as the goal scorer
    pass_events = events_before_goal[
        (events_before_goal['Type'].isin(['PASS', 'CARRY'])) & 
        (events_before_goal['Team'] == goal_team)
    ]
    
    # Get the last 3 passes/carries before the goal
    analysis_events = pass_events.tail(3)
    
    if len(analysis_events) == 0:
        print(f"  No passes found before this goal. Skipping...")
        print()
        continue
    
    print(f"Analyzing {len(analysis_events)} events leading to {goal_scorer}'s goal:")
    for i, event in analysis_events.iterrows():
        print(f"  Event {i}: {event['Type']} by {event.get('From', 'Unknown')} -> {event.get('To', 'Unknown')}")
    print()
    
    # Plot the events leading up to the goal
    mviz.plot_events(analysis_events, color='k', indicators=['Marker','Arrow'], annotate=True)
    plt.title(f"Goal {goal_num}: Events Leading to {goal_scorer}'s Goal (PFF)")
    plt.show()
    
    # Generate pitch control for each event
    for event_idx, event in analysis_events.iterrows():
        event_type = event['Type']
        event_player = event.get('From', 'Unknown')
        
        print(f"  Pitch control for Event {event_idx}: {event_type} by {event_player}")
        
        try:
            # Check if we have tracking data for this event
            event_frame = int(event['Start Frame'])
            if event_frame not in tracking_home.index or event_frame not in tracking_away.index:
                print(f"    Skipping - no tracking data for frame {event_frame}")
                continue
            
            # Generate pitch control surface
            PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
                event_idx, events, tracking_home, tracking_away, params, GK_numbers, 
                field_dimen=(106., 68.), n_grid_cells_x=50
            )
            
            # Plot pitch control
            mviz.plot_pitchcontrol_for_event(
                event_idx, events, tracking_home, tracking_away, PPCF, annotate=True
            )
            plt.title(f"Goal {goal_num} - Pitch Control: Event {event_idx} - {event_type} by {event_player}")
            plt.show()
            
            total_analyzed += 1
            
        except Exception as e:
            print(f"    Error analyzing event {event_idx}: {e}")
            print(f"    Event frame: {event.get('Start Frame', 'N/A')}, Event time: {event.get('Start Time [s]', 'N/A')}")
            continue
    
    print()

print("=" * 70)
print("TUTORIAL 3 COMPLETED - PITCH CONTROL ANALYSIS - PFF")
print("=" * 70)
print()
print(f"Analyzed pitch control for {total_analyzed} total events across {len(goals)} goals")
print("Each plot shows the spatial control at the moment of the pass/carry")
print("Red areas = attacking team controls, Blue areas = defending team controls")
print()

