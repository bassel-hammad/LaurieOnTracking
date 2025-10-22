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
game_id = 10514  # Change this to your desired game ID
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
print("Calculating player velocities...")
tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

print(">>> PITCH CONTROL ANALYSIS FOR WORLD CUP GOALS <<<")
print()

# Get all shots and goals in the match
shots = events[events['Type']=='SHOT']
goals = shots[shots['Subtype'].str.contains('GOAL', na=False)].copy()

print(f"Found {len(goals)} goals in the PFF:")
for i, goal in goals.iterrows():
    period = goal['Period']
    time = goal['Start Time [s]']
    team = goal['Team']
    player = goal['From']
    print(f"  Goal {len(goals[goals.index <= i])}: {team} - {player} at {time:.0f}s (Period {period})")

print()

# Analyze second goal - key passing sequence
if len(goals) >= 2:
    print(">>> ANALYZING SECOND GOAL - PASSING SEQUENCE <<<")
    
    # Find second goal
    second_goal_idx = goals.index[1]  # Second goal
    second_goal = goals.loc[second_goal_idx]
    goal_scorer = second_goal.get('From', 'Unknown')
    
    print(f"Second goal by {goal_scorer} at frame {second_goal['Start Frame']:.0f}, time {second_goal['Start Time [s]']:.0f}s")
    print()
    
    # Find the 3 events leading up to the second goal
    lead_up_start = max(0, second_goal_idx - 3)
    lead_up_events = events.loc[lead_up_start:second_goal_idx]
    
    print(f"Events leading up to {goal_scorer}'s goal:")
    for i, event in lead_up_events.iterrows():
        print(f"  Event {i}: {event['Type']} by {event['Team']} - {event.get('From', 'Unknown')} -> {event.get('To', 'Unknown')}")
    
    # Plot the events leading up to the goal
    print()
    print(f"Plotting events leading up to {goal_scorer}'s goal...")
    mviz.plot_events(lead_up_events, color='k', indicators=['Marker','Arrow'], annotate=True)
    plt.title(f"Events Leading to {goal_scorer}'s Goal - PFF")
    plt.show()
else:
    print(">>> SECOND GOAL NOT FOUND - SKIPPING DETAILED ANALYSIS <<<")
    print("Only one goal found in the match data.")

# Get pitch control model parameters
params = mpc.default_model_params()

# Find goalkeepers for offside calculation
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
print(f"Goalkeepers: {home_team_name} #{GK_numbers[0]}, {away_team_name} #{GK_numbers[1]}")
print()

# Analyze pitch control for the events leading to the goal
print(">>> PITCH CONTROL ANALYSIS <<<")
print()

analysis_events = []
for i, event_idx in enumerate(lead_up_events.index[:-1]):  # Exclude the goal itself
    event = events.loc[event_idx]
    if event['Type'] in ['PASS', 'CARRY']:
        analysis_events.append((event_idx, event))

print(f"Analyzing pitch control for {len(analysis_events)} key events...")

for i, (event_idx, event) in enumerate(analysis_events):
    print(f"Pitch control for event {event_idx}: {event['Type']} by {event['Team']}")
    
    try:
        # Check if we have tracking data for this event
        event_frame = int(event['Start Frame'])
        if event_frame not in tracking_home.index or event_frame not in tracking_away.index:
            print(f"  Skipping - no tracking data for frame {event_frame}")
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
        plt.title(f"Pitch Control - Event {event_idx}: {event['Type']} (PFF)")
        plt.show()
        
    except Exception as e:
        print(f"  Error analyzing event {event_idx}: {e}")
        print(f"  Event frame: {event.get('Start Frame', 'N/A')}, Event time: {event.get('Start Time [s]', 'N/A')}")
        continue

print()
print(">>> PASS SUCCESS PROBABILITY ANALYSIS <<<")
print()

# Calculate pass probability for every Home team successful pass
home_passes = events[(events['Type'].isin(['PASS'])) & (events['Team']=='Home')]

print(f"Analyzing {len(home_passes)} {home_team_name} passes for success probability...")

# List for storing pass probabilities
pass_success_probability = []

for i, row in home_passes.iterrows():
    try:
        pass_start_pos = np.array([row['Start X'], row['Start Y']])
        pass_target_pos = np.array([row['End X'], row['End Y']])
        pass_frame = int(row['Start Frame'])
        
        # Skip if frame data is not available
        if pass_frame not in tracking_home.index or pass_frame not in tracking_away.index:
            continue
            
        attacking_players = mpc.initialise_players(
            tracking_home.loc[pass_frame], 'Home', params, GK_numbers[0]
        )
        defending_players = mpc.initialise_players(
            tracking_away.loc[pass_frame], 'Away', params, GK_numbers[1]
        )
        
        Patt, Pdef = mpc.calculate_pitch_control_at_target(
            pass_target_pos, attacking_players, defending_players, pass_start_pos, params
        )
        
        pass_success_probability.append((i, Patt))
        
    except Exception as e:
        # Skip problematic passes
        continue

print(f"Successfully analyzed {len(pass_success_probability)} passes")
print()

# Plot histogram of pass success probabilities
fig, ax = plt.subplots(figsize=(10, 6))
probabilities = [p[1] for p in pass_success_probability]
ax.hist(probabilities, bins=np.arange(0, 1.1, 0.1), alpha=0.7, color='skyblue', edgecolor='black')
ax.set_xlabel('Pass Success Probability')
ax.set_ylabel('Frequency')
ax.set_title(f'{home_team_name} Pass Success Probability Distribution - PFF')
ax.grid(True, alpha=0.3)

# Add statistics
mean_prob = np.mean(probabilities)
ax.axvline(mean_prob, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_prob:.2f}')
ax.legend()
plt.show()

print(f"{home_team_name} pass statistics:")
print(f"  Mean success probability: {mean_prob:.2f}")
print(f"  Passes with >80% probability: {sum(1 for p in probabilities if p > 0.8)}")
print(f"  Risky passes (<50% probability): {sum(1 for p in probabilities if p < 0.5)}")
print()

# Sort the passes by pitch control probability
pass_success_probability = sorted(pass_success_probability, key=lambda x: x[1])

# Identify the events corresponding to the most risky passes (pitch control < 0.5)
risky_passes = events.loc[[p[0] for p in pass_success_probability if p[1] < 0.5]]

if len(risky_passes) > 0:
    print(f">>> RISKY PASSES ANALYSIS ({len(risky_passes)} passes with <50% success probability) <<<")
    print()
    
    # Plot the risky passes
    mviz.plot_events(risky_passes, color='r', indicators=['Marker','Arrow'], annotate=True)
    plt.title(f'{home_team_name} Risky Passes (<50% Success Probability) - PFF')
    plt.show()
    
    # Print events that followed those risky passes
    print(f"Outcomes following risky {home_team_name} passes:")
    risk_outcomes = {}
    
    for p in pass_success_probability[:min(20, len(risky_passes))]:
        try:
            if p[0] + 1 in events.index:
                outcome = events.loc[p[0] + 1]['Type']
                prob = p[1]
                print(f"  Probability: {prob:.2f} -> Next event: {outcome}")
                
                # Count outcomes
                if outcome not in risk_outcomes:
                    risk_outcomes[outcome] = 0
                risk_outcomes[outcome] += 1
        except:
            continue
    
    print()
    print("Summary of outcomes after risky passes:")
    for outcome, count in risk_outcomes.items():
        print(f"  {outcome}: {count} times")

else:
    print(f"No passes found with <50% success probability - {home_team_name} played very conservatively!")

print()
print(">>> HIGH-SUCCESS PASSES ANALYSIS <<<")

# Analyze high-success passes (>80% probability)
high_success_passes = [p for p in pass_success_probability if p[1] > 0.8]

if len(high_success_passes) > 0:
    print(f"Found {len(high_success_passes)} high-confidence passes (>80% success probability)")
    
    high_success_events = events.loc[[p[0] for p in high_success_passes]]
    
    # Plot high-success passes
    mviz.plot_events(high_success_events, color='g', indicators=['Marker','Arrow'], annotate=True)
    plt.title(f'{home_team_name} High-Confidence Passes (>80% Success Probability) - PFF')
    plt.show()
    
    print("High-confidence pass statistics:")
    periods = high_success_events['Period'].value_counts().sort_index()
    for period, count in periods.items():
        print(f"  Period {period}: {count} passes")

print()
print("=" * 70)
print("TUTORIAL 3 COMPLETED - PITCH CONTROL ANALYSIS - PFF")
print("=" * 70)
print()
print("Key insights from the PFF pitch control analysis:")

print("2. Spatial control during key goal-scoring sequences") 
print("3. Pass success probability distribution across the match")

