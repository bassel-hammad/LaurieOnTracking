#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 4: Expected Possession Value (EPV) Analysis
PFF DATA

Adapted from LaurieOnTracking Tutorial 4 for PFF data.
Analyzes passing options and decision-making using Expected Possession Value (EPV) 
and pitch control during PFF matches.

Original Tutorial by Laurie Shaw (@EightyFivePoint)
PFF Adaptation: 2024

Key concepts:
- Expected Possession Value (EPV): Probability that possession will end in a goal
- EPV-added: Value created by a pass compared to retaining possession
- Optimal passing decisions vs actual decisions made
- Spatial analysis of passing value across the pitch
"""

import Metrica_IO as mio
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import Metrica_EPV as mepv
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 70)
    print("TUTORIAL 4: EXPECTED POSSESSION VALUE (EPV) ANALYSIS")
    print("PFF DATA")
    print("=" * 70)
    
    # =============================================================================
    # CONFIGURATION - Change these values for different matches
    # =============================================================================
    DATADIR = 'Sample Data'
    game_id = 10514  # Change this to your desired game ID
    home_team_name = "Home Team"  # Will be determined from data
    away_team_name = "Away Team"  # Will be determined from data
    
    print("\nLoading PFF data...")
    
    # Read in the event data
    events = mio.read_event_data(DATADIR, game_id)
    
    # Read in tracking data
    tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
    tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')
    
    # Convert positions from Metrica units to meters
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
    
    # Filter out penalty shootout data (data after 7200s)
    normal_match_time = 7200  # 120 minutes
    print(f"Filtering out penalty shootout data after {normal_match_time}s...")
    tracking_home = tracking_home[tracking_home['Time [s]'] <= normal_match_time].copy()
    tracking_away = tracking_away[tracking_away['Time [s]'] <= normal_match_time].copy()
    events = events[events['Start Time [s]'] <= normal_match_time].copy()
    
    # Reverse direction of play so home team always attacks from right->left
    # FIXED: Only flip periods 2 and 4, not period 3
    tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)
    
    # Calculate player velocities
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
    
    print(f"Data loaded: {len(events)} events, {len(tracking_home):,} tracking frames (penalty shootout data filtered out)")
    
    # Get pitch control model parameters
    params = mpc.default_model_params()
    
    # Find goalkeepers for offside calculation
    GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
    print(f"Goalkeepers: {home_team_name} #{GK_numbers[0]}, {away_team_name} #{GK_numbers[1]}")
    
    # Load Expected Possession Value surface
    print("\nLoading Expected Possession Value (EPV) grid...")
    home_attack_direction = mio.find_playing_direction(tracking_home, 'Home')
    EPV = mepv.load_EPV_grid('EPV_grid.csv')
    
    # Plot the EPV surface
    print("Plotting EPV surface...")
    mviz.plot_EPV(EPV, field_dimen=(106.0, 68), attack_direction=home_attack_direction)
    plt.title('Expected Possession Value (EPV) Surface\nProbability that possession ends in a goal')
    plt.show()
    
    print("\n" + "=" * 70)
    print(">>> EPV ANALYSIS FOR PFF GOALS <<<")
    print("=" * 70)
    
    # Find goals in the PFF
    goals = events[events['Subtype'].str.contains('GOAL', na=False)]
    print(f"\nFound {len(goals)} goals in the PFF:")
    
    for i, (idx, goal) in enumerate(goals.iterrows()):
        team = home_team_name if goal['Team'] == 'Home' else away_team_name
        player = goal['From']
        time = goal['Start Time [s]']
        period = goal['Period']
        print(f"  Goal {i+1}: {team} - {player} at {time:.0f}s (Period {period})")
    
    if len(goals) >= 2:
        print("\n>>> ANALYZING SECOND GOAL - EPV BUILD-UP <<<")
        
        # Focus on second goal
        second_goal = goals.iloc[1]  # Second goal
        goal_frame = second_goal.name
        goal_time = second_goal['Start Time [s]']
        goal_scorer = second_goal.get('From', 'Unknown')
        
        print(f"Second goal by {goal_scorer} at frame {goal_frame}, time {goal_time:.0f}s")
        
        # Find events leading up to the second goal
        pre_goal_events = events[(events.index < goal_frame) & 
                                (events.index >= goal_frame - 10) &
                                (events['Team'] == 'Home')]
        
        print(f"\nEvents leading up to {goal_scorer}'s goal:")
        for idx, event in pre_goal_events.iterrows():
            print(f"  Event {idx}: {event['Type']} by {event['Team']} - {event['From']} -> {event['To']}")
    else:
        print("\n>>> SECOND GOAL NOT FOUND - SKIPPING DETAILED ANALYSIS <<<")
        print("Only one goal found in the match data.")
        second_goal = None
    
    # Analyze EPV for the pass that set up the second goal (if available)
    if len(goals) >= 2 and second_goal is not None:
        assist_event = goal_frame - 1  # Event just before the goal
        
        if assist_event in events.index:
            print(f"\nAnalyzing EPV for assist pass (Event {assist_event})...")
            
            try:
                # Calculate EPV-added for the assist
                EEPV_added, EPV_diff = mepv.calculate_epv_added(
                    assist_event, events, tracking_home, tracking_away, GK_numbers, EPV, params
                )
                
                # Generate pitch control for the event
                PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
                    assist_event, events, tracking_home, tracking_away, params, GK_numbers,
                    field_dimen=(106., 68.), n_grid_cells_x=50, offsides=True
                )
                
                # Plot EPV surface for the assist
                fig, ax = mviz.plot_EPV_for_event(
                    assist_event, events, tracking_home, tracking_away, PPCF, EPV,
                    annotate=True, autoscale=True
                )
                fig.suptitle(f'{goal_scorer} Goal Assist - EPV Added: {EEPV_added:.3f}', y=0.95)
                plt.show()
                
                print(f"EPV-added for assist: {EEPV_added:.3f}")
                
            except Exception as e:
                print(f"Could not analyze assist event: {e}")
    else:
        print("\n>>> SKIPPING ASSIST ANALYSIS - SECOND GOAL NOT AVAILABLE <<<")
    
    print("\n>>> COMPREHENSIVE PASS ANALYSIS <<<")
    
    # Get all shots and passes
    shots = events[events['Type'] == 'SHOT']
    home_shots = shots[shots['Team'] == 'Home']
    away_shots = shots[shots['Team'] == 'Away']
    
    home_passes = events[(events['Type'] == 'PASS') & (events['Team'] == 'Home')]
    away_passes = events[(events['Type'] == 'PASS') & (events['Team'] == 'Away')]
    
    print(f"\nMatch statistics:")
    print(f"  {home_team_name}: {len(home_passes)} passes, {len(home_shots)} shots")
    print(f"  {away_team_name}: {len(away_passes)} passes, {len(away_shots)} shots")
    
    # Analyze a sample of Home team passes for EPV-added
    print(f"\nAnalyzing EPV-added for ALL {home_team_name} passes (this may take a bit)...")
    
    home_pass_value_added = []
    analyzed_count = 0
    
    for i, pass_ in home_passes.iterrows():
        try:
            EEPV_added, EPV_diff = mepv.calculate_epv_added(
                i, events, tracking_home, tracking_away, GK_numbers, EPV, params
            )
            home_pass_value_added.append((i, EEPV_added, EPV_diff))
            analyzed_count += 1
            if analyzed_count % 100 == 0:
                print(f"  Analyzed {analyzed_count} passes...")
                
        except Exception as e:
            continue  # Skip passes that can't be analyzed
    
    print(f"Successfully analyzed {len(home_pass_value_added)} {home_team_name} passes (of {len(home_passes)})")
    
    # Sort by EPV-added value
    home_pass_value_added = sorted(home_pass_value_added, key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 5 {home_team_name} passes by EPV-added:")
    for i, (event_idx, epv_added, epv_diff) in enumerate(home_pass_value_added[:5]):
        event = events.loc[event_idx]
        player = event['From']
        time = event['Start Time [s]']
        print(f"  {i+1}. Event {event_idx}: {player} at {time:.0f}s (EPV+: {epv_added:.3f})")
    
    # Find the top 1 EPV-added event for each team in each period
    print(f"\nFinding top 1 EPV-added event for each team in each period...")
    
    # Get all periods in the data
    all_periods = sorted(events['Period'].unique())
    print(f"Periods found: {all_periods}")
    
    # Find top event for each team in each period
    top_events_by_team_period = []
    for period in all_periods:
        print(f"\n  Analyzing Period {period}...")
        
        # Get all passes for this period
        period_passes = events[(events['Type'] == 'PASS') & (events['Period'] == period)]
        if len(period_passes) == 0:
            print(f"    No passes found in Period {period}")
            continue
        
        # Separate by team
        home_passes = period_passes[period_passes['Team'] == 'Home']
        away_passes = period_passes[period_passes['Team'] == 'Away']
        
        print(f"    Found {len(home_passes)} {home_team_name} passes and {len(away_passes)} {away_team_name} passes in Period {period}")
        
        # Find top event for Home team
        if len(home_passes) > 0:
            home_epv_values = []
            home_errors = 0
            for event_idx in home_passes.index:
                try:
                    EEPV_added, EPV_diff = mepv.calculate_epv_added(
                        event_idx, events, tracking_home, tracking_away, GK_numbers, EPV, params
                    )
                    home_epv_values.append((event_idx, EEPV_added, EPV_diff))
                except Exception as e:
                    home_errors += 1
                    continue
            
            print(f"    {home_team_name}: Successfully analyzed {len(home_epv_values)} passes, {home_errors} errors")
            
            if len(home_epv_values) > 0:
                home_epv_values.sort(key=lambda x: x[1], reverse=True)
                top_home_event = home_epv_values[0]
                top_events_by_team_period.append(('Home', period, top_home_event[0], top_home_event[1], top_home_event[2]))
                print(f"    {home_team_name}: Event {top_home_event[0]} with EPV+ {top_home_event[1]:.3f}")
            else:
                print(f"    {home_team_name}: No valid EPV calculations for Period {period}")
        else:
            print(f"    {home_team_name}: No passes found in Period {period}")
        
        # Find top event for Away team
        if len(away_passes) > 0:
            away_epv_values = []
            away_errors = 0
            for event_idx in away_passes.index:
                try:
                    EEPV_added, EPV_diff = mepv.calculate_epv_added(
                        event_idx, events, tracking_home, tracking_away, GK_numbers, EPV, params
                    )
                    away_epv_values.append((event_idx, EEPV_added, EPV_diff))
                except Exception as e:
                    away_errors += 1
                    continue
            
            print(f"    {away_team_name}: Successfully analyzed {len(away_epv_values)} passes, {away_errors} errors")
            
            if len(away_epv_values) > 0:
                away_epv_values.sort(key=lambda x: x[1], reverse=True)
                top_away_event = away_epv_values[0]
                top_events_by_team_period.append(('Away', period, top_away_event[0], top_away_event[1], top_away_event[2]))
                print(f"    {away_team_name}: Event {top_away_event[0]} with EPV+ {top_away_event[1]:.3f}")
            else:
                print(f"    {away_team_name}: No valid EPV calculations for Period {period}")
        else:
            print(f"    {away_team_name}: No passes found in Period {period}")
    
    # Plot the top event from each team in each period
    print(f"\nPlotting top 1 event for each team in each period...")
    for rank, (team, period, event_idx, epv_added, epv_diff) in enumerate(top_events_by_team_period, start=1):
        try:
            PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
                event_idx, events, tracking_home, tracking_away, params, GK_numbers,
                field_dimen=(106., 68.), n_grid_cells_x=50, offsides=True
            )
            fig, ax = mviz.plot_EPV_for_event(
                event_idx, events, tracking_home, tracking_away, PPCF, EPV,
                annotate=True, autoscale=True, contours=True
            )
            # Format game time (per-period and cumulative)
            row = events.loc[event_idx]
            t = float(row['Start Time [s]'])
            period_num = int(row['Period'])
            # Determine actual period start from the data to account for added time
            try:
                period_start = float(events[events['Period']==period_num]['Start Time [s]'].min())
                if np.isnan(period_start):
                    period_start = 0.0
            except Exception:
                period_start = 0.0
            t_period = max(0.0, t - period_start)
            # per-period clock
            pmm = int(t_period // 60)
            pss = int(round(t_period - 60 * pmm))
            # cumulative clock
            cmm = int(t // 60)
            css = int(round(t - 60 * cmm))
            # Passer/receiver and original PFF id if present
            from_player = str(row.get('From', 'Unknown'))
            to_player = str(row.get('To', 'Unknown'))
            team_name = home_team_name if team == 'Home' else away_team_name
            pff_id = row.get('PFF Event ID', None)
            id_part = f"PFF:{int(pff_id)} | " if pff_id is not None and not np.isnan(pff_id) else ""
            fig.suptitle(
                f'#{rank} {team_name} EPV+ {epv_added:.3f} | {id_part}Event {event_idx} | {from_player} → {to_player} | P{period_num} {pmm:02d}:{pss:02d} (cum {cmm:02d}:{css:02d})',
                y=0.95
            )
            plt.show()
        except Exception as e:
            print(f"Could not visualize event {event_idx} from {team} in Period {period}: {e}")
    
    
    print("\n" + "=" * 70)
    print(">>> EPV INSIGHTS FROM THE PFF <<<")
    print("=" * 70)
    
    if top_events_by_team_period:
        print(f"\nEPV Analysis Summary:")
        print(f"  Found {len(top_events_by_team_period)} team-period combinations with valid EPV calculations")
        
        # Group by period for display
        periods = {}
        for team, period, event_idx, epv_added, epv_diff in top_events_by_team_period:
            if period not in periods:
                periods[period] = []
            periods[period].append((team, event_idx, epv_added, epv_diff))
        
        # Show summary of top events by period and team
        print(f"\n  Top EPV-added event per team per period:")
        for period in sorted(periods.keys()):
            print(f"    Period {period}:")
            for team, event_idx, epv_added, epv_diff in periods[period]:
                event = events.loc[event_idx]
                from_player = event.get('From', 'Unknown')
                to_player = event.get('To', 'Unknown')
                team_name = home_team_name if team == 'Home' else away_team_name
                print(f"      {team_name}: {from_player} → {to_player} (EPV+: {epv_added:.3f})")
        
        # Count teams found
        home_team_periods = [p for p in top_events_by_team_period if p[0] == 'Home']
        away_team_periods = [p for p in top_events_by_team_period if p[0] == 'Away']
        print(f"\n  Team Analysis Summary:")
        print(f"    {home_team_name}: {len(home_team_periods)} periods with valid EPV data")
        print(f"    {away_team_name}: {len(away_team_periods)} periods with valid EPV data")
        
        # Find the overall best event
        best_event = max(top_events_by_team_period, key=lambda x: x[3])
        team, period, event_idx, epv_added, epv_diff = best_event
        event = events.loc[event_idx]
        from_player = event.get('From', 'Unknown')
        to_player = event.get('To', 'Unknown')
        team_name = home_team_name if team == 'Home' else away_team_name
        print(f"\n  🏆 Overall best pass: {team_name} {from_player} → {to_player} in Period {period} (EPV+: {epv_added:.3f})")
    else:
        print(f"\nEPV Analysis Summary:")
        print(f"  No valid EPV calculations found for any team in any period")
        print(f"  This could be due to:")
        print(f"    - Missing tracking data for certain periods")
        print(f"    - EPV calculation errors (check data quality)")
        print(f"    - No passes found in the data")
    
    print("\n📊 Key EPV Insights:")
    print("   • EPV measures the probability that current possession ends in a goal")
    print("   • High EPV areas: Near goal, central positions, good passing angles")
    print("   • EPV-added: Value created by a pass vs keeping possession")
    print("   • Positive EPV-added = good passing decision")
    print("   • Negative EPV-added = poor passing decision")
    
    print("\n⚽ PFF EPV Lessons:")
    print("   • Team build-up play created valuable goal-scoring opportunities")
    print("   • Key passes in dangerous areas had high EPV-added values")
    print("   • Spatial positioning crucial for maximizing possession value")
    print("   • EPV analysis reveals quality of passing decisions under pressure")
    
    print("\n" + "=" * 70)
    print("TUTORIAL 4 COMPLETED - EPV ANALYSIS - PFF")
    print("=" * 70)
    
    print("\n🎯 Tutorial 4 completed! You've learned how to:")
    print("   1. Load and visualize Expected Possession Value (EPV) surfaces")
    print("   2. Calculate EPV-added for individual passes")
    print("   3. Identify the most valuable passing decisions")
    print("   4. Compare passing effectiveness between teams")
    print("   5. Understand spatial value creation in football")
    
if __name__ == "__main__":
    main()
