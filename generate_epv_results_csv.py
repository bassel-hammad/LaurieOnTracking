#!/usr/bin/env python3
"""
Generate CSV output with all events and their EPV-added values
Works with any PFF match data
"""

import Metrica_IO as mio
import Metrica_EPV as mepv
import Metrica_PitchControl as mpc
import Metrica_Velocities as mvel
import pandas as pd
import numpy as np

def main():
    print("=" * 70)
    print("GENERATING EPV RESULTS CSV")
    print("PFF DATA ANALYSIS")
    print("=" * 70)
    
    # =============================================================================
    # CONFIGURATION - Change these values for different matches
    # =============================================================================
    DATADIR = 'Sample Data'
    game_id = 10516  # Change this to your desired game ID
    home_team_name = "Home Team"  # Will be determined from data
    away_team_name = "Away Team"  # Will be determined from data
    
    print(f"\nLoading PFF data for game {game_id}...")
    
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
    tracking_home, tracking_away, events = mio.to_single_playing_direction(tracking_home, tracking_away, events)
    
    # Calculate player velocities
    tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
    
    # Load EPV grid
    print("Loading Expected Possession Value (EPV) grid...")
    EPV = mepv.load_EPV_grid('EPV_grid.csv')
    
    # Set up pitch control parameters
    params = mpc.default_model_params()
    # Find actual goalkeeper numbers from the data
    home_gk = mio.find_goalkeeper(tracking_home)
    away_gk = mio.find_goalkeeper(tracking_away)
    GK_numbers = [home_gk, away_gk]
    print(f"Goalkeepers: {home_team_name} #{home_gk}, {away_team_name} #{away_gk}")
    
    print(f"Data loaded: {len(events)} events, {len(tracking_home)} tracking frames")
    
    # Filter for pass events only in periods 1 and 2
    pass_events = events[(events['Type'] == 'PASS') & (events['Period'].isin([1, 2]))].copy()
    print(f"\nFound {len(pass_events)} pass events in periods 1 and 2")
    
    # Initialize results list
    results = []
    successful_count = 0
    error_count = 0
    
    print(f"\nAnalyzing EPV-added for all {len(pass_events)} passes in periods 1 and 2...")
    
    for i, (event_idx, event) in enumerate(pass_events.iterrows()):
        try:
            # Calculate EPV-added
            EEPV_added, EPV_diff = mepv.calculate_epv_added(
                event_idx, events, tracking_home, tracking_away, GK_numbers, EPV, params
            )
            
            # Format game time
            start_time = event['Start Time [s]']
            period = event['Period']
            
            # Determine actual period start from the data to account for added time
            try:
                period_start = float(events[events['Period']==period]['Start Time [s]'].min())
                if np.isnan(period_start):
                    period_start = 0.0
            except Exception:
                period_start = 0.0
            
            t_period = max(0.0, start_time - period_start)
            
            # per-period clock
            pmm = int(t_period // 60)
            pss = int(round(t_period - 60 * pmm))
            
            # cumulative clock
            cmm = int(start_time // 60)
            css = int(round(start_time - 60 * cmm))
            
            # Get PFF Event ID if available
            pff_id = event.get('PFF Event ID', '')
            if pd.isna(pff_id):
                pff_id = ''
            else:
                pff_id = int(pff_id)
            
            # Create result row
            result_row = {
                'Event_ID': event_idx,
                'PFF_Event_ID': pff_id,
                'Team': event['Team'],
                'Type': event['Type'],
                'Subtype': event['Subtype'] if not pd.isna(event['Subtype']) else '',
                'Period': period,
                'Start_Time_s': start_time,
                'Period_Time': f"{pmm:02d}:{pss:02d}",
                'Cumulative_Time': f"{cmm:02d}:{css:02d}",
                'From_Player': event['From'] if not pd.isna(event['From']) else '',
                'To_Player': event['To'] if not pd.isna(event['To']) else '',
                'Start_X': event['Start X'],
                'Start_Y': event['Start Y'],
                'End_X': event['End X'],
                'End_Y': event['End Y'],
                'EPV_Added': EEPV_added,
                'EPV_Difference': EPV_diff
            }
            
            results.append(result_row)
            successful_count += 1
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} passes... (Success: {successful_count}, Errors: {error_count})")
                
        except Exception as e:
            error_count += 1
            # Only print first few errors to avoid spam
            if error_count <= 10:
                print(f"  Error processing event {event_idx}: {e}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 Processing Summary:")
    print(f"  Total pass events: {len(pass_events)}")
    print(f"  Successfully processed: {successful_count}")
    print(f"  Errors encountered: {error_count}")
    
    if len(results_df) == 0:
        print("❌ No events were successfully processed. Cannot generate CSV.")
        return
    
    # Sort by EPV-added (descending)
    results_df = results_df.sort_values('EPV_Added', ascending=False)
    
    # Save to CSV with dynamic filename
    output_file = f'PFF_Game_{game_id}_EPV_Results_Periods_1_2.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"📊 Total events analyzed: {len(results_df)}")
    
    # Show summary statistics
    print(f"\n📈 Summary Statistics (Periods 1 & 2 only):")
    print(f"  {home_team_name} passes: {len(results_df[results_df['Team'] == 'Home'])}")
    print(f"  {away_team_name} passes: {len(results_df[results_df['Team'] == 'Away'])}")
    print(f"  Period 1 passes: {len(results_df[results_df['Period'] == 1])}")
    print(f"  Period 2 passes: {len(results_df[results_df['Period'] == 2])}")
    print(f"  Average EPV-added: {results_df['EPV_Added'].mean():.6f}")
    print(f"  Max EPV-added: {results_df['EPV_Added'].max():.6f}")
    print(f"  Min EPV-added: {results_df['EPV_Added'].min():.6f}")
    
    # Show top 10 events
    print(f"\n🏆 Top 10 Events by EPV-Added:")
    top_10 = results_df.head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        team = home_team_name if row['Team'] == 'Home' else away_team_name
        from_player = row['From_Player']
        to_player = row['To_Player']
        epv_added = row['EPV_Added']
        period_time = row['Period_Time']
        period = row['Period']
        print(f"  {i:2d}. {team} {from_player} → {to_player} | EPV+: {epv_added:.3f} | P{period} {period_time}")
    
    print(f"\n🎯 CSV file contains columns:")
    print(f"  - Event_ID, PFF_Event_ID, Team, Type, Subtype")
    print(f"  - Period, Start_Time_s, Period_Time, Cumulative_Time")
    print(f"  - From_Player, To_Player")
    print(f"  - Start_X, Start_Y, End_X, End_Y")
    print(f"  - EPV_Added, EPV_Difference")
    
    print(f"\n" + "=" * 70)
    print("EPV RESULTS CSV GENERATION COMPLETE")
    print(f"Game ID: {game_id} | {home_team_name} vs {away_team_name}")
    print("=" * 70)

if __name__ == "__main__":
    main()
