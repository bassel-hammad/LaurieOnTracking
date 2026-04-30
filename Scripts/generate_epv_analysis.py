#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPV (Expected Possession Value) Analysis

This script calculates EPV added for each pass event in a match and generates
an Excel file with per-event EPV data and player aggregates.

Outputs an Excel file with:
  - EPV Per Event: EPV added for each pass event with player information
  - Total EPV: Aggregated EPV added by player

Usage:
    python generate_epv_analysis.py
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Metrica_IO as mio
import Metrica_EPV as mepv
import Metrica_PitchControl as mpc

from Scripts.player_influence.config import Config
from Scripts.player_influence.data_loader import DataLoader


class EPVAnalyzer:
    """Analyzes EPV (Expected Possession Value) for all pass events in a game."""
    
    def __init__(self, game_id):
        """
        Initialize the EPV analyzer.
        
        Parameters
        ----------
        game_id : int
            Match ID to analyze
        """
        self.game_id = game_id
        
        # Data containers
        self.data_loader = None
        self.events = None
        self.tracking_home = None
        self.tracking_away = None
        self.gk_numbers = None
        self.EPV = None
        self.params = None
        
        # Results storage
        self.epv_per_event = []  # List of dicts with event info and player EPV
        
        # Aggregated player EPV stats
        self.player_epv_stats = defaultdict(lambda: {
            'team': None,
            'epv_passing': 0.0,        # EPV credit for passing
            'epv_receiving': 0.0,      # EPV credit for receiving
            'epv_split': 0.0,          # EPV credit (50/50 split)
            'pass_count': 0,
            'receive_count': 0,
            'touch_count': 0,
            'shot_count': 0,
            'interception_count': 0,
        })
        
        # Player name to player_id mapping
        self.player_name_to_id = {}
    
    @staticmethod
    def _normalize_name_for_matching(name):
        """Normalize player name for matching, handling encoding issues."""
        import unicodedata
        
        # Try to fix double-encoding issues (UTF-8 bytes interpreted as Latin-1)
        try:
            # If the name was incorrectly decoded, re-encode as latin-1 and decode as utf-8
            fixed_name = name.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            # If that fails, use the original name
            fixed_name = name
        
        # Normalize unicode characters (e.g., é, á, ñ)
        normalized = unicodedata.normalize('NFKD', fixed_name)
        # Remove accents/diacritics
        ascii_name = ''.join(c for c in normalized if not unicodedata.combining(c))
        # Convert to lowercase and remove extra spaces
        return ascii_name.lower().strip()
    
    def load_data(self):
        """Load all match data."""
        print("=" * 70)
        print(f"LOADING DATA FOR MATCH {self.game_id} (BOTH PERIODS)")
        print("=" * 70)
        
        self.data_loader = DataLoader(self.game_id)
        self.events, self.tracking_home, self.tracking_away = self.data_loader.load_all()
        self.gk_numbers = self.data_loader.gk_numbers
        
        print(f"\nAnalyzing both periods (full match)")
        print(f"Total events: {len(self.events)}")
        print(f"Tracking frames: Home={len(self.tracking_home)}, Away={len(self.tracking_away)}")
        
        # Load EPV grid
        print("Loading EPV grid...")
        self.EPV = mepv.load_EPV_grid('EPV_grid.csv')
        
        # Get pitch control parameters
        self.params = mpc.default_model_params()
        
        print(f"Loaded {len(self.events)} events")
        print(f"Home GK: #{self.gk_numbers[0]}, Away GK: #{self.gk_numbers[1]}")
        
        # Build player name to player_id mapping
        self._build_player_name_mapping()
    
    def _build_player_name_mapping(self):
        """Build mapping from player names in events to player_ids (Team_Jersey) using PFF tracking data."""
        print("\nBuilding player name to player_id mapping from PFF tracking data...")
        
        # Load player names directly from PFF tracking JSONL file
        tracking_file = os.path.join('PFF Data', 'Tracking Data', f'{self.game_id}.jsonl')
        
        if not os.path.exists(tracking_file):
            print(f"Warning: Tracking file not found: {tracking_file}")
            print("Using empty player mapping")
            return
        
        import json
        jersey_to_name = {'Home': {}, 'Away': {}}
        
        # Extract player names from tracking data (same method as PFF adapter)
        with open(tracking_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    frame = json.loads(line.strip())
                    
                    # Check game_event for player info
                    if 'game_event' in frame and frame['game_event']:
                        ge = frame['game_event']
                        shirt_number = ge.get('shirt_number')
                        player_name = ge.get('player_name')
                        is_home = ge.get('home_team', 0)
                        
                        if shirt_number and player_name:
                            team = 'Home' if is_home else 'Away'
                            jersey_to_name[team][str(shirt_number)] = player_name
                            
                except json.JSONDecodeError:
                    continue
        
        print(f"Extracted {len(jersey_to_name['Home'])} home players, {len(jersey_to_name['Away'])} away players from tracking data")
        
        # Build reverse mapping: player_name -> player_id (Team_Jersey)
        for team in ['Home', 'Away']:
            for jersey, name in jersey_to_name[team].items():
                player_id = f"{team}_{jersey}"
                self.player_name_to_id[name] = player_id
                # Also add normalized version for fuzzy matching
                normalized_name = self._normalize_name_for_matching(name)
                self.player_name_to_id[normalized_name] = player_id
        
        print(f"Mapped {len(self.player_name_to_id)} player names to player_ids (including normalized versions)")
        
        # Show sample mappings
        sample_count = min(10, len(self.player_name_to_id))
        print(f"\nSample mappings (showing {sample_count}):")
        for name, pid in sorted(self.player_name_to_id.items(), key=lambda x: x[1])[:sample_count]:
            print(f"  {name} -> {pid}")
    
    def _get_player_id_from_name(self, player_name):
        """Convert player name to player_id, or return None if not found."""
        # Try exact match first
        if player_name in self.player_name_to_id:
            return self.player_name_to_id[player_name]
        
        # Try normalized match
        normalized_name = self._normalize_name_for_matching(player_name)
        if normalized_name in self.player_name_to_id:
            return self.player_name_to_id[normalized_name]
        
        return None
    
    def _get_display_name(self, player_id):
        """Get display name for a player_id (reverse lookup to get real name)."""
        # Look for real name
        for name, pid in self.player_name_to_id.items():
            if pid == player_id:
                return name
        
        # Fall back to jersey number format
        if '_' in player_id:
            return f"#{player_id.split('_')[1]}"
        return player_id
    
    def calculate_epv_for_events(self):
        """Calculate EPV added for each pass event."""
        print("\n" + "=" * 70)
        print("CALCULATING EPV FOR PASS EVENTS")
        print("=" * 70)
        
        pass_events = self.events[self.events['Type'] == 'PASS']
        print(f"Found {len(pass_events)} pass events")
        
        epv_count = 0
        failed_count = 0
        
        for idx in pass_events.index:
            try:
                event = self.events.loc[idx]
                player_from = event['From']
                player_to = event.get('To', None)
                team = event['Team']
                
                if pd.isna(player_from) or player_from == 'Unknown Player':
                    continue
                
                # Convert passer name to player_id
                passer_id = self._get_player_id_from_name(player_from)
                if not passer_id:
                    continue
                
                # Convert receiver name to player_id (if available)
                receiver_id = None
                if pd.notna(player_to) and player_to != 'Unknown Player':
                    receiver_id = self._get_player_id_from_name(player_to)
                
                # Calculate EPV added (pass homeTeamStartLeft from data_loader)
                homeTeamStartLeft = self.data_loader.homeTeamStartLeft if self.data_loader else True
                epv_added, epv_diff = mepv.calculate_epv_added(
                    idx, self.events, 
                    self.tracking_home, self.tracking_away,
                    self.gk_numbers, self.EPV, self.params,
                    homeTeamStartLeft=homeTeamStartLeft
                )
                
                if not np.isnan(epv_added):
                    # Calculate receiving EPV from grid value at end position
                    end_x = event.get('End X')
                    end_y = event.get('End Y')
                    epv_receiving = 0.0
                    
                    if pd.notna(end_x) and pd.notna(end_y):
                        # Determine attack direction for this team
                        if (team == 'Home' and homeTeamStartLeft) or (team == 'Away' and not homeTeamStartLeft):
                            attack_direction = 1  # attacking left to right
                        else:
                            attack_direction = -1  # attacking right to left
                        
                        # Get EPV grid value at receiving location
                        epv_receiving = mepv.get_EPV_at_location((end_x, end_y), self.EPV, attack_direction=attack_direction)
                    
                    # Store per-event EPV
                    event_epv_data = {
                        'event_index': idx,
                        'sequence': event.get('Sequence'),
                        'time': round(event.get('Start Time [s]'), 2),
                        'team': team,
                        'passer_id': passer_id,
                        'passer_name': player_from,
                        'receiver_id': receiver_id,
                        'receiver_name': player_to if pd.notna(player_to) else None,
                        'pass_type': event.get('Subtype', 'PASS'),
                        'epv_added': round(epv_added, 6),
                        'epv_receiving_grid': round(epv_receiving, 6),
                        'start_x': round(event.get('Start X'), 2) if pd.notna(event.get('Start X')) else None,
                        'start_y': round(event.get('Start Y'), 2) if pd.notna(event.get('Start Y')) else None,
                        'end_x': round(event.get('End X'), 2) if pd.notna(event.get('End X')) else None,
                        'end_y': round(event.get('End Y'), 2) if pd.notna(event.get('End Y')) else None,
                    }
                    self.epv_per_event.append(event_epv_data)
                    
                    # Credit passer (full EPV for passing)
                    if self.player_epv_stats[passer_id]['team'] is None:
                        self.player_epv_stats[passer_id]['team'] = team
                    self.player_epv_stats[passer_id]['epv_passing'] += epv_added
                    self.player_epv_stats[passer_id]['epv_split'] += (epv_added + epv_receiving) / 2.0
                    self.player_epv_stats[passer_id]['pass_count'] += 1
                    
                    # Credit receiver (if identified)
                    if receiver_id:
                        if self.player_epv_stats[receiver_id]['team'] is None:
                            self.player_epv_stats[receiver_id]['team'] = team
                        self.player_epv_stats[receiver_id]['epv_receiving'] += epv_receiving
                        self.player_epv_stats[receiver_id]['epv_split'] += (epv_added + epv_receiving) / 2.0
                        self.player_epv_stats[receiver_id]['receive_count'] += 1
                    
                    epv_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                # Skip events that fail EPV calculation
                failed_count += 1
                continue
        
        print(f"\nResults:")
        print(f"  Successfully calculated EPV: {epv_count} passes")
        print(f"  Failed/skipped: {failed_count} passes")
        print(f"  Total players with EPV: {len(self.player_epv_stats)}")
        
        # Count additional event types for all players
        self._count_additional_events()
    
    def _count_additional_events(self):
        """Count touches, shots, and interceptions for each player."""
        print("\nCounting touches, shots, and interceptions...")
        
        touch_count = 0
        shot_count = 0
        interception_count = 0
        
        # Debug: count raw shots in events
        raw_shot_events = self.events[self.events['Type'] == 'SHOT']
        print(f"  Raw SHOT events in data: {len(raw_shot_events)}")
        
        # Track shots that fail player matching
        unmatched_shots = []
        
        for idx in self.events.index:
            event = self.events.loc[idx]
            event_type = event.get('Type', '')
            event_subtype = event.get('Subtype', '')
            
            # Get player involved in the event
            player_name = None
            if 'From' in event and pd.notna(event['From']) and event['From'] != 'Unknown Player':
                player_name = event['From']
            elif 'Player' in event and pd.notna(event['Player']) and event['Player'] != 'Unknown Player':
                player_name = event['Player']
            
            if not player_name:
                if event_type == 'SHOT':
                    unmatched_shots.append(f"No player name (From={event.get('From')}, Player={event.get('Player')})")
                continue
            
            # Convert player name to player_id
            player_id = self._get_player_id_from_name(player_name)
            if not player_id:
                if event_type == 'SHOT':
                    unmatched_shots.append(f"{player_name} (not found in mapping)")
                continue
            
            team = event.get('Team')
            
            # Ensure player exists in stats (initialize if needed)
            if self.player_epv_stats[player_id]['team'] is None:
                self.player_epv_stats[player_id]['team'] = team
            
            # Count touches (any ball event where player touches the ball)
            # Include: PASS, SHOT, TOUCH, BALL_CARRY, CROSS, CLEARANCE, CHALLENGE, REBOUND
            if event_type in ['PASS', 'SHOT', 'TOUCH', 'BALL_CARRY', 'CROSS', 'CLEARANCE', 'CHALLENGE', 'REBOUND']:
                self.player_epv_stats[player_id]['touch_count'] += 1
                touch_count += 1
            
            # Count shots (SHOT events)
            if event_type == 'SHOT':
                self.player_epv_stats[player_id]['shot_count'] += 1
                shot_count += 1
            
            # Count interceptions (CHALLENGE events with GROUND-WON subtype)
            if event_type == 'CHALLENGE' and 'GROUND-WON' in str(event_subtype):
                self.player_epv_stats[player_id]['interception_count'] += 1
                interception_count += 1
        
        print(f"  Total touches counted: {touch_count}")
        print(f"  Total shots counted: {shot_count}")
        print(f"  Total interceptions counted: {interception_count}")
        
        if unmatched_shots:
            print(f"\n  WARNING: {len(unmatched_shots)} shots could not be matched to players:")
            for reason in unmatched_shots:
                print(f"    - {reason}")
    
    def export_to_excel(self, output_path=None):
        """Export EPV results to an Excel file."""
        if output_path is None:
            output_path = f'epv_analysis_match_{self.game_id}.xlsx'
        
        print("\n" + "=" * 70)
        print(f"EXPORTING RESULTS TO: {output_path}")
        print("=" * 70)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: EPV Per Event
            if self.epv_per_event:
                df_epv_events = pd.DataFrame(self.epv_per_event)
                # Sort by event index
                df_epv_events = df_epv_events.sort_values('event_index')
                df_epv_events.to_excel(writer, sheet_name='EPV Per Event', index=False)
                print(f"  - EPV Per Event: {len(df_epv_events)} pass events")
            
            # Sheet 2: Total EPV by Player
            player_ids = sorted(self.player_epv_stats.keys(), 
                              key=lambda x: (self.player_epv_stats[x]['team'] or '', x))
            
            total_epv_data = []
            for player_id in player_ids:
                stats = self.player_epv_stats[player_id]
                player_name = self._get_display_name(player_id)
                
                epv_combined = stats['epv_passing'] + stats['epv_receiving']
                
                total_epv_data.append({
                    'Player ID': player_id,
                    'Player Name': player_name,
                    'Team': stats['team'],
                    'Pass Count': stats['pass_count'],
                    'Receive Count': stats['receive_count'],
                    'Total Touches': stats['touch_count'],
                    'Total Shots': stats['shot_count'],
                    'Total Interceptions': stats['interception_count'],
                    'EPV Passing': round(stats['epv_passing'], 6),
                    'EPV Receiving': round(stats['epv_receiving'], 6),
                    'EPV Combined (Pass+Receive)': round(epv_combined, 6),
                    'EPV Split (50/50)': round(stats['epv_split'], 6),
                    'Avg EPV per Pass': round(stats['epv_passing'] / stats['pass_count'], 6) if stats['pass_count'] > 0 else 0.0,
                    'Avg EPV per Reception': round(stats['epv_receiving'] / stats['receive_count'], 6) if stats['receive_count'] > 0 else 0.0,
                    'EPV Combined per Touch': round(epv_combined / stats['touch_count'], 6) if stats['touch_count'] > 0 else 0.0
                })
            
            df_total_epv = pd.DataFrame(total_epv_data)
            df_total_epv.to_excel(writer, sheet_name='Total EPV by Player', index=False)
            print(f"  - Total EPV by Player: {len(df_total_epv)} players")
            
            # Sheet 3: EPV Summary Statistics
            if self.epv_per_event:
                home_events = [e for e in self.epv_per_event if e['team'] == 'Home']
                away_events = [e for e in self.epv_per_event if e['team'] == 'Away']
                
                home_epv_total = sum(e['epv_added'] for e in home_events)
                away_epv_total = sum(e['epv_added'] for e in away_events)
                
                # Calculate team totals for each approach
                home_players = [pid for pid, s in self.player_epv_stats.items() if s['team'] == 'Home']
                away_players = [pid for pid, s in self.player_epv_stats.items() if s['team'] == 'Away']
                
                home_epv_passing = sum(self.player_epv_stats[pid]['epv_passing'] for pid in home_players)
                away_epv_passing = sum(self.player_epv_stats[pid]['epv_passing'] for pid in away_players)
                home_epv_receiving = sum(self.player_epv_stats[pid]['epv_receiving'] for pid in home_players)
                away_epv_receiving = sum(self.player_epv_stats[pid]['epv_receiving'] for pid in away_players)
                home_epv_split = sum(self.player_epv_stats[pid]['epv_split'] for pid in home_players)
                away_epv_split = sum(self.player_epv_stats[pid]['epv_split'] for pid in away_players)
                
                summary_data = [
                    {'Metric': 'Total Pass Events Analyzed', 'Home': len(home_events), 'Away': len(away_events)},
                    {'Metric': '', 'Home': '', 'Away': ''},
                    {'Metric': 'EPV PASSING (Credit to Passer)', 'Home': '', 'Away': ''},
                    {'Metric': '  Total EPV Passing', 'Home': round(home_epv_passing, 6), 'Away': round(away_epv_passing, 6)},
                    {'Metric': '  Avg EPV per Pass', 
                     'Home': round(home_epv_passing / len(home_events), 6) if home_events else 0.0,
                     'Away': round(away_epv_passing / len(away_events), 6) if away_events else 0.0},
                    {'Metric': '', 'Home': '', 'Away': ''},
                    {'Metric': 'EPV RECEIVING (Credit to Receiver)', 'Home': '', 'Away': ''},
                    {'Metric': '  Total EPV Receiving', 'Home': round(home_epv_receiving, 6), 'Away': round(away_epv_receiving, 6)},
                    {'Metric': '', 'Home': '', 'Away': ''},
                    {'Metric': 'EPV SPLIT (50/50 Between Both)', 'Home': '', 'Away': ''},
                    {'Metric': '  Total EPV Split', 'Home': round(home_epv_split, 6), 'Away': round(away_epv_split, 6)},
                    {'Metric': '', 'Home': '', 'Away': ''},
                    {'Metric': 'Max EPV Added (single pass)', 
                     'Home': round(max((e['epv_added'] for e in home_events), default=0.0), 6),
                     'Away': round(max((e['epv_added'] for e in away_events), default=0.0), 6)},
                    {'Metric': 'Min EPV Added (single pass)', 
                     'Home': round(min((e['epv_added'] for e in home_events), default=0.0), 6),
                     'Away': round(min((e['epv_added'] for e in away_events), default=0.0), 6)},
                ]
                
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary Statistics', index=False)
                print(f"  - Summary Statistics: Team-level EPV metrics")
        
        print(f"\nExport complete: {output_path}")
        return output_path


def get_user_inputs():
    """Get user input for match ID."""
    print("=" * 70)
    print("EPV (EXPECTED POSSESSION VALUE) ANALYSIS")
    print("=" * 70)
    print()
    
    # Get match ID
    game_id = input("Enter match ID (e.g., 3822, 10517): ").strip()
    if not game_id:
        print("ERROR: Match ID is required!")
        sys.exit(1)
    
    try:
        game_id = int(game_id)
    except ValueError:
        print("ERROR: Match ID must be a number!")
        sys.exit(1)
    
    print(f"\nConfiguration:")
    print(f"  Match ID: {game_id}")
    print(f"  Analysis: EPV added for all pass events")
    print()
    
    return game_id


def main():
    """Main entry point."""
    # Get user inputs
    game_id = get_user_inputs()
    
    # Create analyzer
    analyzer = EPVAnalyzer(game_id)
    
    # Load data
    analyzer.load_data()
    
    # Calculate EPV for events
    analyzer.calculate_epv_for_events()
    
    # Export to Excel
    output_path = analyzer.export_to_excel()
    
    print("\n" + "=" * 70)
    print("EPV ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_path}")
    print("\nYou can now open the Excel file to explore:")
    print("  - EPV Per Event: Detailed EPV for each pass with passer and receiver")
    print("  - Total EPV by Player: Three EPV metrics (Passing, Receiving, Split)")
    print("  - Summary Statistics: Team-level EPV metrics for all three approaches")


if __name__ == '__main__':
    main()
