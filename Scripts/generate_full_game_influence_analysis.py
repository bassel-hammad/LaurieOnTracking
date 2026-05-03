#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-Game Player Influence Analysis with Event Correlation

This script calculates player influence (additive for attacking, necessity for defending)
for all players across an entire game, grouped by possession sequences.

It also extracts event statistics (touches, shots, interceptions, blocks) and calculates
EPV added/prevented for each player.

Outputs an Excel file with multiple sheets for comprehensive analysis.

Usage:
    python generate_full_game_influence_analysis.py
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc

from Scripts.player_influence.epv_weighting import EPVWeighting

from Scripts.player_influence.config import Config
from Scripts.player_influence.data_loader import DataLoader
from Scripts.player_influence.pitch_control import PitchControlCalculator
from Scripts.player_influence.influence_calculator import InfluenceCalculator


class FullGameInfluenceAnalyzer:
    """Analyzes player influence across an entire game."""
    
    def __init__(self, game_id, sample_interval=1.0, influence_type='additive', weighting_mode='original'):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        game_id : int
            Match ID to analyze
        sample_interval : float
            Time interval for influence sampling (1, 2, or 3 seconds)
        influence_type : str
            Type of influence to calculate ('additive' or 'necessity')
        """
        self.game_id = game_id
        self.sample_interval = sample_interval
        self.influence_type = influence_type
        # weighting_mode: 'original' | 'epv'
        self.weighting_mode = weighting_mode
        self.epv_grid = None
        
        # Data containers
        self.data_loader = None
        self.events = None
        self.tracking_home = None
        self.tracking_away = None
        self.gk_numbers = None
        self.params = None
        
        # Results storage by period
        self.home_period1_transitions = []
        self.home_period2_transitions = []
        self.away_period1_transitions = []
        self.away_period2_transitions = []
        
        # Aggregated player stats
        self.player_stats = defaultdict(lambda: {
            'team': None,
            'minutes_played': 0.0,
            'touches': 0,
            'shots': 0,
            'interceptions': 0,
            'blocks': 0,
            'passes': 0,
            'passes_completed': 0,
            'additive_influence_total': 0.0,
            'necessity_influence_total': 0.0,
            'sequences_attacking': 0,
            'sequences_defending': 0,
        })
        
        # Transition details for Excel export
        self.home_attacking_transitions = []
        self.home_defending_transitions = []
        self.away_attacking_transitions = []
        self.away_defending_transitions = []
        
        # Player name to player_id mapping
        self.player_name_to_id = {}
        
        # Tracking column name to jersey-based player_id mapping
        self.tracking_col_to_id = {}
    
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
        
        # Get pitch control parameters
        self.params = mpc.default_model_params()
        
        print(f"Loaded {len(self.events)} events")
        print(f"Home GK: #{self.gk_numbers[0]}, Away GK: #{self.gk_numbers[1]}")
        
        # Build player name to player_id mapping
        self._build_player_name_mapping()
        
        # Build tracking column to player_id mapping
        self._build_tracking_column_mapping()

        # Load EPV grid once if EPV weighting requested
        if getattr(self, 'weighting_mode', 'original') == 'epv':
            try:
                epv_w = EPVWeighting()
                self.epv_grid = epv_w.get_epv_grid(normalized=True)
                print(f"Loaded EPV grid for weighting: shape {self.epv_grid.shape}")
            except Exception as e:
                print(f"Warning: failed to load EPV grid for EPV weighting: {e}")
                self.epv_grid = None
    
    
    def extract_event_statistics(self):
        """Extract event statistics from the events data."""
        print("\n" + "=" * 70)
        print("EXTRACTING EVENT STATISTICS")
        print("=" * 70)
        
        for idx, event in self.events.iterrows():
            team = event['Team']
            event_type = event['Type']
            subtype = event.get('Subtype', '')
            player_from = event['From']
            player_to = event.get('To', '')
            
            if pd.isna(player_from) or player_from == 'Unknown Player':
                continue
            
            # Convert player name to player_id
            player_id = self._get_player_id_from_name(player_from)
            if not player_id:
                # Can't map this player, skip
                continue
            
            # Initialize player if needed
            if self.player_stats[player_id]['team'] is None:
                self.player_stats[player_id]['team'] = team
            
            # Count touches (all events where player has the ball)
            if event_type in ['TOUCH', 'PASS', 'SHOT', 'CROSS', 'CLEARANCE', 'BALL_CARRY']:
                self.player_stats[player_id]['touches'] += 1
            
            # Count shots
            if event_type == 'SHOT':
                self.player_stats[player_id]['shots'] += 1
            
            # Count passes
            if event_type == 'PASS':
                self.player_stats[player_id]['passes'] += 1
                if pd.isna(subtype) or subtype == '':
                    self.player_stats[player_id]['passes_completed'] += 1
            
            # Count blocks (when a player blocks)
            if subtype and 'BLOCKED' in str(subtype):
                # The blocker is credited - need to find from event context
                # For now, we'll count blocks when the event has BLOCKED subtype
                pass
            
            # Count interceptions (when opponent loses the ball to this team)
            if subtype == 'GROUND-LOST':
                # The next event's player gets the interception credit
                pass
        
        # Count interceptions by looking at possession changes
        sequences = self.events['Sequence'].dropna().unique()
        for i, seq in enumerate(sorted(sequences)):
            if i == 0:
                continue
            
            prev_seq = sorted(sequences)[i-1]
            prev_events = self.events[self.events['Sequence'] == prev_seq]
            curr_events = self.events[self.events['Sequence'] == seq]
            
            if len(prev_events) == 0 or len(curr_events) == 0:
                continue
            
            prev_team = prev_events.iloc[-1]['Team']
            curr_team = curr_events.iloc[0]['Team']
            curr_player = curr_events.iloc[0]['From']
            
            # If team changed, credit the first player of new sequence with interception
            if prev_team != curr_team and not pd.isna(curr_player) and curr_player != 'Unknown Player':
                curr_player_id = self._get_player_id_from_name(curr_player)
                if curr_player_id:
                    self.player_stats[curr_player_id]['interceptions'] += 1
        
        print(f"Extracted stats for {len(self.player_stats)} players")
    
    def calculate_minutes_played(self):
        """Calculate minutes played for each player based on tracking data presence."""
        print("\nCalculating minutes played...")
        
        # Only calculate for players we have in player_stats (from influence calculations)
        if not self.player_stats:
            print("No player stats available yet")
            return
        
        # Get all player columns from tracking data
        home_players = [c.replace('_x', '') for c in self.tracking_home.columns 
                       if c.endswith('_x') and c != 'ball_x']
        away_players = [c.replace('_x', '') for c in self.tracking_away.columns 
                       if c.endswith('_x') and c != 'ball_x']
        
        # Calculate time per frame
        time_col = self.tracking_home['Time [s]']
        total_time = time_col.max() - time_col.min()
        
        for player_id in home_players:
            x_col = f'{player_id}_x'
            if x_col in self.tracking_home.columns and player_id in self.player_stats:
                valid_frames = self.tracking_home[x_col].notna().sum()
                total_frames = len(self.tracking_home)
                minutes = (valid_frames / total_frames) * (total_time / 60.0)
                
                self.player_stats[player_id]['minutes_played'] = round(minutes, 1)
                if self.player_stats[player_id]['team'] is None:
                    self.player_stats[player_id]['team'] = 'Home'
        
        for player_id in away_players:
            x_col = f'{player_id}_x'
            if x_col in self.tracking_away.columns and player_id in self.player_stats:
                valid_frames = self.tracking_away[x_col].notna().sum()
                total_frames = len(self.tracking_away)
                minutes = (valid_frames / total_frames) * (total_time / 60.0)
                
                self.player_stats[player_id]['minutes_played'] = round(minutes, 1)
                if self.player_stats[player_id]['team'] is None:
                    self.player_stats[player_id]['team'] = 'Away'
    
    def _build_tracking_column_mapping(self):
        """Build mapping from tracking column names (Team_PlayerName) to jersey-based player_ids (Team_Jersey)."""
        print("\nBuilding tracking column to player_id mapping...")
        
        # Extract player columns from tracking data
        home_cols = [c.replace('_x', '') for c in self.tracking_home.columns 
                    if c.endswith('_x') and c != 'ball_x']
        away_cols = [c.replace('_x', '') for c in self.tracking_away.columns 
                    if c.endswith('_x') and c != 'ball_x']
        
        # For each tracking column, find its corresponding jersey-based ID
        for col_id in home_cols + away_cols:
            # col_id format: "Home_Julian Alvarez" or "Home_10"
            if '_' not in col_id:
                continue
            
            parts = col_id.split('_', 1)
            if len(parts) != 2:
                continue
            
            team, name_or_jersey = parts
            if team not in ['Home', 'Away']:
                continue
            
            # If it's already a jersey number, map to itself
            if name_or_jersey.isdigit():
                jersey_id = col_id
                self.tracking_col_to_id[col_id] = jersey_id
                continue
            
            # It's a player name - find corresponding jersey
            player_name = name_or_jersey.replace('_', ' ')  # Convert underscores to spaces
            
            # Try direct lookup in player_name_to_id
            if player_name in self.player_name_to_id:
                jersey_id = self.player_name_to_id[player_name]
                self.tracking_col_to_id[col_id] = jersey_id
            else:
                # Name doesn't match exactly - try normalized matching for encoding issues
                normalized_target = self._normalize_name_for_matching(player_name)
                found = False
                
                for pname, pid in self.player_name_to_id.items():
                    normalized_candidate = self._normalize_name_for_matching(pname)
                    # Match if normalized names are the same AND team matches
                    if normalized_candidate == normalized_target and pid.startswith(team):
                        self.tracking_col_to_id[col_id] = pid
                        found = True
                        break
                
                if not found:
                    # Can't map this column - use as-is
                    self.tracking_col_to_id[col_id] = col_id
        
        print(f"Mapped {len(self.tracking_col_to_id)} tracking columns to player_ids")
        
        # Show sample mappings
        sample_count = min(10, len(self.tracking_col_to_id))
        print(f"\nSample tracking column mappings (showing {sample_count}):")
        for col, pid in sorted(self.tracking_col_to_id.items(), key=lambda x: x[1])[:sample_count]:
            print(f"  {col} -> {pid}")
    
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
        
        print(f"Mapped {len(self.player_name_to_id)} player names to player_ids")
        
        # Show sample mappings
        sample_count = min(10, len(self.player_name_to_id))
        print(f"\nSample mappings (showing {sample_count}):")
        for name, pid in sorted(self.player_name_to_id.items(), key=lambda x: x[1])[:sample_count]:
            print(f"  {name} -> {pid}")
    
    
    def _get_player_id_from_name(self, player_name):
        """Convert player name to player_id, or return None if not found."""
        return self.player_name_to_id.get(player_name)
    
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
    

    def analyze_all_sequences(self):
        """Analyze player influence for all sequences in the game."""
        print("\n" + "=" * 70)
        print(f"ANALYZING {self.influence_type.upper()} INFLUENCE BY SEQUENCE (WEIGHTING: {self.weighting_mode.upper()})")
        print("=" * 70)
        
        # Create pitch control calculator
        homeTeamStartLeft = self.data_loader.homeTeamStartLeft if self.data_loader else True
        pc_calculator = PitchControlCalculator(self.gk_numbers, self.params, homeTeamStartLeft)
        
        # Process each period separately
        for period in [1, 2]:
            period_events = self.events[self.events['Period'] == period]
            sequences = sorted(period_events['Sequence'].dropna().unique())
            print(f"\nPeriod {period}: {len(sequences)} sequences")
            
            for seq_num in sequences:
                try:
                    self._analyze_sequence(seq_num, pc_calculator, period)
                except Exception as e:
                    continue
        
        print(f"\nCompleted analysis:")
        print(f"  Home Period 1: {len(self.home_period1_transitions)} transitions")
        print(f"  Home Period 2: {len(self.home_period2_transitions)} transitions")
        print(f"  Away Period 1: {len(self.away_period1_transitions)} transitions")
        print(f"  Away Period 2: {len(self.away_period2_transitions)} transitions")
    
    def _normalize_player_id(self, player_key):
        """Ensure player_key is in player_id format (Team_Jersey), not a name."""
        if not isinstance(player_key, str) or '_' not in player_key:
            return player_key
        
        # First check if this is a tracking column name we've already mapped
        if player_key in self.tracking_col_to_id:
            return self.tracking_col_to_id[player_key]
        
        parts = player_key.split('_', 1)  # Split only on first underscore
        if len(parts) != 2:
            return player_key
        
        team, rest = parts
        if team not in ['Home', 'Away']:
            return player_key
        
        # Check if rest is already a jersey number (pure digits)
        if rest.isdigit():
            return player_key  # Already in correct format (Team_Jersey)
        
        # rest contains a player name - look it up in player_name_to_id
        player_name = rest.replace('_', ' ')  # Convert underscores back to spaces
        if player_name in self.player_name_to_id:
            return self.player_name_to_id[player_name]
        
        # If we can't resolve it, return as-is
        return player_key
    
    def _analyze_sequence(self, sequence_number, pc_calculator, period):
        """Analyze a single sequence."""
        # Get sequence events
        seq_events = self.events[self.events['Sequence'] == sequence_number]
        
        if len(seq_events) == 0:
            return
        
        # Determine attacking team from first event
        attacking_team = seq_events.iloc[0]['Team']
        
        # Get frame range
        start_frame = int(seq_events['Start Frame'].min())
        end_frame = int(seq_events['End Frame'].max())
        
        # Ensure frames exist in tracking data
        if start_frame not in self.tracking_home.index or end_frame not in self.tracking_home.index:
            return
        
        # Get time range (handle potential duplicate indices)
        start_time_val = self.tracking_home.loc[start_frame, 'Time [s]']
        end_time_val = self.tracking_home.loc[end_frame, 'Time [s]']
        
        # Handle Series (duplicate indices) vs scalar
        if hasattr(start_time_val, 'iloc'):
            start_time = float(start_time_val.iloc[0])
        else:
            start_time = float(start_time_val)
        
        if hasattr(end_time_val, 'iloc'):
            end_time = float(end_time_val.iloc[0])
        else:
            end_time = float(end_time_val)
        
        duration = end_time - start_time
        if duration < self.sample_interval:
            return  # Sequence too short
        
        # Generate frames for analysis
        frames_to_analyze = list(range(start_frame, end_frame + 1))
        
        # Determine analysis mode and team based on influence type
        if self.influence_type == 'additive':
            # For additive: analyze attacking team
            analysis_mode = 'attacking'
            target_team = attacking_team
        else:
            # For necessity: analyze defending team
            analysis_mode = 'defending'
            target_team = 'Away' if attacking_team == 'Home' else 'Home'
        
        try:
            influence_calc = InfluenceCalculator(
                self.data_loader, pc_calculator,
                attacking_team=attacking_team,
                analysis_mode=analysis_mode,
                epv_grid=(self.epv_grid if getattr(self, 'weighting_mode', 'original') == 'epv' else None)
            )
            influence_calc.sample_interval = self.sample_interval
            
            results = influence_calc.analyze_sequence(frames_to_analyze, verbose=False)
            
            # Store results
            for result in results:
                transition_data = {
                    'sequence': sequence_number,
                    'time_t': result['time_t'],
                    'time_t1': result['time_t1'],
                }
                
                # Add individual player influences
                for player_key, influence in result['player_influences'].items():
                    player_id = self._normalize_player_id(player_key)
                    
                    # Select metric based on weighting mode and influence type
                    if self.weighting_mode == 'epv':
                        if self.influence_type == 'additive':
                            value = influence.get('epv_weighted_net_additive', 0.0)
                        else:
                            value = influence.get('epv_weighted_net_necessity', 0.0)
                    else:
                        # Original (uniform) weighting
                        if self.influence_type == 'additive':
                            value = influence['net_additive'] if influence['net_additive'] is not None else 0.0
                        else:
                            value = influence['net_necessity'] if influence['net_necessity'] is not None else 0.0
                    
                    transition_data[player_id] = value
                
                # Store in appropriate list based on team and period
                if target_team == 'Home':
                    if period == 1:
                        self.home_period1_transitions.append(transition_data)
                    else:
                        self.home_period2_transitions.append(transition_data)
                else:
                    if period == 1:
                        self.away_period1_transitions.append(transition_data)
                    else:
                        self.away_period2_transitions.append(transition_data)
                    
        except Exception as e:
            pass  # Skip failed sequences
    
    def export_to_excel(self, output_path=None):
        """Export all results to an Excel file."""
        if output_path is None:
            output_path = os.path.join('Metrica_Output', 'Full Match Analysis', f'player_influence_analysis_{self.influence_type}_{self.weighting_mode}_match_{self.game_id}.xlsx')
        
        print("\n" + "=" * 70)
        print(f"EXPORTING RESULTS TO: {output_path}")
        print("=" * 70)
        
        # Get all unique player IDs from all transitions
        all_player_ids = set()
        for trans in (self.home_period1_transitions + self.home_period2_transitions + 
                     self.away_period1_transitions + self.away_period2_transitions):
            for key in trans.keys():
                if key not in ['sequence', 'time_t', 'time_t1']:
                    all_player_ids.add(key)
        
        # Sort player IDs
        home_players = sorted([pid for pid in all_player_ids if pid.startswith('Home_')])
        away_players = sorted([pid for pid in all_player_ids if pid.startswith('Away_')])
        
        # Get display names
        home_player_names = {pid: self._get_display_name(pid) for pid in home_players}
        away_player_names = {pid: self._get_display_name(pid) for pid in away_players}
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Home Period 1
            if self.home_period1_transitions:
                rows = []
                for trans in self.home_period1_transitions:
                    row = {
                        'Sequence': trans['sequence'],
                        'Time_t': trans['time_t'],
                        'Time_t1': trans['time_t1'],
                    }
                    for pid in home_players:
                        row[home_player_names[pid]] = trans.get(pid, 0.0)
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name='Home Period 1', index=False)
                print(f"  - Home Period 1: {len(df)} transitions, {len(home_players)} players")
            
            # Sheet 2: Home Period 2
            if self.home_period2_transitions:
                rows = []
                for trans in self.home_period2_transitions:
                    row = {
                        'Sequence': trans['sequence'],
                        'Time_t': trans['time_t'],
                        'Time_t1': trans['time_t1'],
                    }
                    for pid in home_players:
                        row[home_player_names[pid]] = trans.get(pid, 0.0)
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name='Home Period 2', index=False)
                print(f"  - Home Period 2: {len(df)} transitions, {len(home_players)} players")
            
            # Sheet 3: Away Period 1
            if self.away_period1_transitions:
                rows = []
                for trans in self.away_period1_transitions:
                    row = {
                        'Sequence': trans['sequence'],
                        'Time_t': trans['time_t'],
                        'Time_t1': trans['time_t1'],
                    }
                    for pid in away_players:
                        row[away_player_names[pid]] = trans.get(pid, 0.0)
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name='Away Period 1', index=False)
                print(f"  - Away Period 1: {len(df)} transitions, {len(away_players)} players")
            
            # Sheet 4: Away Period 2
            if self.away_period2_transitions:
                rows = []
                for trans in self.away_period2_transitions:
                    row = {
                        'Sequence': trans['sequence'],
                        'Time_t': trans['time_t'],
                        'Time_t1': trans['time_t1'],
                    }
                    for pid in away_players:
                        row[away_player_names[pid]] = trans.get(pid, 0.0)
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name='Away Period 2', index=False)
                print(f"  - Away Period 2: {len(df)} transitions, {len(away_players)} players")
            
            # Sheet 5: Home Period 1 Total
            if self.home_period1_transitions:
                totals = {pid: 0.0 for pid in home_players}
                for trans in self.home_period1_transitions:
                    for pid in home_players:
                        totals[pid] += trans.get(pid, 0.0)
                
                total_data = [{'Player': home_player_names[pid], 'Total Influence': round(totals[pid], 4)} 
                             for pid in home_players]
                df = pd.DataFrame(total_data)
                df.to_excel(writer, sheet_name='Home Period 1 Total', index=False)
                print(f"  - Home Period 1 Total: {len(home_players)} players")
            
            # Sheet 6: Home Period 2 Total
            if self.home_period2_transitions:
                totals = {pid: 0.0 for pid in home_players}
                for trans in self.home_period2_transitions:
                    for pid in home_players:
                        totals[pid] += trans.get(pid, 0.0)
                
                total_data = [{'Player': home_player_names[pid], 'Total Influence': round(totals[pid], 4)} 
                             for pid in home_players]
                df = pd.DataFrame(total_data)
                df.to_excel(writer, sheet_name='Home Period 2 Total', index=False)
                print(f"  - Home Period 2 Total: {len(home_players)} players")
            
            # Sheet 7: Away Period 1 Total
            if self.away_period1_transitions:
                totals = {pid: 0.0 for pid in away_players}
                for trans in self.away_period1_transitions:
                    for pid in away_players:
                        totals[pid] += trans.get(pid, 0.0)
                
                total_data = [{'Player': away_player_names[pid], 'Total Influence': round(totals[pid], 4)} 
                             for pid in away_players]
                df = pd.DataFrame(total_data)
                df.to_excel(writer, sheet_name='Away Period 1 Total', index=False)
                print(f"  - Away Period 1 Total: {len(away_players)} players")
            
            # Sheet 8: Away Period 2 Total
            if self.away_period2_transitions:
                totals = {pid: 0.0 for pid in away_players}
                for trans in self.away_period2_transitions:
                    for pid in away_players:
                        totals[pid] += trans.get(pid, 0.0)
                
                total_data = [{'Player': away_player_names[pid], 'Total Influence': round(totals[pid], 4)} 
                             for pid in away_players]
                df = pd.DataFrame(total_data)
                df.to_excel(writer, sheet_name='Away Period 2 Total', index=False)
                print(f"  - Away Period 2 Total: {len(away_players)} players")
        
        print(f"\nExport complete: {output_path}")
        return output_path


def get_user_inputs():
    """Get user inputs for match ID, sample interval, and influence type."""
    print("=" * 70)
    print("FULL-GAME PLAYER INFLUENCE ANALYSIS")
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
    
    # Get influence type
    print("\nSelect influence type to calculate:")
    print("  1. Additive Influence (attacking players)")
    print("  2. Necessity Influence (defending players)")
    
    influence_input = input("Enter your choice (1 or 2): ").strip()
    
    if influence_input == '1':
        influence_type = 'additive'
    elif influence_input == '2':
        influence_type = 'necessity'
    else:
        print("Invalid choice, defaulting to additive influence")
        influence_type = 'additive'
    
    # Get sample interval
    print("\nSelect sample interval for influence calculations:")
    print("  1. 1 second (most detailed, slowest)")
    print("  2. 2 seconds")
    print("  3. 3 seconds (least detailed, fastest)")
    
    interval_input = input("Enter your choice (1, 2, or 3): ").strip()
    
    if interval_input == '1':
        sample_interval = 1.0
    elif interval_input == '2':
        sample_interval = 2.0
    elif interval_input == '3':
        sample_interval = 3.0
    else:
        print("Invalid choice, defaulting to 1 second interval")
        sample_interval = 1.0

    # Get weighting mode
    print("\nSelect spatial weighting mode:")
    print("  1. Original (uniform integration)")
    print("  2. EPV-weighted (weight by EPV grid)")
    weighting_input = input("Enter your choice (1 or 2): ").strip()
    if weighting_input == '1':
        weighting_mode = 'original'
    elif weighting_input == '2':
        weighting_mode = 'epv'
    else:
        print("Invalid choice, defaulting to original weighting")
        weighting_mode = 'original'
    
    print(f"\nConfiguration:")
    print(f"  Match ID: {game_id}")
    print(f"  Influence Type: {influence_type.upper()}")
    print(f"  Sample Interval: {sample_interval} seconds")
    print(f"  Weighting Mode: {weighting_mode.upper()}")
    print()
    return game_id, sample_interval, influence_type, weighting_mode


def main():
    """Main entry point."""
    # Get user inputs
    game_id, sample_interval, influence_type, weighting_mode = get_user_inputs()
    
    # Create analyzer
    analyzer = FullGameInfluenceAnalyzer(game_id, sample_interval, influence_type, weighting_mode=weighting_mode)
    
    # Load data
    analyzer.load_data()
    
    # Analyze all sequences
    analyzer.analyze_all_sequences()
    
    # Export to Excel
    output_path = analyzer.export_to_excel()
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_path}")
    print(f"\nExcel file contains {influence_type.upper()} influence with 8 sheets:")
    print("  Per-transition data:")
    print("    - Home Period 1")
    print("    - Home Period 2")
    print("    - Away Period 1")
    print("    - Away Period 2")
    print("  Total influence per player:")
    print("    - Home Period 1 Total")
    print("    - Home Period 2 Total")
    print("    - Away Period 1 Total")
    print("    - Away Period 2 Total")


if __name__ == '__main__':
    main()
