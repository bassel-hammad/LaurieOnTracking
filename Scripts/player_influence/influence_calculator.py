"""
Player Influence Calculator module.

Calculates individual player contributions to pitch control changes.
"""

import numpy as np
import pandas as pd

from .pitch_control import PitchControlCalculator


class InfluenceCalculator:
    """Calculates individual player influence on pitch control."""
    
    def __init__(self, data_loader, pc_calculator, attacking_team='Home', analysis_mode='attacking'):
        """
        Initialize the influence calculator.
        
        Parameters
        ----------
        data_loader : DataLoader
            Data loader instance with loaded data
        pc_calculator : PitchControlCalculator
            Pitch control calculator instance
        attacking_team : str
            'Home' or 'Away'
        analysis_mode : str
            'attacking' - Analyze attackers using additive approach
            'defending' - Analyze defenders using necessity approach
        """
        self.data_loader = data_loader
        self.pc_calculator = pc_calculator
        self.attacking_team = attacking_team
        self.defending_team = 'Away' if attacking_team == 'Home' else 'Home'
        self.analysis_mode = analysis_mode
        
        self.influence_results = []
        self.attacker_influences = {}
        self.defender_influences = {}
    
    def analyze_sequence(self, frames_to_analyze, verbose=True):
        """
        Analyze player influence for a sequence using 1-second intervals.
        Starts at the sequence start time and analyzes every 1 second until the end.
        
        Parameters
        ----------
        frames_to_analyze : list
            List of frame numbers to determine sequence time range
        verbose : bool
            Print progress messages
            
        Returns
        -------
        list
            List of influence results for each 1-second interval
        """
        self.influence_results = []
        
        tracking_home = self.data_loader.tracking_home
        tracking_away = self.data_loader.tracking_away
        
        # Get the sequence time range
        start_frame = frames_to_analyze[0]
        end_frame = frames_to_analyze[-1]
        
        start_time = tracking_home.loc[start_frame, 'Time [s]']
        end_time = tracking_home.loc[end_frame, 'Time [s]']
        
        if hasattr(start_time, 'iloc'):
            start_time = float(start_time.iloc[0])
        if hasattr(end_time, 'iloc'):
            end_time = float(end_time.iloc[0])
        
        if verbose:
            print(f"  Sequence time range: {start_time:.2f}s - {end_time:.2f}s")
        
        # Generate 1-second intervals throughout the sequence
        current_time = start_time
        interval_count = 0
        
        while current_time + 1.0 <= end_time:
            if verbose and interval_count % 10 == 0:
                print(f"  Processing interval {interval_count+1} (t={current_time:.2f}s)...")
            
            time_t = current_time
            time_t1 = current_time + 1.0
            
            # Find closest frame to time_t
            time_col = tracking_home['Time [s]']
            time_diffs = np.abs(time_col - time_t)
            closest_frame = time_col.index[np.argmin(time_diffs)]
            
            result = self._analyze_time_transition(
                tracking_home, tracking_away, closest_frame, time_t, time_t1
            )
            
            if result:
                self.influence_results.append(result)
            
            current_time += 1.0
            interval_count += 1
        
        # Aggregate influences
        self._aggregate_influences()
        
        if verbose:
            print(f"\nAnalysis complete: {len(self.influence_results)} 1-second intervals analyzed")
            print(f"  Total duration: {end_time - start_time:.2f}s")
        
        return self.influence_results
    
    def _interpolate_row_at_time(self, tracking_data, target_time):
        """
        Interpolate player positions and velocities at a specific time.
        Uses linear interpolation between the surrounding frames.
        
        Parameters
        ----------
        tracking_data : DataFrame
            Tracking data (home or away)
        target_time : float
            Target time in seconds
            
        Returns
        -------
        Series
            Interpolated row at target_time
        """
        # Find frames bracketing target_time
        time_col = tracking_data['Time [s]']
        
        # Find closest frame before and after target_time
        before_mask = time_col <= target_time
        after_mask = time_col >= target_time
        
        if not before_mask.any():
            # Target time is before all data - use first frame
            return tracking_data.iloc[0].copy()
        
        if not after_mask.any():
            # Target time is after all data - use last frame
            return tracking_data.iloc[-1].copy()
        
        before_idx = time_col[before_mask].index[-1]
        after_idx = time_col[after_mask].index[0]
        
        # If exact match, return that row
        if before_idx == after_idx:
            return tracking_data.loc[before_idx].copy()
        
        # Get the two rows
        row_before = tracking_data.loc[before_idx]
        row_after = tracking_data.loc[after_idx]
        
        # Get times
        t_before = float(row_before['Time [s]'].iloc[0] if hasattr(row_before['Time [s]'], 'iloc') else row_before['Time [s]'])
        t_after = float(row_after['Time [s]'].iloc[0] if hasattr(row_after['Time [s]'], 'iloc') else row_after['Time [s]'])
        
        # Calculate interpolation weight
        if t_after == t_before:
            weight = 0
        else:
            weight = (target_time - t_before) / (t_after - t_before)
        
        # Create interpolated row
        interpolated = row_before.copy()
        
        # Interpolate numeric columns (positions, velocities, etc.)
        numeric_cols = tracking_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Period', 'Frame']:
                val_before = row_before[col]
                val_after = row_after[col]
                
                # Handle Series (duplicate indices)
                if hasattr(val_before, 'iloc'):
                    val_before = val_before.iloc[0]
                if hasattr(val_after, 'iloc'):
                    val_after = val_after.iloc[0]
                
                # Interpolate if both values are not NaN
                if pd.notna(val_before) and pd.notna(val_after):
                    interpolated[col] = val_before + weight * (val_after - val_before)
                elif pd.notna(val_before):
                    interpolated[col] = val_before
                elif pd.notna(val_after):
                    interpolated[col] = val_after
        
        # Update Time field
        interpolated['Time [s]'] = target_time
        
        return interpolated
    
    def _backfill_interpolated_row(self, row):
        """
        Backfill NaN values in an interpolated row.
        Mimics PitchControlCalculator.get_backfilled_row behavior.
        
        Parameters
        ----------
        row : Series
            Interpolated row
            
        Returns
        -------
        Series
            Backfilled row
        """
        row_bf = row.copy()
        
        # Get player IDs
        player_ids = [c.replace('_x', '') for c in row.index 
                     if c.endswith('_x') and c != 'ball_x']
        
        for player_id in player_ids:
            # Position
            if pd.isna(row[f'{player_id}_x']):
                row_bf[f'{player_id}_x'] = 0
                row_bf[f'{player_id}_y'] = 0
            
            # Velocity
            if pd.isna(row[f'{player_id}_vx']):
                row_bf[f'{player_id}_vx'] = 0
                row_bf[f'{player_id}_vy'] = 0
        
        return row_bf
    
    def _analyze_time_transition(self, tracking_home, tracking_away, frame_t, time_t, time_t1):
        """
        Analyze a time transition from time_t to time_t1 (always 1 second apart).
        
        Parameters
        ----------
        tracking_home : DataFrame
            Home team tracking data
        tracking_away : DataFrame
            Away team tracking data
        frame_t : int
            Frame number at time_t
        time_t : float
            Start time in seconds
        time_t1 : float
            End time in seconds (time_t + 1.0)
        """
        # Get row at time_t (use actual frame)
        home_t = tracking_home.loc[frame_t]
        away_t = tracking_away.loc[frame_t]
        
        # Interpolate positions at time_t1 (1 second later)
        home_t1 = self._interpolate_row_at_time(tracking_home, time_t1)
        away_t1 = self._interpolate_row_at_time(tracking_away, time_t1)
        
        # Get backfilled rows for time_t
        home_t_bf = PitchControlCalculator.get_backfilled_row(tracking_home, frame_t)
        away_t_bf = PitchControlCalculator.get_backfilled_row(tracking_away, frame_t)
        
        # For interpolated t1, we need to backfill manually
        home_t1_bf = self._backfill_interpolated_row(home_t1)
        away_t1_bf = self._backfill_interpolated_row(away_t1)
        
        # Baseline PC at time t
        PC_baseline, xgrid, ygrid = self.pc_calculator.calculate_surface(
            home_t_bf, away_t_bf, 
            attacking_half_only=True, 
            attacking_team=self.attacking_team
        )
        
        # Actual PC at time t+1
        PC_actual, _, _ = self.pc_calculator.calculate_surface(
            home_t1_bf, away_t1_bf,
            attacking_half_only=True,
            attacking_team=self.attacking_team
        )
        
        ΔPC_actual = PC_actual - PC_baseline
        
        # Set up team references
        if self.attacking_team == 'Home':
            attack_t, attack_t1 = home_t, home_t1
            attack_t_bf, attack_t1_bf = home_t_bf, home_t1_bf
            defend_t, defend_t1 = away_t, away_t1
            defend_t_bf, defend_t1_bf = away_t_bf, away_t1_bf
        else:
            attack_t, attack_t1 = away_t, away_t1
            attack_t_bf, attack_t1_bf = away_t_bf, away_t1_bf
            defend_t, defend_t1 = home_t, home_t1
            defend_t_bf, defend_t1_bf = home_t_bf, home_t1_bf
        
        # Get players
        attacking_players = self._get_active_players(attack_t)
        defending_players = self._get_active_players(defend_t)
        
        # Calculate individual influences
        player_influences = {}
        
        # For away team attacking, we need baseline in away perspective
        if self.attacking_team == 'Away':
            PC_baseline_attack = 1 - PC_baseline
            PC_actual_attack = 1 - PC_actual
        else:
            PC_baseline_attack = PC_baseline
            PC_actual_attack = PC_actual
        
        # Analyze based on mode
        if self.analysis_mode == 'attacking':
            # Attacking Analysis: Only analyze attackers using additive approach
            for player_id in attacking_players:
                influence = self._calculate_player_influence(
                    player_id, attack_t, attack_t1, attack_t_bf, attack_t1_bf,
                    defend_t_bf, defend_t1_bf, PC_baseline_attack, PC_actual_attack, 'attacking'
                )
                player_influences[player_id] = influence
        else:  # self.analysis_mode == 'defending'
            # Defending Analysis: Only analyze defenders using necessity approach
            for player_id in defending_players:
                influence = self._calculate_player_influence(
                    player_id, defend_t, defend_t1, defend_t_bf, defend_t1_bf,
                    attack_t_bf, attack_t1_bf, PC_baseline_attack, PC_actual_attack, 'defending'
                )
                player_influences[player_id] = influence
        
        # Calculate sum and interaction
        if len(player_influences) > 0:
            ΔPC_sum = np.nansum([p['delta_PC'] for p in player_influences.values()], axis=0)
        else:
            ΔPC_sum = np.zeros_like(PC_baseline)
        
        interaction = ΔPC_actual - ΔPC_sum
        
        return {
            'frame_t': frame_t,
            'frame_t1': None,  # Not using frame_t1 anymore (using interpolated time)
            'time_t': time_t,
            'time_t1': time_t1,
            'PC_baseline': PC_baseline,
            'PC_actual': PC_actual,
            'ΔPC_actual': ΔPC_actual,
            'ΔPC_sum': ΔPC_sum,
            'interaction': interaction,
            'player_influences': player_influences,
            'num_attacking_players': len(attacking_players),
            'num_defending_players': len(defending_players),
            'xgrid': xgrid,
            'ygrid': ygrid,
        }
    
    def _get_active_players(self, row):
        """Get list of active player IDs from a row."""
        player_cols = [c for c in row.keys() 
                      if c[-2:].lower() == '_x' and c != 'ball_x' 
                      and 'visibility' not in c.lower()]
        
        players = []
        for col in player_cols:
            player_id = col.replace('_x', '')
            if not pd.isna(row[col]):
                players.append(player_id)
        
        return players
    
    def _calculate_player_influence(self, player_id, row_t, row_t1, row_t_bf, row_t1_bf,
                                    opponent_row_bf, opponent_row_t1_bf, PC_baseline, PC_actual, team_type):
        """
        Calculate influence for a single player using mode-specific approach:
        
        Attacking mode (team_type='attacking'):
            Additive Approach: "What does this player's movement ADD?"
            - Hybrid: Only this player at t+1, everyone else at t
            - ΔPC = PC_hybrid - PC_baseline
        
        Defending mode (team_type='defending'):
            Necessity Approach: "What would we LOSE without this movement?"
            - Hybrid: Only this player at t, everyone else at t+1
            - ΔPC = PC_actual - PC_frozen
        """
        results = {}
        
        # Select approach based on team type
        use_additive = (team_type == 'attacking')
        use_necessity = (team_type == 'defending')
        
        ΔPC_additive = None
        ΔPC_necessity = None
        
        # =====================================================================
        # APPROACH 1: ADDITIVE (for attacking analysis)
        # "Move only this player to t+1, keep everyone else at t"
        # =====================================================================
        if use_additive:
            hybrid_additive = row_t_bf.copy()
            
            # Update position to t+1
            hybrid_additive[f'{player_id}_x'] = row_t1_bf[f'{player_id}_x']
            hybrid_additive[f'{player_id}_y'] = row_t1_bf[f'{player_id}_y']
            
            # Update velocities if available
            if f'{player_id}_vx' in row_t1_bf.keys():
                hybrid_additive[f'{player_id}_vx'] = row_t1_bf[f'{player_id}_vx']
                hybrid_additive[f'{player_id}_vy'] = row_t1_bf[f'{player_id}_vy']
                hybrid_additive[f'{player_id}_speed'] = row_t1_bf[f'{player_id}_speed']
            
            # Calculate PC with only this player moved forward
            if self.attacking_team == 'Home':
                PC_hybrid_add, _, _ = self.pc_calculator.calculate_surface(
                    hybrid_additive, opponent_row_bf,
                    attacking_half_only=True,
                    attacking_team=self.attacking_team
                )
            else:
                PC_hybrid_add, _, _ = self.pc_calculator.calculate_surface(
                    opponent_row_bf, hybrid_additive,
                    attacking_half_only=True,
                    attacking_team=self.attacking_team
                )
                PC_hybrid_add = 1 - PC_hybrid_add
            
            ΔPC_additive = PC_hybrid_add - PC_baseline
        
        # =====================================================================
        # APPROACH 2: NECESSITY (for defending analysis)
        # "Freeze only this player at t, move everyone else to t+1"
        # =====================================================================
        if use_necessity:
            hybrid_frozen = row_t1_bf.copy()
            
            # Keep this player at t (frozen)
            hybrid_frozen[f'{player_id}_x'] = row_t_bf[f'{player_id}_x']
            hybrid_frozen[f'{player_id}_y'] = row_t_bf[f'{player_id}_y']
            
            # Keep velocities at t if available
            if f'{player_id}_vx' in row_t_bf.keys():
                hybrid_frozen[f'{player_id}_vx'] = row_t_bf[f'{player_id}_vx']
                hybrid_frozen[f'{player_id}_vy'] = row_t_bf[f'{player_id}_vy']
                hybrid_frozen[f'{player_id}_speed'] = row_t_bf[f'{player_id}_speed']
            
            # Calculate PC with this player frozen at t
            # For defending analysis, we measure opponent's control
            if self.attacking_team == 'Home':
                # Attacking team is Home, so defenders are Away
                PC_frozen, _, _ = self.pc_calculator.calculate_surface(
                    opponent_row_t1_bf, hybrid_frozen,
                    attacking_half_only=True,
                    attacking_team=self.attacking_team
                )
            else:
                # Attacking team is Away, so defenders are Home
                PC_frozen, _, _ = self.pc_calculator.calculate_surface(
                    hybrid_frozen, opponent_row_t1_bf,
                    attacking_half_only=True,
                    attacking_team=self.attacking_team
                )
                PC_frozen = 1 - PC_frozen
            
            # Necessity = what we would lose without this movement
            # For defenders: if freezing them increases attacking PC, they were good at defending
            # So we INVERT the sign: negative ΔPC (attacks worse) = positive defender influence
            ΔPC_necessity = PC_frozen - PC_actual
        
        # Build results based on which approach was used
        if use_additive:
            ΔPC = ΔPC_additive
        else:  # use_necessity
            ΔPC = ΔPC_necessity
        
        return {
            # Primary metrics (unified interface)
            'delta_PC': ΔPC,
            'total_influence': np.nansum(np.abs(ΔPC)),
            'positive_influence': np.nansum(np.where(ΔPC > 0, ΔPC, 0)),
            'negative_influence': np.nansum(np.where(ΔPC < 0, ΔPC, 0)),
            'net_influence': np.nansum(ΔPC),
            
            # Detailed metrics by approach
            'delta_PC_additive': ΔPC_additive if use_additive else None,
            'total_additive': np.nansum(np.abs(ΔPC_additive)) if use_additive else None,
            'positive_additive': np.nansum(np.where(ΔPC_additive > 0, ΔPC_additive, 0)) if use_additive else None,
            'negative_additive': np.nansum(np.where(ΔPC_additive < 0, ΔPC_additive, 0)) if use_additive else None,
            'net_additive': np.nansum(ΔPC_additive) if use_additive else None,
            
            'delta_PC_necessity': ΔPC_necessity if use_necessity else None,
            'total_necessity': np.nansum(np.abs(ΔPC_necessity)) if use_necessity else None,
            'positive_necessity': np.nansum(np.where(ΔPC_necessity > 0, ΔPC_necessity, 0)) if use_necessity else None,
            'negative_necessity': np.nansum(np.where(ΔPC_necessity < 0, ΔPC_necessity, 0)) if use_necessity else None,
            'net_necessity': np.nansum(ΔPC_necessity) if use_necessity else None,
            
            # Position info
            'position_t': (row_t[f'{player_id}_x'], row_t[f'{player_id}_y']),
            'position_t1': (row_t1[f'{player_id}_x'], row_t1[f'{player_id}_y']),
            'team': team_type,
            'approach': 'additive' if use_additive else 'necessity'
        }
    
    def _aggregate_influences(self):
        """Aggregate player influences across all frames for BOTH approaches."""
        self.attacker_influences = {}
        self.defender_influences = {}
        
        for result in self.influence_results:
            for player_id, influence_data in result['player_influences'].items():
                team = influence_data.get('team', 'attacking')
                
                if team == 'attacking':
                    target = self.attacker_influences
                else:
                    target = self.defender_influences
                
                if player_id not in target:
                    target[player_id] = {
                        # Approach 1: Additive
                        'total_additive': 0.0,
                        'positive_additive': 0.0,
                        'negative_additive': 0.0,
                        'net_additive': 0.0,
                        # Approach 2: Necessity
                        'total_necessity': 0.0,
                        'positive_necessity': 0.0,
                        'negative_necessity': 0.0,
                        'net_necessity': 0.0,
                        # Legacy (for backward compatibility)
                        'total': 0.0,
                        'positive': 0.0,
                        'negative': 0.0,
                        'net': 0.0,
                        'frames': 0
                    }
                
                # Approach-specific aggregation
                if influence_data.get('total_additive') is not None:
                    target[player_id]['total_additive'] += influence_data['total_additive']
                    target[player_id]['positive_additive'] += influence_data['positive_additive']
                    target[player_id]['negative_additive'] += influence_data['negative_additive']
                    target[player_id]['net_additive'] += influence_data['net_additive']
                
                if influence_data.get('total_necessity') is not None:
                    target[player_id]['total_necessity'] += influence_data['total_necessity']
                    target[player_id]['positive_necessity'] += influence_data['positive_necessity']
                    target[player_id]['negative_necessity'] += influence_data['negative_necessity']
                    target[player_id]['net_necessity'] += influence_data['net_necessity']
                
                # Legacy (unified metrics - work for both approaches)
                target[player_id]['total'] += influence_data['total_influence']
                target[player_id]['positive'] += influence_data['positive_influence']
                target[player_id]['negative'] += influence_data['negative_influence']
                target[player_id]['net'] = target[player_id]['positive'] + target[player_id]['negative']
                target[player_id]['frames'] += 1
    
    def get_sorted_players(self, reverse=True):
        """Get players sorted by the metric appropriate for current analysis_mode.
        
        Returns
        -------
        list of tuples
            (player_id, stats_dict) sorted by net influence for current mode
        """
        if self.analysis_mode == 'attacking':
            # Attacking mode: sort attackers by net_additive
            return sorted(
                self.attacker_influences.items(),
                key=lambda x: x[1].get('net_additive', 0),
                reverse=reverse
            )
        else:
            # Defending mode: sort defenders by net_necessity
            return sorted(
                self.defender_influences.items(),
                key=lambda x: x[1].get('net_necessity', 0),
                reverse=reverse
            )
    
    def get_sorted_attackers(self, by='positive_additive', reverse=True):
        """Get attackers sorted by specified metric."""
        return sorted(
            self.attacker_influences.items(),
            key=lambda x: x[1].get(by, x[1].get('positive', 0)),
            reverse=reverse
        )
    
    def get_sorted_defenders(self, by='net_necessity', reverse=True):
        """Get defenders sorted by specified metric (defaults to net_necessity for defending mode)."""
        return sorted(
            self.defender_influences.items(),
            key=lambda x: x[1].get(by, x[1].get('negative', 0)),
            reverse=reverse
        )
    
    def get_interaction_stats(self):
        """Calculate interaction statistics."""
        total_interaction = np.nansum([
            np.nansum(np.abs(r['interaction'])) for r in self.influence_results
        ])
        total_actual_change = np.nansum([
            np.nansum(np.abs(r['ΔPC_actual'])) for r in self.influence_results
        ])
        
        interaction_pct = (total_interaction / total_actual_change * 100) if total_actual_change > 0 else 0
        
        return {
            'total_interaction': total_interaction,
            'total_actual_change': total_actual_change,
            'interaction_percentage': interaction_pct
        }
    
    def print_summary(self, data_loader):
        """Print summary statistics respecting the current analysis_mode."""
        print("=" * 90)
        
        if self.analysis_mode == 'attacking':
            print("SUMMARY STATISTICS - ATTACKING TEAM ANALYSIS")
            print("=" * 90)
            print()
            print("APPROACH: Additive - 'What does this player's movement ADD?'")
            print("          → Move only this player to t+1, keep everyone else at t")
            print()
            
            sorted_players = self.get_sorted_players(reverse=True)
            
            print(f"ATTACKING TEAM ({self.attacking_team}) - NET INFLUENCE (Additive)")
            print("=" * 80)
            print(f"{'Rank':<6} {'Player':<30} {'Net Influence':<15}")
            print("-" * 80)
            
            for rank, (player_id, stats) in enumerate(sorted_players, 1):
                name = data_loader.get_player_display_name(player_id)
                net_val = stats['net_additive']
                print(f"{rank:<6} {name:<30} {net_val:>14.2f}")
        else:
            print("SUMMARY STATISTICS - DEFENDING TEAM ANALYSIS")
            print("=" * 90)
            print()
            print("APPROACH: Necessity - 'What would we LOSE without this movement?'")
            print("          → Freeze only this player at t, move everyone else to t+1")
            print()
            
            sorted_players = self.get_sorted_players(reverse=True)
            
            print(f"DEFENDING TEAM ({self.defending_team}) - NET INFLUENCE (Necessity)")
            print("=" * 80)
            print(f"{'Rank':<6} {'Player':<30} {'Net Influence':<15}")
            print("-" * 80)
            
            for rank, (player_id, stats) in enumerate(sorted_players, 1):
                name = data_loader.get_player_display_name(player_id)
                net_val = stats['net_necessity']
                print(f"{rank:<6} {name:<30} {net_val:>14.2f}")
        
        print()
        
        # Interaction stats
        stats = self.get_interaction_stats()
        print(f"Total actual ΔPC (attacking half only): {stats['total_actual_change']:.3f}")
        print(f"Total interaction term: {stats['total_interaction']:.3f}")
        print(f"Interaction percentage: {stats['interaction_percentage']:.1f}%")
        print()
        
        if stats['interaction_percentage'] < 10:
            print("✓ Low interaction - individual contributions mostly independent")
        elif stats['interaction_percentage'] < 20:
            print("~ Moderate interaction - some player coupling effects")
        else:
            print("! High interaction - significant player coupling")
        
        print()
