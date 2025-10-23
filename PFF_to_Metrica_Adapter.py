#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PFF to Metrica Data Format Adapter

Converts PFF data format (FIFA World Cup Final 2022: Argentina vs France) 
to Metrica Sports format for compatibility with LaurieOnTracking tutorials.

Author: Automated conversion for tutorial compatibility
Date: 2024
"""

import pandas as pd
import json
import numpy as np
import os
from datetime import datetime

class PFFToMetricaAdapter:
    def __init__(self, pff_data_dir, output_dir, game_id="10517"):
        """
        Initialize PFF to Metrica adapter
        
        Parameters:
        -----------
        pff_data_dir : str
            Path to PFF data directory
        output_dir : str 
            Path to output directory for Metrica format files
        game_id : str
            Game identifier 
        """
        self.pff_data_dir = pff_data_dir
        self.output_dir = output_dir
        self.game_id = game_id
        self.field_length = 105.0  # meters (from metadata)
        self.field_width = 68.0    # meters (from metadata)
        self.original_fps = 29.97  # from metadata
        self.target_fps = 25.0     # Metrica standard
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"PFF to Metrica Adapter initialized")
        print(f"Input: {pff_data_dir}")
        print(f"Output: {output_dir}")
        print(f"Game: FIFA World Cup Final 2022 (ID: {game_id})")
        print(f"Processing: First 2 periods only (excludes extra time)")


    def normalize_coordinates(self, x_meters, y_meters):
        """
        Convert PFF coordinates (meters, center origin) to Metrica format (0-1 scale, top-left origin)
        
        PFF System: Center origin, meters, y increases upward (bottom-to-top)
        Metrica System: Top-left origin, 0-1 scale, y increases downward (top-to-bottom)
        """
        if pd.isna(x_meters) or pd.isna(y_meters):
            return np.nan, np.nan
        
        # Convert from center origin to top-left origin, then normalize to 0-1
        x_normalized = (x_meters + self.field_length/2) / self.field_length
        
        # FLIP Y-AXIS: PFF y increases upward, Metrica y increases downward
        y_flipped = -y_meters  # Flip the y coordinate
        y_normalized = (y_flipped + self.field_width/2) / self.field_width
        
        # Clamp to 0-1 range
        x_normalized = max(0.0, min(1.0, x_normalized))
        y_normalized = max(0.0, min(1.0, y_normalized))
        
        return x_normalized, y_normalized


    def map_outcome_to_subtype(self, possession_data, event_type):
        """Map PFF outcomes to Metrica subtypes based on event type and outcome data"""
        try:
            # Get the appropriate outcome field based on event type
            outcome = None
            if event_type == 'PA':  # Pass
                outcome = possession_data.get('passOutcomeType', '')
            elif event_type == 'SH':  # Shot
                outcome = possession_data.get('shotOutcomeType', '')
            elif event_type == 'CR':  # Cross
                outcome = possession_data.get('crossOutcomeType', '')
            elif event_type == 'CH':  # Challenge
                outcome = possession_data.get('challengeOutcomeType', '')
            elif event_type == 'CL':  # Clearance
                outcome = possession_data.get('clearanceOutcomeType', '')
            elif event_type == 'RE':  # Rebound
                outcome = possession_data.get('reboundOutcomeType', '')
            elif event_type == 'TC':  # Touch
                outcome = possession_data.get('touchOutcomeType', '')
            
            if not outcome or pd.isna(outcome):
                return ''
            
            # CONTEXT-AWARE MAPPING: Different mappings for each event type
            # This prevents key collisions (e.g., 'C' means different things for passes vs shots)
            
            if event_type == 'PA':  # Pass outcomes
                pass_mapping = {
                    'C': '',                  # Complete
                    'D': 'GROUND-LOST',       # Defensive Interception
                    'O': 'OUT',               # Out of Play
                    'B': 'BLOCKED',           # Blocked
                    'I': 'OFF TARGET',        # Inadvertent Shot at Goal
                    'G': 'ON TARGET-GOAL',    # Inadvertent Own Goal
                    'S': 'STOPPAGE',          # Stoppage
                }
                return pass_mapping.get(outcome, '')
                
            elif event_type == 'SH':  # Shot outcomes
                shot_mapping = {
                    'G': 'ON TARGET-GOAL',      # Goal
                    'S': 'ON TARGET-SAVED',     # Save on target
                    'F': 'OFF TARGET-SAVED',    # Save off target (goalkeeper touched)
                    'O': 'OFF TARGET',          # Off target (no save)
                    'B': 'BLOCKED-ON TARGET',   # Block on target
                    'C': 'BLOCKED-OFF TARGET',  # Block off target
                    'L': 'BLOCKED-GOALLINE',    # Goalline clearance
                }
                return shot_mapping.get(outcome, '')
                
            elif event_type == 'CR':  # Cross outcomes
                cross_mapping = {
                    'C': 'PASS-CROSS',       # Complete cross
                    'D': 'GROUND-LOST',      # Intercepted
                    'O': 'OUT',              # Out of play
                    'B': 'BLOCKED',          # Blocked
                }
                return cross_mapping.get(outcome, '')
                
            elif event_type == 'CH':  # Challenge outcomes
                challenge_mapping = {
                    'D': 'GROUND-WON',       # Dribble won
                    'F': 'GROUND-LOST',      # Foul
                }
                return challenge_mapping.get(outcome, '')
                
            elif event_type == 'CL':  # Clearance outcomes
                clearance_mapping = {
                    'P': 'GROUND-WON',       # Successful clearance
                }
                return clearance_mapping.get(outcome, '')
                
            elif event_type == 'RE':  # Rebound outcomes
                rebound_mapping = {
                    'P': 'GROUND-WON',       # Successful rebound
                }
                return rebound_mapping.get(outcome, '')
                
            elif event_type == 'TC':  # Touch outcomes
                touch_mapping = {
                    'O': 'OUT',              # Out of play
                }
                return touch_mapping.get(outcome, '')
            
            # Default fallback for unknown event types
            return ''
            
        except Exception as e:
            return ''
        
    def map_possession_event_to_metrica(self, possession_type, possession_data):
        """Map PFF possession event type to Metrica event type"""
        mapping = {
            'PA': 'PASS',      # Pass
            'SH': 'SHOT',      # Shot
            'CR': 'CROSS',     # Cross
            'BC': 'BALL_CARRY', # Ball Carry
            'CH': 'CHALLENGE', # Challenge
            'CL': 'CLEARANCE', # Clearance
            'FO': 'FOUL',      # Foul
            'IT': 'TOUCH',     # Initial Touch
            'RE': 'REBOUND',   # Rebound
            'TC': 'TOUCH',     # Touch
        }
        return mapping.get(possession_type, '')

    def get_event_coordinates(self, event, position='start', next_event=None):
        """Extract coordinates from event tracking data with improved logic"""
        try:
            possession = event.get('possessionEvents', {})
            game_event = event.get('gameEvents', {})
            
            # Get ball coordinates as base reference
            ball_data = event.get('ball', [])
            ball_x, ball_y = 0.0, 0.0
            if ball_data and len(ball_data) > 0:
                ball = ball_data[0]
                ball_x = ball.get('x', 0.0)
                ball_y = ball.get('y', 0.0)
            
            # For start position, try to get player position first
            if position == 'start':
                player_coords = self.get_player_coordinates(event, possession, 'start')
                if player_coords:
                    return player_coords
                # Fallback to ball position
                normalized = self.normalize_coordinates(ball_x, ball_y)
                return normalized[0], normalized[1]
            
            # For end position, use different logic based on event type
            elif position == 'end':
                event_type = possession.get('possessionEventType', '')
                
                # For shots, use goal position
                if event_type == 'SH':
                    goal_coords = self.get_goal_coordinates(ball_x, ball_y)
                    return goal_coords
                
                # For all other events (passes, touches, challenges, etc.), use next event start position
                else:
                    if next_event is not None:
                        # Get the start coordinates of the next event (where ball ended up)
                        next_coords = self.get_event_coordinates(next_event, 'start')
                        if next_coords:
                            return next_coords
                    # Fallback to ball position if no next event
                    normalized = self.normalize_coordinates(ball_x, ball_y)
                    return normalized[0], normalized[1]
            
            # Default fallback
            normalized = self.normalize_coordinates(ball_x, ball_y)
            return normalized[0], normalized[1]
            
        except Exception as e:
            print(f"   Warning: Error extracting coordinates: {e}")
            return 0.0, 0.0

    def get_player_coordinates(self, event, possession, player_type='start'):
        """Get coordinates for a specific player involved in the event"""
        try:
            # Get player ID based on type
            player_id = None
            game_event = event.get('gameEvents', {})
            
            if player_type == 'start':
                # Always prioritize playerId from gameEvents (always populated)
                player_id = game_event.get('playerId')
                
                # If not found in gameEvents, fall back to possessionEvents fields
                if not player_id:
                    player_id = (possession.get('passerPlayerId') or 
                               possession.get('shooterPlayerId') or 
                               possession.get('carrierPlayerId') or 
                               possession.get('dribblerPlayerId') or 
                               possession.get('touchPlayerId'))
            elif player_type == 'receiver':
                player_id = possession.get('receiverPlayerId') or possession.get('targetPlayerId')
            
            if not player_id:
                return None
            
            # Determine if player is home or away
            is_home_team = game_event.get('homeTeam', False)
            players_data = event.get('homePlayers' if is_home_team else 'awayPlayers', [])
            
            # Find player in tracking data
            for player in players_data:
                if player.get('playerId') == player_id:
                    x = player.get('x', 0.0)
                    y = player.get('y', 0.0)
                    normalized = self.normalize_coordinates(x, y)
                    return normalized[0], normalized[1]
            
            return None
            
        except Exception as e:
            return None

    def get_goal_coordinates(self, ball_x, ball_y):
        """Get goal coordinates based on ball position and field orientation"""
        try:
            # Determine which goal based on x-coordinate
            if ball_x > 0:  # Attacking right goal
                goal_x = self.field_length / 2  # Goal line at +52.5m
            else:  # Attacking left goal  
                goal_x = -self.field_length / 2  # Goal line at -52.5m
            
            # Goal posts are 7.32m apart, so goal mouth is 3.66m from center
            goal_mouth_half_width = 3.66  # meters
            
            # Clamp the y-coordinate to be within the goal posts
            goal_y = max(-goal_mouth_half_width, min(goal_mouth_half_width, ball_y))
            
            normalized = self.normalize_coordinates(goal_x, goal_y)
            return normalized[0], normalized[1]
            
        except Exception as e:
            return 0.0, 0.0

    def get_next_event_coordinates(self, event):
        """Get coordinates from the next event in sequence (if available)"""
        # This would require access to the full event sequence
        # For now, return None to use fallback logic
        return None

    def convert_events_data(self):
        """Convert PFF event data to Metrica format"""
        print(" Converting event data...")
        
        # Load PFF events from JSON
        events_file = os.path.join(self.pff_data_dir, 'Event Data', f'{self.game_id}.json')
        with open(events_file, 'r', encoding='utf-8') as f:
            pff_data = json.load(f)
        
        print(f" Loaded JSON data with {len(pff_data)} events")
        
        # Extract event data from JSON structure
        events_list = []
        total_events = 0
        filtered_events = 0
        
        # Get team mapping from first event
        if pff_data and 'gameEvents' in pff_data[0]:
            first_event = pff_data[0]['gameEvents']
            home_team = first_event.get('teamName', 'Unknown')
            is_home = first_event.get('homeTeam', False)
            
            # Determine team mapping based on homeTeam field
            if is_home:
                team_mapping = {home_team: 'Home'}
                # Find away team from other events
                for event in pff_data[1:]:
                    if 'gameEvents' in event and event['gameEvents'].get('teamName') != home_team:
                        away_team = event['gameEvents']['teamName']
                        team_mapping[away_team] = 'Away'
                        break
            else:
                # If first team is not home, they are away
                team_mapping = {home_team: 'Away'}
                # Find home team from other events
                for event in pff_data[1:]:
                    if 'gameEvents' in event and event['gameEvents'].get('homeTeam', False):
                        actual_home_team = event['gameEvents']['teamName']
                        team_mapping[actual_home_team] = 'Home'
                        break
        else:
            team_mapping = {}
        
        print(f" Team mapping: {team_mapping}")
        
        # Process each event
        for i, event in enumerate(pff_data):
            if 'possessionEvents' not in event or 'gameEvents' not in event:
                continue
                
            possession = event['possessionEvents']
            game_event = event['gameEvents']
            total_events += 1
            
            # Skip non-event possessions
            if possession.get('nonEvent', False):
                continue
            
            # Only process events from first 2 periods (exclude extra time)
            period = game_event.get('period', 1)
            if period > 2:
                filtered_events += 1
                continue
            
            # Extract basic event info
            event_type = possession.get('possessionEventType', '')
            if not event_type:
                continue
                
            # Map possession event type to Metrica type
            metrica_type = self.map_possession_event_to_metrica(event_type, possession)
            if not metrica_type:
                continue
            
            # Get team info
            team_name = game_event.get('teamName', 'Unknown')
            team = team_mapping.get(team_name, 'Unknown')
            
            # Get period
            period = game_event.get('period', 1)
            
            # Get time (already in seconds)
            start_time = event.get('eventTime', 0.0)
            end_time = event.get('endTime', start_time)
            
            # Get player info from possessionEvents first, then fall back to gameEvents
            from_player = (possession.get('passerPlayerName') or 
                          possession.get('shooterPlayerName') or 
                          possession.get('carrierPlayerName') or 
                          possession.get('dribblerPlayerName') or 
                          possession.get('touchPlayerName') or 
                          possession.get('ballCarrierPlayerName') or 
                          game_event.get('playerName') or  # Fallback to gameEvents
                          'Unknown Player')
            to_player = (possession.get('receiverPlayerName') or 
                        possession.get('targetPlayerName') or 
                        'Unknown Player')
            
            # Get next event for pass end position (look ahead in the pff_data list)
            next_event = None
            for j in range(i + 1, len(pff_data)):
                next_candidate = pff_data[j]
                if 'possessionEvents' in next_candidate and 'gameEvents' in next_candidate:
                    next_poss = next_candidate['possessionEvents']
                    next_game = next_candidate['gameEvents']
                    # Use next event only if it's not a non-event and is in same period
                    if not next_poss.get('nonEvent', False) and next_game.get('period', 1) == period:
                        next_event = next_candidate
                        break
            
            # Get coordinates from tracking data if available
            start_x, start_y = self.get_event_coordinates(event, 'start', next_event)
            end_x, end_y = self.get_event_coordinates(event, 'end', next_event)
            
            # Create event record
            event_record = {
                'Team': team,
                'Type': metrica_type,
                'Subtype': self.map_outcome_to_subtype(possession, event_type),
                'Period': period,
                'Start Time [s]': start_time,
                'End Time [s]': end_time,
                'Start Frame': None,  # Will be filled later
                'End Frame': None,   # Will be filled later
                'From': from_player,
                'To': to_player,
                'PFF Event ID': event.get('gameEventId', i),
                'Start X': start_x,
                'Start Y': start_y,
                'End X': end_x,
                'End Y': end_y
            }
            
            events_list.append(event_record)
        
        # Convert to DataFrame
        metrica_events = pd.DataFrame(events_list)
        
        # Print filtering results
        print(f" Event filtering: {total_events} total events, {filtered_events} filtered out (periods > 2), {len(events_list)} processed")
        
        # Reorder columns to match Metrica format
        column_order = [
            'Team', 'Type', 'Subtype', 'Period', 'Start Frame', 'Start Time [s]',
            'End Frame', 'End Time [s]', 'From', 'To', 'Start X', 'Start Y', 'End X', 'End Y', 'PFF Event ID'
        ]
        metrica_events = metrica_events[column_order]
        
        print(f" Converted to {len(metrica_events)} Metrica events (frames will be mapped after tracking data)")
        return metrica_events
    
    def map_events_to_tracking_frames_direct(self, events_df, tracking_df):
        """
        Map events to tracking frames using GAME EVENT ID matching
        Each event has a gameEventId that corresponds to a game_event_id in tracking data
        """
        print(" Mapping events to tracking frames using GAME EVENT ID MATCHING...")
        
        mapped_events = events_df.copy()
        
        # Create a mapping from game_event_id to frame number in tracking data
        tracking_id_to_frame = {}
        for frame_idx, row in tracking_df.iterrows():
            game_event_id = row.get('game_event_id')
            if game_event_id is not None and not pd.isna(game_event_id):
                tracking_id_to_frame[int(game_event_id)] = frame_idx
        
        print(f"    Found {len(tracking_id_to_frame)} tracking frames with game_event_id")
        
        mapped_count = 0
        for idx, event in events_df.iterrows():
            event_id = event.get('PFF Event ID')
            if event_id is not None and event_id in tracking_id_to_frame:
                target_frame = tracking_id_to_frame[event_id]
                mapped_events.loc[idx, 'Start Frame'] = target_frame
                mapped_events.loc[idx, 'End Frame'] = target_frame
                mapped_count += 1
                
                # Debug info for first few events
                if idx < 5:
                    event_time = event['Start Time [s]']
                    tracking_time = tracking_df.loc[target_frame, 'Time [s]']
                    print(f"   Event {idx} (ID: {event_id}): {event_time:.1f}s  Frame {target_frame} [Track time: {tracking_time:.1f}s]")
            else:
                mapped_events.loc[idx, 'Start Frame'] = None
                mapped_events.loc[idx, 'End Frame'] = None
                if idx < 10:  # Show first 10 unmapped events for debugging
                    print(f"     Event {idx} (ID: {event_id}) not found in tracking data")
        
        # Convert to integers
        mapped_events['Start Frame'] = mapped_events['Start Frame'].astype('Int64')
        mapped_events['End Frame'] = mapped_events['End Frame'].astype('Int64')
        
        print(f" Successfully mapped {mapped_count}/{len(events_df)} events using GAME EVENT ID matching")
        
        return mapped_events

    def process_tracking_frame(self, frame_data):
        """Process a single frame of PFF tracking data"""
        if not frame_data:
            return None
        
        frame_info = {
            'Period': frame_data.get('period', 1),
            'Frame': frame_data.get('frameNum', 0),
            'Time [s]': frame_data.get('periodGameClockTime', 0.0),  # Use original time (lag corrected via frame mapping)
            'game_event_id': frame_data.get('game_event_id', None)
        }
        
        # Process home team players (Argentina) - RAW COORDINATES ONLY
        home_players_original = frame_data.get('homePlayers', [])
        
        for player in home_players_original:
            jersey = player.get('jerseyNum', '')
            
            # Always use original coordinates regardless of visibility
            x_meters = player.get('x', np.nan)
            y_meters = player.get('y', np.nan)
            visibility = player.get('visibility', 'UNKNOWN')  # Get visibility status
            
            x_norm, y_norm = self.normalize_coordinates(x_meters, y_meters)
            
            frame_info[f'Home_{jersey}_x'] = x_norm
            frame_info[f'Home_{jersey}_y'] = y_norm
            frame_info[f'Home_{jersey}_visibility'] = visibility  # Store visibility
        
        # Process away team players (France) - RAW COORDINATES ONLY
        away_players_original = frame_data.get('awayPlayers', [])
        
        for player in away_players_original:
            jersey = player.get('jerseyNum', '')
            
            # Always use original coordinates regardless of visibility
            x_meters = player.get('x', np.nan)
            y_meters = player.get('y', np.nan)
            visibility = player.get('visibility', 'UNKNOWN')  # Get visibility status
            
            x_norm, y_norm = self.normalize_coordinates(x_meters, y_meters)
            
            frame_info[f'Away_{jersey}_x'] = x_norm
            frame_info[f'Away_{jersey}_y'] = y_norm
            frame_info[f'Away_{jersey}_visibility'] = visibility  # Store visibility
        
        # Process ball - RAW COORDINATES ONLY
        balls_original = frame_data.get('balls', [])
        
        if balls_original and len(balls_original) > 0:
            ball_original = balls_original[0]
            
            # Always use original coordinates regardless of visibility
            ball_x = ball_original.get('x', np.nan)
            ball_y = ball_original.get('y', np.nan)
            
            ball_x_norm, ball_y_norm = self.normalize_coordinates(ball_x, ball_y)
            frame_info['ball_x'] = ball_x_norm
            frame_info['ball_y'] = ball_y_norm
        else:
            frame_info['ball_x'] = np.nan
            frame_info['ball_y'] = np.nan
        
        return frame_info

    def resample_tracking_data(self, tracking_df):
        """Resample tracking data from ~30 FPS to 25 FPS"""
        if tracking_df.empty:
            return tracking_df
        
        print(f" Resampling from {self.original_fps:.2f} FPS to {self.target_fps} FPS...")
        
        # Calculate resampling ratio
        ratio = self.original_fps / self.target_fps
        
        # Create new frame numbers at 25 FPS
        max_time = tracking_df['Time [s]'].max()
        new_times = np.arange(0, max_time, 1/self.target_fps)
        
        # Interpolate all numeric columns
        resampled_data = []
        for time in new_times:
            # Find closest original frame
            closest_idx = (tracking_df['Time [s]'] - time).abs().idxmin()
            resampled_data.append(tracking_df.loc[closest_idx].copy())
        
        resampled_df = pd.DataFrame(resampled_data)
        
        # Update frame numbers and times to be sequential at 25 FPS
        resampled_df.reset_index(drop=True, inplace=True)
        resampled_df['Frame'] = range(1, len(resampled_df) + 1)
        resampled_df['Time [s]'] = (resampled_df['Frame'] - 1) / self.target_fps
        
        print(f" Resampled: {len(tracking_df)}  {len(resampled_df)} frames")
        return resampled_df

    def process_tracking_frame_from_jsonl(self, frame_data):
        """Process a single tracking frame from JSONL data"""
        try:
            processed_frame = {
                'Period': frame_data.get('period', 1),
                'Frame': frame_data.get('frameNum', 0),
                'Time [s]': frame_data.get('periodElapsedTime', 0.0),
                'game_event_id': frame_data.get('game_event_id', None)
            }
            
            # Process home players
            home_players = frame_data.get('homePlayers', [])
            for i, player in enumerate(home_players):
                player_id = player.get('playerId', i)
                visibility = player.get('visibility', 'UNKNOWN')
                
                # Store coordinates
                processed_frame[f'Home_{player_id}_x'] = player.get('x', 0.0)
                processed_frame[f'Home_{player_id}_y'] = player.get('y', 0.0)
                processed_frame[f'Home_{player_id}_speed'] = player.get('speed', 0.0)
                
                # Store visibility - VISIBLE is high quality, ESTIMATED is low quality
                processed_frame[f'Home_{player_id}_visibility'] = visibility
            
            # Process away players
            away_players = frame_data.get('awayPlayers', [])
            for i, player in enumerate(away_players):
                player_id = player.get('playerId', i)
                visibility = player.get('visibility', 'UNKNOWN')
                
                # Store coordinates
                processed_frame[f'Away_{player_id}_x'] = player.get('x', 0.0)
                processed_frame[f'Away_{player_id}_y'] = player.get('y', 0.0)
                processed_frame[f'Away_{player_id}_speed'] = player.get('speed', 0.0)
                
                # Store visibility
                processed_frame[f'Away_{player_id}_visibility'] = visibility
            
            # Process ball
            ball_data = frame_data.get('balls', [])
            if ball_data and len(ball_data) > 0:
                ball = ball_data[0]  # Take first ball entry
                processed_frame['ball_x'] = ball.get('x', 0.0)
                processed_frame['ball_y'] = ball.get('y', 0.0)
                processed_frame['ball_z'] = ball.get('z', 0.0)
            else:
                processed_frame['ball_x'] = 0.0
                processed_frame['ball_y'] = 0.0
                processed_frame['ball_z'] = 0.0
            
            return processed_frame
            
        except Exception as e:
            print(f"   Warning: Error processing frame: {e}")
            return None

    def process_tracking_frame_from_json(self, event, frame_id):
        """Process a single tracking frame from JSON data (legacy method)"""
        try:
            frame_data = {
                'Period': event.get('gameEvents', {}).get('period', 1),
                'Frame': frame_id + 1,
                'Time [s]': event.get('eventTime', 0.0)
            }
            
            # Process home players
            home_players = event.get('homePlayers', [])
            for i, player in enumerate(home_players):
                player_id = player.get('playerId', i)
                frame_data[f'Home_{player_id}_x'] = player.get('x', 0.0)
                frame_data[f'Home_{player_id}_y'] = player.get('y', 0.0)
                frame_data[f'Home_{player_id}_speed'] = player.get('speed', 0.0)
            
            # Process away players
            away_players = event.get('awayPlayers', [])
            for i, player in enumerate(away_players):
                player_id = player.get('playerId', i)
                frame_data[f'Away_{player_id}_x'] = player.get('x', 0.0)
                frame_data[f'Away_{player_id}_y'] = player.get('y', 0.0)
                frame_data[f'Away_{player_id}_speed'] = player.get('speed', 0.0)
            
            # Process ball
            ball_data = event.get('ball', [])
            if ball_data and len(ball_data) > 0:
                ball = ball_data[0]  # Take first ball entry
                frame_data['ball_x'] = ball.get('x', 0.0)
                frame_data['ball_y'] = ball.get('y', 0.0)
                frame_data['ball_z'] = ball.get('z', 0.0)
            else:
                frame_data['ball_x'] = 0.0
                frame_data['ball_y'] = 0.0
                frame_data['ball_z'] = 0.0
            
            return frame_data
            
        except Exception as e:
            print(f"   Warning: Error processing frame {frame_id}: {e}")
            return None

    def convert_tracking_data(self, max_frames=None):
        """Convert PFF tracking data to Metrica format"""
        print(" Converting tracking data...")
        
        # Load tracking data from the separate JSONL file
        tracking_file = os.path.join(self.pff_data_dir, 'Tracking Data', f'{self.game_id}.jsonl')
        
        frames_data = []
        frame_count = 0
        
        print(" Processing tracking data from JSONL...")
        
        # Process each tracking frame
        with open(tracking_file, 'r', encoding='utf-8') as f:
            for line in f:
                if max_frames and frame_count >= max_frames:
                    break
                
                try:
                    frame_data = json.loads(line.strip())
                    
                    # Only process tracking data from first 2 periods (exclude extra time)
                    period = frame_data.get('period', 1)
                    if period > 2:
                        continue
                    
                    processed_frame = self.process_tracking_frame_from_jsonl(frame_data)
                    
                    if processed_frame:
                        frames_data.append(processed_frame)
                        frame_count += 1
                        
                        if frame_count % 1000 == 0:
                            print(f"   Processed {frame_count} frames...")
                            
                except json.JSONDecodeError:
                    continue
        
        print(f" Processed {len(frames_data)} tracking frames")
        
        if not frames_data:
            print(" No tracking data could be processed")
            return None, None
        
        # Convert to DataFrame
        tracking_df = pd.DataFrame(frames_data)
        
        # Resample to 25 FPS
        tracking_df = self.resample_tracking_data(tracking_df)
        
        # Split into home and away team data
        home_columns = ['Period', 'Frame', 'Time [s]'] + [col for col in tracking_df.columns if col.startswith('Home_')] + ['ball_x', 'ball_y']
        away_columns = ['Period', 'Frame', 'Time [s]'] + [col for col in tracking_df.columns if col.startswith('Away_')] + ['ball_x', 'ball_y']
        
        tracking_home = tracking_df[home_columns].copy()
        tracking_away = tracking_df[away_columns].copy()
        
        print(f" Created home team tracking: {len(tracking_home)} frames")
        print(f" Created away team tracking: {len(tracking_away)} frames")
        
        return tracking_home, tracking_away

    def convert_tracking_data_smart_sampling(self):
        """Convert PFF tracking data with intelligent sampling for full match coverage"""
        print(" Converting tracking data with SMART SAMPLING...")
        print(" Target: Full match coverage at ~7.5 FPS (efficient and effective)")
        print(" Using RAW COORDINATES ONLY: All players and ball use original tracking data")
        
        tracking_file = os.path.join(self.pff_data_dir, 'Tracking Data', f'{self.game_id}.jsonl')
        
        frames_data = []
        frame_count = 0
        
        # Smart sampling: Process every 4th frame to get ~7.5 FPS (avoids interpolation issues)
        sampling_rate = 4
        
        print(" Reading JSONL tracking data with smart sampling...")
        with open(tracking_file, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    frame_data = json.loads(line.strip())
                    
                    # Always include frames that have game_event_id (event frames)
                    has_event_id = frame_data.get('game_event_id') is not None
                    
                    # For other frames, use sampling rate
                    should_include = has_event_id or (line_num % sampling_rate == 0)
                    
                    if should_include:
                        processed_frame = self.process_tracking_frame(frame_data)
                        
                        if processed_frame:
                            frames_data.append(processed_frame)
                            frame_count += 1
                            
                            if frame_count % 5000 == 0:
                                print(f"   Processed {frame_count} frames (sampling every {sampling_rate} frames + event frames)...")
                            
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"   Warning: Error processing frame {frame_count}: {e}")
                    continue
        
        print(f" Smart sampling processed {len(frames_data)} frames from full match")
        
        if not frames_data:
            print(" No tracking data could be processed")
            return None, None
        
        # Convert to DataFrame
        tracking_df = pd.DataFrame(frames_data)
        
        # Set Frame as index to match how Metrica_IO loads CSV files
        if 'Frame' in tracking_df.columns:
            tracking_df = tracking_df.set_index('Frame')
            print(f" Set Frame column as index (range: {tracking_df.index.min()} to {tracking_df.index.max()})")
        
        # Skip resampling - use data as-is to avoid interpolation artifacts
        print(f" Using smart sampled data (~{29.97/sampling_rate:.1f} FPS) to avoid interpolation artifacts")
        
        # Split into home and away team data (Frame is now the index)
        home_columns = ['Period', 'Time [s]', 'game_event_id'] + [col for col in tracking_df.columns if col.startswith('Home_')] + ['ball_x', 'ball_y']
        away_columns = ['Period', 'Time [s]', 'game_event_id'] + [col for col in tracking_df.columns if col.startswith('Away_')] + ['ball_x', 'ball_y']
        
        tracking_home = tracking_df[home_columns].copy()
        tracking_away = tracking_df[away_columns].copy()
        
        print(f" Smart sampling created home team tracking: {len(tracking_home)} frames")
        print(f" Smart sampling created away team tracking: {len(tracking_away)} frames")
        
        return tracking_home, tracking_away

    def create_metrica_tracking_format(self, tracking_df, team_name):
        """Create Metrica-style tracking data with 3-row header"""
        if tracking_df is None:
            return None
        
        # Extract player jersey numbers (keep all players)
        player_columns = [col for col in tracking_df.columns if col.startswith(f'{team_name}_') and col.endswith('_x')]
        jersey_numbers = [col.split('_')[1] for col in player_columns]
        
        print(f" {team_name} team: {len(jersey_numbers)} players ({jersey_numbers})")
        
        # Create the 3-row header structure
        # Row 1: Team names
        row1 = ['', '', '']  # Period, Frame, Time
        for jersey in jersey_numbers:
            row1.extend([team_name, '', ''])  # Team name for x, y, and visibility
        row1.extend(['', ''])  # Ball x and y
        
        # Row 2: Jersey numbers  
        row2 = ['', '', '']  # Period, Frame, Time
        for jersey in jersey_numbers:
            row2.extend([jersey, '', ''])  # Jersey number for x, y, and visibility
        row2.extend(['', ''])  # Ball x and y
        
        # Row 3: Column headers
        row3 = ['Period', 'Frame', 'Time [s]']
        for jersey in jersey_numbers:
            row3.extend([f'Player{jersey}', '', 'visibility'])  # PlayerX for x, y, and visibility
        row3.extend(['Ball', ''])  # Ball for x and y
        
        # Prepare data rows
        data_rows = []
        for frame_idx, row in tracking_df.iterrows():
            data_row = [row['Period'], frame_idx, row['Time [s]']]
            
            # Add player positions (x, y, visibility alternating)
            for jersey in jersey_numbers:
                x_col = f'{team_name}_{jersey}_x'
                y_col = f'{team_name}_{jersey}_y'
                vis_col = f'{team_name}_{jersey}_visibility'
                data_row.extend([
                    row.get(x_col, np.nan), 
                    row.get(y_col, np.nan),
                    row.get(vis_col, 'UNKNOWN')
                ])
            
            # Add ball position
            data_row.extend([row.get('ball_x', np.nan), row.get('ball_y', np.nan)])
            data_rows.append(data_row)
        
        return [row1, row2, row3] + data_rows

    def save_metrica_format(self, events_df, tracking_home, tracking_away):
        """Save data in Metrica format structure"""
        print(" Saving Metrica format files...")
        
        # Create Sample_Game directory structure
        sample_dir = os.path.join(self.output_dir, f'Sample_Game_{self.game_id}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save event data
        events_file = os.path.join(sample_dir, f'Sample_Game_{self.game_id}_RawEventsData.csv')
        events_df.to_csv(events_file, index=False)
        print(f" Saved: {events_file}")
        
        # Save tracking data with proper Metrica format
        if tracking_home is not None:
            home_file = os.path.join(sample_dir, f'Sample_Game_{self.game_id}_RawTrackingData_Home_Team.csv')
            home_metrica_format = self.create_metrica_tracking_format(tracking_home, 'Home')
            
            # Write with proper CSV format (no headers, manual row writing)
            with open(home_file, 'w', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f)
                for row in home_metrica_format:
                    writer.writerow(row)
            print(f" Saved: {home_file}")
        
        if tracking_away is not None:
            away_file = os.path.join(sample_dir, f'Sample_Game_{self.game_id}_RawTrackingData_Away_Team.csv')
            away_metrica_format = self.create_metrica_tracking_format(tracking_away, 'Away')
            
            # Write with proper CSV format (no headers, manual row writing)
            with open(away_file, 'w', newline='', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f)
                for row in away_metrica_format:
                    writer.writerow(row)
            print(f" Saved: {away_file}")
        
        print(" Metrica format conversion complete!")

    def run_conversion(self):
        """Run complete PFF to Metrica conversion pipeline"""
        print(" Starting PFF to Metrica conversion...")
        print("=" * 60)
        
        try:
            # Convert event data (without frame mapping)
            events_df = self.convert_events_data()
            
            # Convert tracking data with smart sampling
            tracking_home, tracking_away = self.convert_tracking_data_smart_sampling()
            
            # Map events to tracking frames using DIRECT CALCULATION (FIXED)
            if tracking_home is not None and len(tracking_home) > 0:
                events_df = self.map_events_to_tracking_frames_direct(events_df, tracking_home)
            else:
                print("  Warning: No tracking data available for frame mapping")
            
            # Save in Metrica format
            self.save_metrica_format(events_df, tracking_home, tracking_away)
            
            print("=" * 60)
            print(" SUCCESS: PFF data converted to Metrica format!")
            print(f" Events: {len(events_df)} converted")
            print(f" Tracking: {len(tracking_home) if tracking_home is not None else 0} frames per team")
            print(f" Output location: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f" ERROR during conversion: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run the conversion"""
    print(" FIFA World Cup Final 2022 - PFF to Metrica Converter")
    print("=" * 60)
    
    # Configuration
    pff_data_dir = "PFF Data"
    output_dir = "Sample Data"
    game_id = "10517"
    
    # Smart sampling approach: Process full match efficiently
    # No frame limit needed - intelligent sampling handles efficiency
    
    # Create adapter and run conversion
    adapter = PFFToMetricaAdapter(pff_data_dir, output_dir, game_id)
    
    success = adapter.run_conversion()
    
    if success:
        print("\n PFF to Metrica conversion completed successfully!")
    else:
        print("\n Conversion failed. Check error messages above.")

if __name__ == "__main__":
    main()
