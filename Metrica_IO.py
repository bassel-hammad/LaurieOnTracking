#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:18:49 2020

Module for reading in Metrica sample data.

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)
"""

import pandas as pd
import csv as csv
import numpy as np

def read_match_data(DATADIR,gameid):
    '''
    read_match_data(DATADIR,gameid):
    read all Metrica match data (tracking data for home & away teams, and ecvent data)
    '''
    tracking_home = tracking_data(DATADIR,gameid,'Home')
    tracking_away = tracking_data(DATADIR,gameid,'Away')
    events = read_event_data(DATADIR,gameid)
    return tracking_home,tracking_away,events

def read_event_data(DATADIR,game_id):
    '''
    read_event_data(DATADIR,game_id):
    read Metrica event data  for game_id and return as a DataFrame
    '''
    eventfile = '/Sample_Game_%d/Sample_Game_%d_RawEventsData.csv' % (game_id,game_id) # filename
    events = pd.read_csv('{}/{}'.format(DATADIR, eventfile)) # read data
    return events

def tracking_data(DATADIR,game_id,teamname):
    '''
    tracking_data(DATADIR,game_id,teamname):
    read Metrica tracking data for game_id and return as a DataFrame. 
    teamname is the name of the team in the filename. For the sample data this is either 'Home' or 'Away'.
    Enhanced with flexible player count support for custom datasets.
    Now also reads PFF speed data if available.
    '''
    teamfile = '/Sample_Game_%d/Sample_Game_%d_RawTrackingData_%s_Team.csv' % (game_id,game_id,teamname)
    # First:  deal with file headers so that we can get the player names correct
    csvfile =  open('{}/{}'.format(DATADIR, teamfile), 'r') # create a csv file reader
    reader = csv.reader(csvfile) 
    
    # Row 0: Team names (extract team name from 4th column if available)
    row0 = next(reader)
    teamnamefull = row0[3].lower() if len(row0) > 3 else teamname.lower()
    print("Reading team: %s" % teamnamefull)
    
    # Row 1: Jersey numbers
    jerseys = [x for x in next(reader) if x != ''] # extract player jersey numbers from second row
    
    # Row 2: Column headers template - check if pff_speed is included
    row2 = next(reader)
    has_pff_speed = any('pff_speed' in str(col) for col in row2)
    
    # Build dynamic column names based on actual number of players
    columns = ['Period', 'Frame', 'Time [s]']  # First 3 columns
    
    # Add player columns - check for pff_speed (x, y, visibility, [pff_speed] per player)
    for jersey in jerseys:
        if has_pff_speed:
            columns.extend([f"{teamname}_{jersey}_x", f"{teamname}_{jersey}_y", 
                          f"{teamname}_{jersey}_visibility", f"{teamname}_{jersey}_pff_speed"])
        else:
            columns.extend([f"{teamname}_{jersey}_x", f"{teamname}_{jersey}_y", f"{teamname}_{jersey}_visibility"])
    
    # Add ball columns
    columns.extend(["ball_x", "ball_y"])
    
    csvfile.close()
    
    # Second: read in tracking data and place into pandas Dataframe
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
        tracking = pd.read_csv('{}/{}'.format(DATADIR, teamfile), names=columns, skiprows=3, low_memory=False)
    
    # Set Frame as index if it exists and is not already the index
    if 'Frame' in tracking.columns:
        tracking = tracking.set_index('Frame')
    
    return tracking

def merge_tracking_data(home,away):
    '''
    merge home & away tracking data files into single data frame
    '''
    return home.drop(columns=['ball_x', 'ball_y']).merge( away, left_index=True, right_index=True )
    
def to_metric_coordinates(data,field_dimen=(106.,68.) ):
    '''
    Convert positions from Metrica units to meters (with origin at centre circle)
    '''
    x_columns = [c for c in data.columns if c[-1].lower()=='x']
    y_columns = [c for c in data.columns if c[-1].lower()=='y' and 'visibility' not in c.lower()]
    data[x_columns] = ( data[x_columns]-0.5 ) * field_dimen[0]
    data[y_columns] = -1 * ( data[y_columns]-0.5 ) * field_dimen[1]
    ''' 
    ------------ ***NOTE*** ------------
    Metrica actually define the origin at the *top*-left of the field, not the bottom-left, as discussed in the YouTube video. 
    I've changed the line above to reflect this. It was originally:
    data[y_columns] = ( data[y_columns]-0.5 ) * field_dimen[1]
    ------------ ********** ------------
    '''
    return data

def to_single_playing_direction(home,away,events):
    '''
    Flip coordinates so that each team always shoots in the same direction through the match.
    This ensures consistent analysis regardless of which half the action occurs in.
    
    For World Cup Final data:
    - Period 1: First half (no flip)
    - Period 2: Second half (flip)
    - Period 3: First half of extra time (no flip - same as period 1)
    - Period 4: Second half of extra time (flip - same as period 2)
    '''
    for team in [home,away,events]:
        columns = [c for c in team.columns if c[-1].lower() in ['x','y']]
        
        # Flip coordinates for period 2 (second half) - flip only period 2
        if 2 in team['Period'].values:
            period2_mask = team['Period'] == 2
            team.loc[period2_mask, columns] *= -1
            
        # No flip for period 3 (first half of extra time) - same as period 1
        
        # Flip coordinates for period 4 (second half of extra time) - same as period 2
        if 4 in team['Period'].values:
            period4_mask = team['Period'] == 4
            team.loc[period4_mask, columns] *= -1
            
    return home,away,events

def load_player_mapping(DATADIR, game_id):
    '''
    Load player name mapping from JSON file (generated by PFF adapter)
    
    Parameters
    -----------
    DATADIR : str
        Path to directory containing Sample Data
    game_id : int or str
        Game identifier
        
    Returns
    -----------
    dict : {
        'game_id': str,
        'teams': {
            'Home': {'id': str, 'name': str, 'shortName': str},
            'Away': {'id': str, 'name': str, 'shortName': str}
        },
        'players': {
            'Home': {jersey: name, ...},
            'Away': {jersey: name, ...}
        }
    }
    
    Returns None if mapping file not found
    '''
    import json
    
    mapping_file = f'{DATADIR}/Sample_Game_{game_id}/Sample_Game_{game_id}_PlayerMapping.json'
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Player mapping file not found: {mapping_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error reading player mapping file: {e}")
        return None

def get_player_name(mapping, team, jersey_number):
    '''
    Get player name from jersey number using the mapping
    
    Parameters
    -----------
    mapping : dict
        Player mapping dictionary from load_player_mapping()
    team : str
        'Home' or 'Away'
    jersey_number : str or int
        Jersey number of the player
        
    Returns
    -----------
    str : Player name, or 'Player {jersey}' if not found
    '''
    if mapping is None:
        return f"Player {jersey_number}"
    
    players = mapping.get('players', {}).get(team, {})
    name = players.get(str(jersey_number))
    
    if name:
        return name
    return f"Player {jersey_number}"

def get_team_name(mapping, team):
    '''
    Get full team name from mapping
    
    Parameters
    -----------
    mapping : dict
        Player mapping dictionary from load_player_mapping()
    team : str
        'Home' or 'Away'
        
    Returns
    -----------
    str : Team name (e.g., 'Argentina', 'France')
    '''
    if mapping is None:
        return team
    
    return mapping.get('teams', {}).get(team, {}).get('name', team)

def get_all_players(mapping, team):
    '''
    Get all players for a team as a dictionary of jersey -> name
    
    Parameters
    -----------
    mapping : dict
        Player mapping dictionary from load_player_mapping()
    team : str
        'Home' or 'Away'
        
    Returns
    -----------
    dict : {jersey: name, ...}
    '''
    if mapping is None:
        return {}
    
    return mapping.get('players', {}).get(team, {})

def find_playing_direction(team,teamname):
    '''
    Find the direction of play for the team (based on where the goalkeepers are at kickoff). +1 is left->right and -1 is right->left
    '''    
    GK_column_x = teamname+"_"+find_goalkeeper(team)+"_x"
    # +ve is left->right, -ve is right->left
    return -np.sign(team.iloc[0][GK_column_x])
    
def find_goalkeeper(team):
    '''
    Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
    ''' 
    x_columns = [c for c in team.columns if c[-2:].lower()=='_x' and c[:4] in ['Home','Away']]
    GK_col = team.iloc[0][x_columns].abs().idxmax()
    return GK_col.split('_')[1]
    