#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:52:19 2020

Module for measuring player velocities, smoothed using a Savitzky-Golay filter, with Metrica tracking data.

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)

"""
import numpy as np
import pandas as pd
import scipy.signal as signal

def calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
    """ calc_player_velocities( tracking_data )
    
    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
    
    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
        
    Returrns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added

    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)
    
    # Get the player ids (only from _x columns to avoid visibility columns)
    player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in ['Home','Away'] and c.endswith('_x') ] )

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team['Time [s]'].diff()
    
    # index of first frame in second half
    second_half_idx = team[team.Period==2].index.min()
    
    # estimate velocities for players in team
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = team[player+"_x"].diff() / dt
        vy = team[player+"_y"].diff() / dt

        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>maxspeed ] = np.nan
            vy[ raw_speed>maxspeed ] = np.nan
            
        if smoothing:
            if filter_=='Savitzky-Golay':
                # calculate first half velocity
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)        
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window 
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' ) 
                vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )      
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' ) 
                vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' ) 
                
        
        # put player speed in x,y direction, and total speed back in the data frame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt( vx**2 + vy**2 )

    return team

def remove_player_velocities(team):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    # Preserve pff_speed columns as they contain raw PFF data
    columns = [c for c in team.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration'] and 'pff_speed' not in c] # Get the player ids
    team = team.drop(columns=columns)
    return team


def calc_player_velocities_hybrid(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed=12, use_pff_speed=True):
    """ calc_player_velocities_hybrid( tracking_data )
    
    Calculate player velocities using a HYBRID approach:
    - Direction (angle): Calculated from position differences (smoothed)
    - Magnitude (speed): Use PFF's raw speed values (more accurate, calculated at higher FPS)
    
    This provides more accurate velocities because PFF calculates speed internally at ~15 FPS
    while exported positions are only available at ~3.75 FPS (after deduplication).
    
    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter_: type of filter to use when smoothing the velocities. Default is Savitzky-Golay
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter
        maxspeed: the maximum speed that a player can realistically achieve (in meters/second)
        use_pff_speed: if True, use PFF's raw speed values; if False, calculate from positions
        
    Returns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added
    """
    import warnings
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)
    
    # Get the player ids (only from _x columns to avoid visibility columns)
    player_ids = np.unique([c[:-2] for c in team.columns if c[:4] in ['Home', 'Away'] and c.endswith('_x')])
    
    # Calculate the timestep from one frame to the next
    dt = team['Time [s]'].diff()
    
    # index of first frame in second half
    second_half_idx = team[team.Period == 2].index.min()
    if pd.isna(second_half_idx):
        second_half_idx = team.index.max() + 1  # If no second half, use end of data
    
    # Collect all new columns to add at once (avoids fragmentation)
    new_columns = {}
    # Check if PFF speed columns exist
    has_pff_speed = any(c.endswith('_pff_speed') for c in team.columns)
    
    for player in player_ids:
        # Calculate direction from position differences
        dx = team[player + "_x"].diff()
        dy = team[player + "_y"].diff()
        
        # Calculate raw direction (angle)
        raw_distance = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero - use small epsilon
        safe_distance = raw_distance.replace(0, np.nan)
        
        # Unit direction vector
        dir_x = dx / safe_distance
        dir_y = dy / safe_distance
        
        # Fill NaN directions with previous valid direction (forward fill)
        dir_x = dir_x.ffill().bfill()
        dir_y = dir_y.ffill().bfill()
        
        # Get speed magnitude
        if use_pff_speed and (player + "_pff_speed") in team.columns:
            # Use PFF's raw speed (more accurate, calculated at higher FPS)
            # PFF speed is in m/s, need to convert to normalized units per second
            pff_speed_col = player + "_pff_speed"
            
            # Convert PFF speed from m/s to normalized units/s
            # Field is 105m x 68m, normalized to 0-1
            # Average scale factor: (1/105 + 1/68) / 2 ≈ 0.012
            # But direction matters: use separate scaling for x and y components
            field_length = 105.0  # meters
            field_width = 68.0    # meters
            
            # PFF speed is scalar in m/s
            pff_speed = team[pff_speed_col].copy()
            
            # Apply maxspeed filter
            if maxspeed > 0:
                pff_speed[pff_speed > maxspeed] = np.nan
            
            # Smooth the speed if requested
            if smoothing and filter_ == 'Savitzky-Golay':
                # Handle NaN values for filtering
                pff_speed_filled = pff_speed.ffill().bfill().fillna(0)
                try:
                    pff_speed.loc[:second_half_idx] = signal.savgol_filter(
                        pff_speed_filled.loc[:second_half_idx], window_length=window, polyorder=polyorder)
                    pff_speed.loc[second_half_idx:] = signal.savgol_filter(
                        pff_speed_filled.loc[second_half_idx:], window_length=window, polyorder=polyorder)
                except:
                    pass  # Keep unsmoothed if filter fails
            
            # Convert speed from m/s to normalized velocity components
            # vx in normalized units = speed_m_s * dir_x / field_length
            # vy in normalized units = speed_m_s * dir_y / field_width
            vx = pff_speed * dir_x / field_length
            vy = pff_speed * dir_y / field_width
            
            # Store the speed in m/s
            speed = pff_speed
        else:
            # Fall back to original method: calculate from position differences
            vx = dx / dt
            vy = dy / dt
            
            if maxspeed > 0:
                raw_speed = np.sqrt(vx**2 + vy**2)
                vx[raw_speed > maxspeed] = np.nan
                vy[raw_speed > maxspeed] = np.nan
            
            if smoothing and filter_ == 'Savitzky-Golay':
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx], window_length=window, polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx], window_length=window, polyorder=polyorder)
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:], window_length=window, polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:], window_length=window, polyorder=polyorder)
            
            speed = np.sqrt(vx**2 + vy**2)
        
        # Collect velocity columns (add all at once later to avoid fragmentation)
        new_columns[player + "_vx"] = vx
        new_columns[player + "_vy"] = vy
        new_columns[player + "_speed"] = speed
    
    # Add all velocity columns at once to avoid DataFrame fragmentation
    velocity_df = pd.DataFrame(new_columns, index=team.index)
    team = pd.concat([team, velocity_df], axis=1)
    
    return team