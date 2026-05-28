#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Player Influence vs EPV Correlation Analysis

This script loads player influence analysis and EPV analysis results,
merges them by player, and generates correlation plots with statistics
to understand the relationship between total additive influence and 
various EPV metrics.

Outputs:
    - 3 scatter plots (Influence vs EPV metrics)
    - Correlation statistics and regression analysis
    - Combined visualization with trend lines

Usage:
    python plot_influence_epv_correlation.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100


def load_player_influence_data(file_path):
    """
    Load player influence data from Excel file.
    
    Parameters
    ----------
    file_path : str
        Path to the player influence Excel file
    
    Returns
    -------
    dict
        Dictionary with keys like 'Home_Period_1', 'Home_Period_2', etc.
        Each value is a DataFrame with Player, Team, Period, and Total Additive Influence
    """
    print(f"Loading player influence data from: {file_path}")
    
    # Read all available sheets to determine structure
    xl = pd.ExcelFile(file_path)
    available_sheets = xl.sheet_names
    print(f"  Available sheets: {available_sheets}")
    
    period_data = {}
    
    # Check if new format (separate period sheets) or old format (single Total Additive sheet)
    if 'Total Additive' in available_sheets:
        # Old format: single sheet - treat as full game
        df = pd.read_excel(file_path, sheet_name='Total Additive')
        df['Period'] = 'Full Game'
        period_data['Full_Game'] = df
    else:
        # New format: separate sheets for each period
        for sheet in available_sheets:
            if 'Total' in sheet:
                temp_df = pd.read_excel(file_path, sheet_name=sheet)
                
                # Rename column from 'Total Influence' to 'Total Additive Influence'
                if 'Total Influence' in temp_df.columns:
                    temp_df.rename(columns={'Total Influence': 'Total Additive Influence'}, inplace=True)
                
                # Add Team and Period columns
                if 'Home' in sheet:
                    temp_df['Team'] = 'Home'
                    if 'Period 1' in sheet:
                        temp_df['Period'] = 'Period 1'
                        period_data['Home_Period_1'] = temp_df
                    elif 'Period 2' in sheet:
                        temp_df['Period'] = 'Period 2'
                        period_data['Home_Period_2'] = temp_df
                elif 'Away' in sheet:
                    temp_df['Team'] = 'Away'
                    if 'Period 1' in sheet:
                        temp_df['Period'] = 'Period 1'
                        period_data['Away_Period_1'] = temp_df
                    elif 'Period 2' in sheet:
                        temp_df['Period'] = 'Period 2'
                        period_data['Away_Period_2'] = temp_df
    
    print(f"  Loaded {len(period_data)} period datasets")
    for key, df in period_data.items():
        print(f"    {key}: {len(df)} players")
    
    return period_data


def load_team_names_from_metadata(match_id):
    """
    Load team names from PFF metadata JSON file.
    
    Parameters
    ----------
    match_id : str
        Match ID
    
    Returns
    -------
    dict
        Dictionary mapping 'Home' and 'Away' to actual team names
    """
    import json
    
    metadata_path = os.path.join('PFF Data', 'Meta Data', f'{match_id}.json')
    team_names = {'Home': 'Home Team', 'Away': 'Away Team'}
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Metadata is a list with one item
            if isinstance(metadata, list) and len(metadata) > 0:
                match_data = metadata[0]
            else:
                match_data = metadata
            
            # Extract team names
            if 'homeTeam' in match_data and 'name' in match_data['homeTeam']:
                team_names['Home'] = match_data['homeTeam']['name']
            
            if 'awayTeam' in match_data and 'name' in match_data['awayTeam']:
                team_names['Away'] = match_data['awayTeam']['name']
            
            print(f"  Loaded team names from metadata: {team_names}")
            
        except Exception as e:
            print(f"  Warning: Could not load team names from metadata: {e}")
            print(f"  Using default team names")
    else:
        print(f"  Metadata file not found: {metadata_path}")
        print(f"  Using default team names")
    
    return team_names


def load_epv_data(file_path):
    """
    Load EPV analysis data from Excel file.
    
    Parameters
    ----------
    file_path : str
        Path to the EPV analysis Excel file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with player EPV metrics
    """
    print(f"Loading EPV data from: {file_path}")
    
    # Read the "Total EPV by Player" sheet
    df = pd.read_excel(file_path, sheet_name='Total EPV by Player')
    
    print(f"  Loaded {len(df)} players")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def load_player_distances(file_path):
    """
    Load player distances data from Excel file.
    
    Parameters
    ----------
    file_path : str
        Path to the player distances Excel file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with player distance metrics
    """
    print(f"Loading player distances from: {file_path}")
    
    # Read both sheets (Home Team and Away Team)
    try:
        home_df = pd.read_excel(file_path, sheet_name='Home Team')
        away_df = pd.read_excel(file_path, sheet_name='Away Team')
        
        print(f"  Loaded Home Team: {len(home_df)} players")
        print(f"  Loaded Away Team: {len(away_df)} players")
        
        # Combine both teams
        df = pd.concat([home_df, away_df], ignore_index=True)
        print(f"  Total: {len(df)} players")
        
    except Exception as e:
        print(f"  Warning: Could not read Home Team/Away Team sheets: {e}")
        # Try alternative sheet names
        try:
            home_df = pd.read_excel(file_path, sheet_name='Home')
            away_df = pd.read_excel(file_path, sheet_name='Away')
            df = pd.concat([home_df, away_df], ignore_index=True)
            print(f"  Loaded from Home/Away sheets: {len(df)} players")
        except:
            print(f"  Trying default sheet...")
            df = pd.read_excel(file_path)
            print(f"  Loaded {len(df)} players from default sheet")
    
    print(f"  Columns: {list(df.columns)}")
    
    # Convert Distance from km to meters
    if 'Distance [km]' in df.columns:
        df['Total Distance (m)'] = df['Distance [km]'] * 1000
        print(f"  Converted distance from km to meters")
    
    # Keep only necessary columns
    if 'Player ID' in df.columns:
        df = df[['Player ID', 'Minutes Played', 'Total Distance (m)']].copy()
        print(f"  Using Player ID column for merging")
    
    return df


def merge_datasets(influence_df, epv_df, distances_df=None, period_label=''):
    """
    Merge influence, EPV, and distance datasets by player name.
    
    Parameters
    ----------
    influence_df : pd.DataFrame
        Player influence data for a specific period
    epv_df : pd.DataFrame
        EPV analysis data
    distances_df : pd.DataFrame, optional
        Player distances data
    period_label : str
        Label for the period (e.g., 'Home_Period_1')
    
    Returns
    -------
    pd.DataFrame
        Merged dataset with influence, EPV, and distance metrics
    """
    print("\nMerging datasets...")
    
    # Merge on Player name (influence_df has 'Player', epv_df has 'Player Name')
    merged = pd.merge(
        influence_df,
        epv_df,
        left_on='Player',
        right_on='Player Name',
        how='inner',
        suffixes=('_influence', '_epv')
    )
    
    print(f"  Merged influence + EPV: {len(merged)} players")
    
    # Debug: Show team breakdown
    if 'Team_influence' in merged.columns:
        home_count_initial = (merged['Team_influence'] == 'Home').sum()
        away_count_initial = (merged['Team_influence'] == 'Away').sum()
        print(f"    Home: {home_count_initial}, Away: {away_count_initial}")
    
    # Merge with distances if provided
    if distances_df is not None:
        print(f"  Distance data columns: {list(distances_df.columns)}")
        
        # Debug: Show what's in distance data
        if 'Player ID' in distances_df.columns:
            home_dist_count = distances_df['Player ID'].str.startswith('Home').sum()
            away_dist_count = distances_df['Player ID'].str.startswith('Away').sum()
            print(f"  Distance data breakdown - Home: {home_dist_count}, Away: {away_dist_count}")
        
        # Merge by Player ID (should be Team_Jersey format like "Home_10")
        merged = pd.merge(
            merged,
            distances_df,
            on='Player ID',
            how='left',
            suffixes=('', '_dist')
        )
        
        print(f"  Merged with distances: {len(merged)} players")
        print(f"  Players with distance data: {merged['Total Distance (m)'].notna().sum()}")
    
    # Keep relevant columns
    columns_to_keep = [
        'Player',
        'Player ID',  # Keep Player ID for jersey numbers
        'Team_influence',
        'Total Additive Influence',
        'Avg EPV per Pass',
        'Avg EPV per Reception',
        'EPV Combined per Touch'
    ]
    
    # Add distance column if it exists
    if 'Total Distance (m)' in merged.columns:
        columns_to_keep.append('Total Distance (m)')
    
    merged = merged[columns_to_keep].copy()
    merged.rename(columns={'Team_influence': 'Team'}, inplace=True)
    
    # Remove players with NaN EPV values only (keep all players with valid EPV)
    print(f"\nFiltering players with valid data...")
    original_count = len(merged)
    
    # Only remove NaN values for EPV metrics (keep players even without influence or distance data)
    merged = merged[
        (merged['Avg EPV per Pass'].notna()) &
        (merged['Avg EPV per Reception'].notna()) &
        (merged['EPV Combined per Touch'].notna())
    ].copy()
    
    print(f"  Kept {len(merged)} players (removed {original_count - len(merged)} with NaN EPV values)")
    
    # Show counts by team - DEBUG
    if 'Team' in merged.columns:
        print(f"\n  Team breakdown after filtering:")
        home_count = (merged['Team'] == 'Home').sum()
        away_count = (merged['Team'] == 'Away').sum()
        print(f"    Home team: {home_count} players")
        print(f"    Away team: {away_count} players")
        
        # Show player IDs for debugging
        print(f"\n  Sample Player IDs:")
        print(f"    Home: {merged[merged['Team'] == 'Home']['Player ID'].head(5).tolist()}")
        print(f"    Away: {merged[merged['Team'] == 'Away']['Player ID'].head(5).tolist()}")
    else:
        home_count = 0
        away_count = 0
    
    if 'Total Additive Influence' in merged.columns:
        influence_count = merged['Total Additive Influence'].notna().sum()
        print(f"  Players with influence data: {influence_count}")
    
    if 'Total Distance (m)' in merged.columns:
        distance_count = merged['Total Distance (m)'].notna().sum()
        print(f"  Players with distance data: {distance_count}")
        
        # Show which team has distance data
        home_dist = merged[(merged['Team'] == 'Home') & (merged['Total Distance (m)'].notna())]
        away_dist = merged[(merged['Team'] == 'Away') & (merged['Total Distance (m)'].notna())]
        print(f"    Home team with distance: {len(home_dist)} players")
        print(f"    Away team with distance: {len(away_dist)} players")
    
    print(f"  Note: Players without influence/distance data will show as (0 or NaN) in those metrics")
    
    return merged


def calculate_correlation_stats(x, y, metric_name):
    """
    Calculate correlation statistics for x and y.
    
    Parameters
    ----------
    x : array-like
        Independent variable (Total Additive Influence)
    y : array-like
        Dependent variable (EPV metric)
    metric_name : str
        Name of the EPV metric
    
    Returns
    -------
    dict
        Dictionary with correlation statistics
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Check if we have enough data points
    if len(x_clean) < 2:
        print(f"  WARNING: Not enough valid data points ({len(x_clean)}) for {metric_name}")
        return {
            'metric': metric_name,
            'pearson_r': 0.0,
            'pearson_p': 1.0,
            'spearman_r': 0.0,
            'spearman_p': 1.0,
            'slope': 0.0,
            'intercept': 0.0,
            'r_squared': 0.0,
            'p_value': 1.0,
            'std_err': 0.0
        }
    
    # Pearson correlation (linear relationship)
    pearson_r, pearson_p = pearsonr(x_clean, y_clean)
    
    # Spearman correlation (monotonic relationship)
    spearman_r, spearman_p = spearmanr(x_clean, y_clean)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    # R-squared
    r_squared = r_value ** 2
    
    stats_dict = {
        'metric': metric_name,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err
    }
    
    return stats_dict


def plot_correlation(df, x_col, y_col, title, output_path, stats_dict):
    """
    Create a scatter plot with trend line and statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to plot
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    title : str
        Plot title
    output_path : str
        Path to save the plot
    stats_dict : dict
        Dictionary with correlation statistics
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by team
    home_data = df[df['Team'] == 'Home']
    away_data = df[df['Team'] == 'Away']
    
    # Scatter plots
    ax.scatter(home_data[x_col], home_data[y_col], 
              c='#E74C3C', alpha=0.6, s=200, edgecolors='black', linewidth=0.5,
              label='Home', zorder=3)
    
    ax.scatter(away_data[x_col], away_data[y_col], 
              c='#3498DB', alpha=0.6, s=200, edgecolors='black', linewidth=0.5,
              label='Away', zorder=3)
    
    # Add jersey numbers as annotations
    for idx, row in home_data.iterrows():
        # Extract jersey number from Player ID column (format: "Home_#")
        player_id = row['Player ID']
        if '_' in player_id:
            jersey = player_id.split('_')[-1]
        else:
            jersey = '?'
        
        ax.annotate(jersey, (row[x_col], row[y_col]), 
                   fontsize=9, fontweight='bold', ha='center', va='center',
                   color='black', zorder=4)
    
    for idx, row in away_data.iterrows():
        # Extract jersey number from Player ID column (format: "Away_#")
        player_id = row['Player ID']
        if '_' in player_id:
            jersey = player_id.split('_')[-1]
        else:
            jersey = '?'
        
        ax.annotate(jersey, (row[x_col], row[y_col]), 
                   fontsize=9, fontweight='bold', ha='center', va='center',
                   color='black', zorder=4)
    
    # Regression line
    x = df[x_col].values
    y = df[y_col].values
    slope = stats_dict['slope']
    intercept = stats_dict['intercept']
    
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.7, 
           label=f'Trend: y = {slope:.6f}x + {intercept:.6f}', zorder=2)
    
    # Labels and title
    ax.set_xlabel(x_col, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='best', fontsize=10, frameon=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Add statistics text box
    stats_text = (
        f"Pearson r = {stats_dict['pearson_r']:.4f} (p = {stats_dict['pearson_p']:.4e})\n"
        f"Spearman ρ = {stats_dict['spearman_r']:.4f} (p = {stats_dict['spearman_p']:.4e})\n"
        f"R² = {stats_dict['r_squared']:.4f}\n"
        f"Slope = {stats_dict['slope']:.6f} ± {stats_dict['std_err']:.6f}"
    )
    
    # Interpretation
    pearson_r = abs(stats_dict['pearson_r'])
    if pearson_r > 0.7:
        strength = "Strong"
    elif pearson_r > 0.4:
        strength = "Moderate"
    elif pearson_r > 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"
    
    direction = "positive" if stats_dict['pearson_r'] > 0 else "negative"
    significance = "significant" if stats_dict['pearson_p'] < 0.05 else "not significant"
    
    interpretation = f"\n{strength} {direction} correlation\n({significance})"
    stats_text += interpretation
    
    # Position text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {output_path}")
    plt.close()


def create_combined_plot(period_data_dict, stats_dict, output_path, match_id, team_names):
    """
    Create a combined figure with influence correlation plots for all periods and teams.
    
    Parameters
    ----------
    period_data_dict : dict
        Dictionary with merged data for each period (keys: 'Home_Period_1', etc.)
    stats_dict : dict
        Dictionary with statistics for each period
    output_path : str
        Path to save the combined plot
    match_id : str
        Match ID for the title
    team_names : dict
        Dictionary mapping 'Home' and 'Away' to actual team names
    """
    fig = plt.figure(figsize=(24, 32))
    # Create a grid: title + 4 rows of 3 plots (Home P1, Home P2, Away P1, Away P2)
    gs = fig.add_gridspec(5, 3, height_ratios=[0.05, 1, 1, 1, 1], hspace=0.35, wspace=0.3)
    
    # Title and match info axes
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    # Create axes for all 4 rows
    axes_home_p1 = [fig.add_subplot(gs[1, i]) for i in range(3)]
    axes_home_p2 = [fig.add_subplot(gs[2, i]) for i in range(3)]
    axes_away_p1 = [fig.add_subplot(gs[3, i]) for i in range(3)]
    axes_away_p2 = [fig.add_subplot(gs[4, i]) for i in range(3)]
    
    y_cols = ['Avg EPV per Pass', 'Avg EPV per Reception', 'EPV Combined per Touch']
    x_col = 'Total Additive Influence'
    
    # Define all period configurations with actual team names
    home_name = team_names.get('Home', 'Home Team')
    away_name = team_names.get('Away', 'Away Team')
    
    period_configs = [
        ('Home_Period_1', axes_home_p1, f'{home_name} - Period 1', '#E74C3C'),
        ('Home_Period_2', axes_home_p2, f'{home_name} - Period 2', '#C0392B'),
        ('Away_Period_1', axes_away_p1, f'{away_name} - Period 1', '#3498DB'),
        ('Away_Period_2', axes_away_p2, f'{away_name} - Period 2', '#2874A6')
    ]
    
    # Plot all periods
    for period_key, axes, period_label, color in period_configs:
        if period_key not in period_data_dict:
            continue
            
        df = period_data_dict[period_key]
        period_stats = stats_dict.get(period_key, [])
        
        if len(period_stats) != 3:
            continue
        
        titles = [
            f'{period_label}: Influence vs EPV per Pass',
            f'{period_label}: Influence vs EPV per Reception',
            f'{period_label}: Influence vs EPV per Touch'
        ]
        
        for i, (ax, y_col, title, stats) in enumerate(zip(axes, y_cols, titles, period_stats)):
            # Scatter plot
            ax.scatter(df[x_col], df[y_col], 
                      c=color, alpha=0.7, s=200, edgecolors='black', linewidth=1,
                      label=f'{period_label} ({len(df)} players)', zorder=3)
            
            # Add jersey numbers
            for idx, row in df.iterrows():
                player_id = row.get('Player ID', '')
                if '_' in player_id:
                    jersey = player_id.split('_')[-1]
                else:
                    jersey = '?'
                
                ax.annotate(jersey, (row[x_col], row[y_col]), 
                           fontsize=10, fontweight='bold', ha='center', va='center',
                           color='black', zorder=4)
            
            # Regression line
            x = df[x_col].values
            y = df[y_col].values
            slope = stats['slope']
            intercept = stats['intercept']
            
            if len(x) > 0:
                x_line = np.array([x.min(), x.max()])
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, 'k--', linewidth=2.5, alpha=0.8, zorder=2)
            
            # Labels
            ax.set_xlabel(x_col, fontsize=14, fontweight='bold')
            ax.set_ylabel(y_col, fontsize=14, fontweight='bold')
            ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
            
            # Legend
            ax.legend(loc='best', fontsize=11, frameon=True, shadow=True, fancybox=True)
            
            # Grid
            ax.grid(True, alpha=0.3, zorder=0, linewidth=1)
            
            # Tick labels
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            # Statistics
            stats_text = (
                f"r = {stats['pearson_r']:.3f}\n"
                f"R² = {stats['r_squared']:.3f}\n"
                f"p = {stats['pearson_p']:.3e}"
            )
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=props, family='monospace', fontweight='bold')
    
    # # DISTANCE PLOTS - COMMENTED OUT
    # if has_distance and 'Total Distance (m)' in df.columns:
    #     x_col_distance = 'Total Distance (m)'
    #     titles_distance = [
    #         'Total Distance vs EPV per Pass',
    #         'Total Distance vs EPV per Reception',
    #         'Total Distance vs EPV per Touch'
    #     ]
    
    # Add title with team names
    home_name = team_names.get('Home', 'Home Team')
    away_name = team_names.get('Away', 'Away Team')
    title_text = f'Player Influence vs EPV Metrics by Period - Match {match_id}\n{home_name} vs {away_name}'
    ax_title.text(0.5, 0.5, title_text, ha='center', va='center', 
                 fontsize=20, fontweight='bold', transform=ax_title.transAxes)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved combined plot: {output_path}")
    plt.close()


def print_summary_statistics(stats_list):
    """
    Print a summary of correlation statistics for all metrics.
    
    Parameters
    ----------
    stats_list : list
        List of statistics dictionaries
    """
    print("\n" + "=" * 70)
    print("CORRELATION STATISTICS SUMMARY")
    print("=" * 70)
    
    for stats in stats_list:
        print(f"\n{stats['metric']}:")
        print(f"  Pearson correlation (r):  {stats['pearson_r']:7.4f}  (p = {stats['pearson_p']:.4e})")
        print(f"  Spearman correlation (ρ): {stats['spearman_r']:7.4f}  (p = {stats['spearman_p']:.4e})")
        print(f"  R² (coefficient of determination): {stats['r_squared']:.4f}")
        print(f"  Linear regression:")
        print(f"    Slope:     {stats['slope']:10.6f} ± {stats['std_err']:.6f}")
        print(f"    Intercept: {stats['intercept']:10.6f}")
        print(f"    p-value:   {stats['p_value']:.4e}")
        
        # Interpretation
        pearson_r = abs(stats['pearson_r'])
        if pearson_r > 0.7:
            strength = "STRONG"
        elif pearson_r > 0.4:
            strength = "MODERATE"
        elif pearson_r > 0.2:
            strength = "WEAK"
        else:
            strength = "VERY WEAK"
        
        direction = "positive" if stats['pearson_r'] > 0 else "negative"
        significance = "SIGNIFICANT" if stats['pearson_p'] < 0.05 else "NOT SIGNIFICANT"
        
        print(f"  Interpretation: {strength} {direction} correlation ({significance})")
    
    print("\n" + "=" * 70)
    print("COMPARISON OF METRICS")
    print("=" * 70)
    
    # Sort by absolute correlation
    sorted_stats = sorted(stats_list, key=lambda x: abs(x['pearson_r']), reverse=True)
    
    print("\nRanked by correlation strength (|r|):")
    for i, stats in enumerate(sorted_stats, 1):
        print(f"  {i}. {stats['metric']:<30} |r| = {abs(stats['pearson_r']):.4f}")
    
    print("\nRanked by R² (variance explained):")
    sorted_by_r2 = sorted(stats_list, key=lambda x: x['r_squared'], reverse=True)
    for i, stats in enumerate(sorted_by_r2, 1):
        print(f"  {i}. {stats['metric']:<30} R² = {stats['r_squared']:.4f} ({stats['r_squared']*100:.1f}% variance)")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    best_metric = sorted_stats[0]
    print(f"\n'{best_metric['metric']}' shows the strongest correlation")
    print(f"with Total Additive Influence (r = {best_metric['pearson_r']:.4f}).")
    print(f"\nThis suggests that {best_metric['metric']} is the most related EPV metric")
    print(f"to a player's additive influence on pitch control.")


def main():
    """Main entry point."""
    print("=" * 70)
    print("PLAYER INFLUENCE vs EPV CORRELATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Get match ID
    match_id = input("Enter match ID (e.g., 3822, 10517): ").strip()
    if not match_id:
        print("ERROR: Match ID is required!")
        sys.exit(1)
    
    print()
    
    # File paths
    influence_file = os.path.join(
        'Metrica_Output',
        'Full Match Analysis',
        f'player_influence_analysis_additive_epv_match_{match_id}.xlsx'
    )
    epv_file = os.path.join(
        'Metrica_Output',
        'Full Match Analysis',
        f'epv_analysis_match_{match_id}.xlsx'
    )
    if not os.path.exists(epv_file):
        epv_file = f'epv_analysis_match_{match_id}.xlsx'
    
    # Check if files exist
    if not os.path.exists(influence_file):
        print(f"ERROR: Player influence file not found: {influence_file}")
        print("Please run generate_full_game_influence_analysis.py first.")
        sys.exit(1)
    
    if not os.path.exists(epv_file):
        print(f"ERROR: EPV analysis file not found: {epv_file}")
        print("Please run generate_epv_analysis.py first.")
        sys.exit(1)
    
    # Load data
    influence_period_data = load_player_influence_data(influence_file)
    epv_df = load_epv_data(epv_file)
    team_names = load_team_names_from_metadata(match_id)
    
    # Try to load distance data
    distance_file = os.path.join('Metrica_Output', f'game_{match_id}_player_distances.xlsx')
    distances_df = None
    has_distance = False
    
    if os.path.exists(distance_file):
        try:
            distances_df = load_player_distances(distance_file)
            has_distance = True
        except Exception as e:
            print(f"Warning: Could not load distance data: {e}")
            print("Continuing without distance analysis...")
    else:
        print(f"\nDistance file not found: {distance_file}")
        print("Continuing with influence analysis only...")
    
    # Merge datasets for each period
    merged_period_data = {}
    for period_key, influence_df in influence_period_data.items():
        merged_df = merge_datasets(influence_df, epv_df, distances_df, period_key)
        if len(merged_df) > 0:
            merged_period_data[period_key] = merged_df
            print(f"\n{period_key}: {len(merged_df)} players merged")
    
    if len(merged_period_data) == 0:
        print("\nERROR: No players found in datasets!")
        sys.exit(1)
    
    # Calculate statistics for each period
    print("\n" + "=" * 70)
    print("CALCULATING CORRELATIONS BY PERIOD")
    print("=" * 70)
    print()
    
    y_cols = ['Avg EPV per Pass', 'Avg EPV per Reception', 'EPV Combined per Touch']
    x_col_influence = 'Total Additive Influence'
    
    all_stats = {}
    for period_key, merged_df in merged_period_data.items():
        print(f"\n{period_key.upper()} CORRELATIONS:")
        period_stats = []
        
        for y_col in y_cols:
            print(f"  Analyzing: {x_col_influence} vs {y_col}")
            stats_dict = calculate_correlation_stats(
                merged_df[x_col_influence].values,
                merged_df[y_col].values,
                f"{period_key} - {y_col}"
            )
            period_stats.append(stats_dict)
            print(f"    Pearson r = {stats_dict['pearson_r']:.4f}, R² = {stats_dict['r_squared']:.4f}")
        
        all_stats[period_key] = period_stats
    
    # # Distance correlations (COMMENTED OUT)
    # stats_list_distance = []
    # if has_distance and 'Total Distance (m)' in merged_df.columns:
    #     print("\nDISTANCE CORRELATIONS:")
    #     x_col_distance = 'Total Distance (m)'
    #     
    #     # Count how many players have distance data
    #     valid_distance_count = merged_df[x_col_distance].notna().sum()
    #     print(f"  Players with valid distance data: {valid_distance_count} out of {len(merged_df)}")
    #     
    #     if valid_distance_count < 2:
    #         print(f"  WARNING: Not enough players with distance data ({valid_distance_count}). Skipping distance correlations.")
    #         has_distance = False
    #     else:
    #         for y_col in y_cols:
    #             print(f"  Analyzing: {x_col_distance} vs {y_col}")
    #             # Filter out NaN values for distance
    #             valid_mask = merged_df[x_col_distance].notna()
    #             stats_dict = calculate_correlation_stats(
    #                 merged_df.loc[valid_mask, x_col_distance].values,
    #                 merged_df.loc[valid_mask, y_col].values,
    #                 y_col
    #             )
    #             stats_list_distance.append(stats_dict)
    #             print(f"    Pearson r = {stats_dict['pearson_r']:.4f}, R² = {stats_dict['r_squared']:.4f}")
    
    # Create output directory
    output_dir = 'Metrica_Output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate combined plot with all periods
    print("\n" + "=" * 70)
    print("GENERATING COMBINED PLOT (4 PERIODS)")
    print("=" * 70)
    print()
    combined_output = os.path.join(output_dir, f'correlation_influence_vs_epv_combined_match_{match_id}.png')
    create_combined_plot(merged_period_data, all_stats, combined_output, match_id, team_names)
    
    # Print summary for each period
    for period_key, stats_list in all_stats.items():
        print("\n" + "=" * 70)
        print(f"{period_key.upper()} CORRELATIONS SUMMARY")
        print("=" * 70)
        print_summary_statistics(stats_list)
    
    # # Export merged data to CSV (DISABLED)
    # for period_key, merged_df in merged_period_data.items():
    #     csv_output = os.path.join(output_dir, f'merged_influence_epv_{period_key}_match_{match_id}.csv')
    #     merged_df.to_csv(csv_output, index=False)
    #     print(f"\n{period_key} data saved to: {csv_output}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print()
    print(f"\nGenerated files:")
    print(f"  - 1 combined plot (12 subplots: 4 periods × 3 EPV metrics)")
    print(f"\nPlot saved to: {output_dir}/")


if __name__ == '__main__':
    main()
