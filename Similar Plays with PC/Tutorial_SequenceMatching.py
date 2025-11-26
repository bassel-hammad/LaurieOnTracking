#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial: Finding Similar Game Sequences Using Pitch Control

This tutorial demonstrates how to find similar possession sequences based on 
pitch control surface evolution. Uses SSIM (Structural Similarity Index) and 
DTW (Dynamic Time Warping) to compare sequences like comparing video clips.

Example: Find sequences similar to Di María's goal-scoring sequence

Author: Sequence Matching System
Date: November 2025
"""

import sys
sys.path.append('..')

import Metrica_IO as mio
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import sequence_similarity as ss
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

print("=" * 70)
print("TUTORIAL: FINDING SIMILAR GAME SEQUENCES")
print("Using Pitch Control + SSIM + DTW")
print("=" * 70)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================
DATADIR = '../Sample Data'
PFF_DATA_DIR = '../PFF Data'

# All available games from World Cup 2022
GAME_IDS = [10514, 10515, 10516, 10517]
GAME_NAMES = {
    10514: "Argentina vs Croatia (Semi-final)",
    10515: "France vs Morocco (Semi-final)",
    10516: "Croatia vs Morocco (3rd Place)",
    10517: "Argentina vs France (Final)"
}

MIN_EVENTS = 4      # Minimum events per sequence
MIN_DURATION = 5.0  # Minimum duration in seconds
TOP_K = 5          # Number of similar sequences to find

print("Configuration:")
print(f"  Games: {len(GAME_IDS)} World Cup 2022 matches")
for gid in GAME_IDS:
    print(f"    - {gid}: {GAME_NAMES[gid]}")
print(f"  Min sequence length: {MIN_EVENTS} events OR {MIN_DURATION}s")
print(f"  Finding top {TOP_K} similar sequences")
print()

# =============================================================================
# STEP 1: BUILD OR LOAD MULTI-GAME SEQUENCE DATABASE
# =============================================================================
print("=" * 70)
print("STEP 1: MULTI-GAME SEQUENCE DATABASE")
print("=" * 70)
print()

# Check if database already exists
db_filename = f'database_multi_game_min{MIN_EVENTS}ev_{MIN_DURATION}s.pkl'
db_path = os.path.join('.', db_filename)

if os.path.exists(db_path):
    print(f"✓ Found existing database: {db_filename}")
    print("Loading saved database...")
    with open(db_path, 'rb') as f:
        database = pickle.load(f)
    print(f"✓ Loaded {len(database)} sequences from disk")
    print()
else:
    print("No saved database found. Building new database from all games...")
    print()
    
    # Build combined database from all games
    database = []
    
    for game_id in GAME_IDS:
        print(f"Processing game {game_id}: {GAME_NAMES[game_id]}")
        print("-" * 60)
        
        # Load game data
        events = mio.read_event_data(DATADIR, game_id)
        tracking_home = mio.tracking_data(DATADIR, game_id, 'Home')
        tracking_away = mio.tracking_data(DATADIR, game_id, 'Away')
        
        # Convert to metric coordinates
        tracking_home = mio.to_metric_coordinates(tracking_home)
        tracking_away = mio.to_metric_coordinates(tracking_away)
        events = mio.to_metric_coordinates(events)
        
        # Reverse direction for consistent analysis
        tracking_home, tracking_away, events = mio.to_single_playing_direction(
            tracking_home, tracking_away, events
        )
        
        # Calculate velocities
        print(f"  Calculating player velocities...")
        tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
        tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
        
        # Get pitch control parameters
        params = mpc.default_model_params()
        GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
        
        print(f"  ✓ Loaded {len(events)} events, {len(tracking_home):,} tracking frames")
        print(f"  ✓ Goalkeepers: Home #{GK_numbers[0]}, Away #{GK_numbers[1]}")
        
        # Build database for this game
        game_sequences = ss.build_sequence_database(
            events, tracking_home, tracking_away, params, GK_numbers,
            game_id, PFF_DATA_DIR, 
            min_events=MIN_EVENTS, 
            min_duration=MIN_DURATION,
            field_dimen=(106., 68.), 
            n_grid_cells_x=50
        )
        
        print(f"  ✓ Extracted {len(game_sequences)} sequences from game {game_id}")
        database.extend(game_sequences)
        print()
    
    print("=" * 60)
    print(f"✓ Combined database built with {len(database)} sequences from {len(GAME_IDS)} games")
    
    # Save database for future use
    print(f"Saving database to {db_filename}...")
    with open(db_path, 'wb') as f:
        pickle.dump(database, f)
    print("✓ Database saved")
    print()

# Show statistics by game
print("Database breakdown by game:")
for game_id in GAME_IDS:
    game_seqs = [s for s in database if s.game_id == game_id]
    # Group by unique team names
    teams = {}
    for s in game_seqs:
        team_name = s.team
        if team_name not in teams:
            teams[team_name] = 0
        teams[team_name] += 1
    
    print(f"  Game {game_id} ({GAME_NAMES[game_id]}): {len(game_seqs)} sequences")
    for team, count in teams.items():
        print(f"    {team}: {count}")
print()

# =============================================================================
# STEP 2: COMPARE ALL SEQUENCES WITH EACH OTHER
# =============================================================================
print("=" * 70)
print("STEP 2: COMPARING ALL SEQUENCES (PAIRWISE)")
print("=" * 70)
print()

print(f"Computing pairwise distances between {len(database)} sequences...")
print("This will calculate similarity between every pair of sequences")
print()

# Calculate all pairwise distances
n_sequences = len(database)
pairwise_results = []

for i in range(n_sequences):
    for j in range(i + 1, n_sequences):  # Only compare each pair once
        seq1 = database[i]
        seq2 = database[j]
        
        try:
            # Compare the two sequences
            distance, path, quality = ss.compare_sequences_dtw(
                seq1, seq2, distance_method='ssim'
            )
            
            pairwise_results.append({
                'seq1_id': seq1.sequence_id,
                'seq1_team': seq1.team,
                'seq1_events': len(seq1),
                'seq2_id': seq2.sequence_id,
                'seq2_team': seq2.team,
                'seq2_events': len(seq2),
                'distance': distance,
                'path_length': quality['path_length']
            })
        except Exception as e:
            continue
    
    # Progress update every 10 sequences
    if (i + 1) % 10 == 0:
        progress = (i + 1) / n_sequences * 100
        print(f"  Progress: {i+1}/{n_sequences} sequences ({progress:.1f}%)")

print()
print(f"✓ Computed {len(pairwise_results)} pairwise comparisons")
print()

# Sort by distance (most similar first)
pairwise_results.sort(key=lambda x: x['distance'])

# =============================================================================
# STEP 3: FIND MOST SIMILAR SEQUENCE PAIRS
# =============================================================================
print("=" * 70)
print("STEP 3: MOST SIMILAR SEQUENCE PAIRS")
print("=" * 70)
print()

print(f"TOP {TOP_K} MOST SIMILAR SEQUENCE PAIRS:")
print()

for rank, result in enumerate(pairwise_results[:TOP_K], start=1):
    # Find the actual sequence objects to get event details
    seq1 = [s for s in database if s.sequence_id == result['seq1_id']][0]
    seq2 = [s for s in database if s.sequence_id == result['seq2_id']][0]
    
    # Get game names
    game1_name = GAME_NAMES.get(seq1.game_id, f"Game {seq1.game_id}")
    game2_name = GAME_NAMES.get(seq2.game_id, f"Game {seq2.game_id}")
    
    print(f"{rank}. Sequence {result['seq1_id']} from {game1_name}")
    print(f"   ({result['seq1_team']}, {len(seq1)} events)")
    for i, metadata in enumerate(seq1.event_metadata[:len(seq1)], start=1):
        print(f"      Event {i}: {metadata['type']} by {metadata['player']}")
    
    print(f"   ↔ Sequence {result['seq2_id']} from {game2_name}")
    print(f"   ({result['seq2_team']}, {len(seq2)} events)")
    for i, metadata in enumerate(seq2.event_metadata[:len(seq2)], start=1):
        print(f"      Event {i}: {metadata['type']} by {metadata['player']}")
    
    print(f"   Distance: {result['distance']:.4f} (lower = more similar)")
    print(f"   DTW path length: {result['path_length']}")
    
    # Check if sequences are from same game or different games
    if seq1.game_id == seq2.game_id:
        print(f"   [Same game - repeated pattern within match]")
    else:
        print(f"   [Cross-game similarity - tactical convergence]")
    print()

print()
print("=" * 70)
print("TUTORIAL COMPLETE!")
print("=" * 70)
print()
print("Summary:")
print(f"✓ Loaded game data and calculated player velocities")
print(f"✓ Built database of {len(database)} possession sequences")
print(f"✓ Computed {len(pairwise_results)} pairwise sequence comparisons")
print(f"✓ Identified top {TOP_K} most similar sequence pairs")
print()
print("Key insights:")
print("- Lower distance = more similar pitch control evolution")
print("- SSIM compares spatial patterns (like comparing images)")
print("- DTW aligns sequences of different lengths")
print("- Can find tactically similar sequences within and across teams")
print()
print("Analysis observations:")
print("- Check if similar sequences are from same team (repeated patterns)")
print("- Or across teams (tactical convergence)")
print("- Similar sequences may represent common attacking/defending shapes")
print()
print("Next steps:")
print("- Analyze distance distribution (histogram)")
print("- Cluster sequences by similarity")
print("- Identify tactical patterns (e.g., wing attacks vs central)")
print("- Compare Home vs Away sequence similarities")
