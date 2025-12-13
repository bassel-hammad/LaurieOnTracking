#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Player Influence Analysis - Main Entry Point

This script analyzes individual player contributions to pitch control changes
by isolating their movement while keeping all other players frozen.

Usage:
    python -m player_influence.main
    
Or run directly:
    python main.py
"""

import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .config import Config
from .data_loader import DataLoader
from .pitch_control import PitchControlCalculator
from .influence_calculator import InfluenceCalculator
from .visualization import Visualizer


def print_header():
    """Print the script header."""
    print("=" * 70)
    print("PLAYER INFLUENCE ANALYSIS")
    print("Individual Contribution to Pitch Control Changes")
    print("=" * 70)
    print()


def get_user_inputs():
    """Get game ID and sequence number from user."""
    # Get match ID
    game_id = input("Enter match ID (e.g., 10517): ").strip()
    if not game_id:
        print("ERROR: Match ID is required!")
        sys.exit(1)
    
    try:
        game_id = int(game_id)
    except ValueError:
        print("ERROR: Match ID must be a number!")
        sys.exit(1)
    
    print(f"\nSelected match ID: {game_id}")
    print()
    
    return game_id


def get_sequence_input(data_loader):
    """Get sequence number from user."""
    available_sequences = data_loader.get_available_sequences()
    print(f"\nAvailable sequences: {available_sequences}")
    
    sequence_input = input("Enter sequence number (e.g., 1): ").strip()
    
    if not sequence_input:
        print("ERROR: Sequence number is required!")
        sys.exit(1)
    
    try:
        sequence_number = float(sequence_input)
    except ValueError:
        print("ERROR: Sequence must be a number!")
        sys.exit(1)
    
    return sequence_number


def analyze_sequence(data_loader, sequence_number):
    """Analyze a specific sequence."""
    # Get sequence events
    sequence_events = data_loader.get_sequence_events(sequence_number)
    
    if len(sequence_events) == 0:
        print(f"ERROR: No events found for sequence {sequence_number}!")
        sys.exit(1)
    
    # Get frame range
    start_frame, end_frame, start_time, end_time = data_loader.get_sequence_frame_range(sequence_events)
    
    print(f"\nAnalyzing sequence {sequence_number}")
    print(f"  Events in sequence: {len(sequence_events)}")
    print(f"  Frame range: {start_frame} to {end_frame}")
    print(f"  Time window: {start_time:.1f}s to {end_time:.1f}s")
    print()
    
    # Show first few events
    print("First few events in this sequence:")
    print(sequence_events[['Team', 'Type', 'From', 'To', 'Start Frame', 'Start Time [s]']].head(10).to_string(index=False))
    print()
    
    # Determine attacking team
    team_counts = sequence_events['Team'].value_counts()
    attacking_team = team_counts.idxmax()
    defending_team = 'Away' if attacking_team == 'Home' else 'Home'
    
    print(f"Detected attacking team: {attacking_team}")
    print(f"Detected defending team: {defending_team}")
    print()
    
    return sequence_events, start_time, end_time, attacking_team, defending_team


def run_influence_analysis(data_loader, sequence_events, attacking_team):
    """Run the player influence analysis."""
    print("=" * 70)
    print("CALCULATING PLAYER INFLUENCES")
    print("=" * 70)
    print()
    print(f"NOTE: Only calculating pitch control in the ATTACKING HALF")
    print(f"      {attacking_team} attacks towards {'RIGHT (x > 0)' if attacking_team == 'Home' else 'LEFT (x < 0)'}")
    print()
    
    # Get frames to analyze
    event_frames = data_loader.get_event_frames(sequence_events)
    frames_to_analyze = data_loader.get_frames_for_analysis(event_frames)
    
    print(f"Analyzing {len(frames_to_analyze)} frames (at event frames)")
    print()
    
    # Create pitch control calculator
    pc_calculator = PitchControlCalculator(data_loader.gk_numbers)
    
    print(f"Goalkeepers: Home #{data_loader.gk_numbers[0]}, Away #{data_loader.gk_numbers[1]}")
    print()
    
    # Create influence calculator and run analysis
    influence_calc = InfluenceCalculator(data_loader, pc_calculator, attacking_team)
    influence_calc.analyze_sequence(frames_to_analyze, verbose=True)
    
    # Print summary
    influence_calc.print_summary(data_loader)
    
    return influence_calc, pc_calculator


def generate_movie(data_loader, influence_calc, sequence_number, start_time, end_time):
    """Generate the visualization movie and correlation plot."""
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()
    
    # Create visualizer
    visualizer = Visualizer(data_loader, influence_calc)
    
    # Generate correlation plot first
    print("Generating correlation plot...")
    correlation_filename = f'sequence_{int(sequence_number)}_correlation.png'
    correlation_path = Config.get_output_path(correlation_filename)
    correlation = visualizer.generate_correlation_plot(sequence_number, save_path=correlation_path)
    print(f"  Correlation coefficient: {correlation:.3f}")
    print()
    
    # Get movie frames
    movie_frames = data_loader.get_movie_frames(start_time, end_time)
    
    print(f"Calculating pitch control for movie at {Config.TARGET_FPS} FPS...")
    print(f"  Time range: {start_time:.1f}s to {end_time:.1f}s")
    print(f"  Found {len(movie_frames)} valid frames to render")
    print()
    
    output_filename = Config.get_movie_filename(sequence_number)
    output_path = Config.get_output_path(output_filename)
    
    visualizer.generate_movie(movie_frames, output_path, sequence_number)


def main():
    """Main entry point."""
    print_header()
    
    # Get user inputs
    game_id = get_user_inputs()
    
    # Load data
    data_loader = DataLoader(game_id)
    data_loader.load_all(verbose=True)
    
    # Get sequence
    sequence_number = get_sequence_input(data_loader)
    
    # Analyze sequence
    sequence_events, start_time, end_time, attacking_team, defending_team = analyze_sequence(
        data_loader, sequence_number
    )
    
    # Run influence analysis
    influence_calc, pc_calculator = run_influence_analysis(
        data_loader, sequence_events, attacking_team
    )
    
    # Generate movie
    generate_movie(data_loader, influence_calc, sequence_number, start_time, end_time)
    
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print()
    print(f"Match ID: {game_id}")
    print(f"Sequence: {sequence_number}")
    print(f"Time window: {start_time:.1f}s to {end_time:.1f}s")
    print(f"Events analyzed: {len(sequence_events)}")
    print()


if __name__ == "__main__":
    main()
