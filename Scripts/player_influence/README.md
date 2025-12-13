# Player Influence Analysis Package

A modular package for analyzing individual player contributions to pitch control changes in football/soccer matches.

## Package Structure

```
player_influence/
├── __init__.py              # Package initialization and exports
├── config.py                # Configuration constants and settings
├── data_loader.py           # Data loading and preprocessing
├── pitch_control.py         # Pitch control calculations
├── influence_calculator.py  # Player influence analysis logic
├── visualization.py         # Plotting and movie generation
└── main.py                  # Main entry point
```

## Module Descriptions

### config.py
Contains all configuration constants:
- Directory paths
- Analysis parameters (FPS, tolerances)
- Pitch dimensions
- Visualization settings (colors, sizes)

### data_loader.py
Handles all data I/O operations:
- Loading tracking and event data
- Coordinate transformations
- Velocity calculations
- Frame selection utilities
- Player name resolution

### pitch_control.py
Pitch control calculation engine:
- Grid setup
- Surface calculation
- Attacking half filtering
- Velocity backfilling

### influence_calculator.py
Core analysis logic:
- Frame transition analysis
- Individual player influence calculation
- Aggregation and statistics
- Summary reporting

### visualization.py
All visualization functionality:
- Figure setup
- Pitch drawing
- Player plotting
- Bar chart animation
- Movie generation and saving

### main.py
Orchestrates the complete workflow:
- User input handling
- Module coordination
- Output management

## Usage

### From the Scripts folder:
```bash
python run_player_influence.py
```

### As a module:
```bash
cd Scripts
python -m player_influence.main
```

### Programmatic usage:
```python
from player_influence import DataLoader, PitchControlCalculator, InfluenceCalculator, Visualizer

# Load data
loader = DataLoader(game_id=10517)
loader.load_all()

# Get sequence data
sequence_events = loader.get_sequence_events(1)
event_frames = loader.get_event_frames(sequence_events)
frames = loader.get_frames_for_analysis(event_frames)

# Calculate influence
pc_calc = PitchControlCalculator(loader.gk_numbers)
influence_calc = InfluenceCalculator(loader, pc_calc, attacking_team='Home')
influence_calc.analyze_sequence(frames)

# Print results
influence_calc.print_summary(loader)

# Generate visualization
visualizer = Visualizer(loader, influence_calc)
movie_frames = loader.get_movie_frames(start_time, end_time)
visualizer.generate_movie(movie_frames, 'output.mp4', sequence_number=1)
```

## Output

The analysis produces:
1. **Console output**: Summary tables showing player rankings by influence
2. **MP4 video**: Animated visualization of the sequence with:
   - Player positions on pitch
   - Top 5 attackers/defenders highlighted with rankings
   - Stacked bar charts showing cumulative influence

## Dependencies

- numpy
- pandas
- matplotlib
- Metrica_IO, Metrica_Velocities, Metrica_PitchControl (project modules)
- FFmpeg (for video export)
