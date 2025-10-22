# LaurieOnTracking
Laurie's code for reading and working with Metrica's tracking and event data.

The sample data can be found in Metrica's github repository here: https://github.com/metrica-sports/sample-data

We'll be updating this repo as the friends of tracking series develops, adding code for measuring player velocity and acceleration, measuring team formations, and evaluating pitch control using a model published by Will Spearman. 

To create movies from the tracking data you'll need to have ffmpeg installed. You can do this by following the instructions here: https://anaconda.org/conda-forge/ffmpeg (you may need to restart python afterwards).


Tutorial Synopsis & Implementation Guide
-----------------------------------------

## Tutorial 1: Getting Started with Football Analytics
**YouTube**: https://www.youtube.com/watch?v=8TrleFklEsE  
**Script**: `Tutorial1_GettingStarted.py`

### What It Does:
An introduction to working with Metrica Sport's player tracking and event data, providing the foundation for all advanced football analytics.

### Key Features:
- **Data Loading & Processing**: Uses `Metrica_IO.py` to read tracking and event data from CSV files
- **Coordinate System**: Converts Metrica units to standard meters using `to_metric_coordinates()`
- **Event Analysis**: Analyzes shot frequency, goal locations, and passing sequences
- **Player Tracking**: Visualizes individual player trajectories over time
- **Formation Analysis**: Shows team positioning at kick-off and during key events

### Visualizations Created:
1. **Goal Location Plot**: Red dots showing shot positions with directional arrows
2. **Passing Sequence Plot**: Complete passing moves leading up to goals with arrows
3. **Player Trajectory Plot**: Movement paths of 5 players over 1500 frames (colored trails)
4. **Kick-off Formation**: Team formations at match start
5. **Goal Moment Analysis**: Player positions during goal events

### Core Modules Used:
- `Metrica_IO`: Data loading and coordinate conversion
- `Metrica_Viz`: Professional pitch visualization and plotting functions

### Technical Implementation:
- Pandas DataFrames for efficient data manipulation
- Matplotlib for professional football pitch visualizations
- Event filtering and analysis (shots, passes, goals by team/player)

---

## Tutorial 2: Advanced Player Performance & Video Analysis
**YouTube**: https://www.youtube.com/watch?v=VX3T-4lB2o0  
**Script**: `Tutorial2_DelvingDeeper.py`

### What It Does:
Deep dive into player physical performance metrics and video generation capabilities for tactical analysis.

### Key Features:
- **Velocity Calculations**: Uses `Metrica_Velocities.py` with Savitzky-Golay filtering for smooth velocity estimation
- **Physical Performance Metrics**: Distance covered, speed categorization, sprint analysis
- **Video Generation**: Creates MP4 clips of key match moments using FFmpeg integration
- **Playing Direction Normalization**: Ensures consistent attacking direction across both halves

### Advanced Analytics:
- **Speed Categories**: Walking (<2 m/s), Jogging (2-4 m/s), Running (4-7 m/s), Sprinting (>7 m/s)
- **Sprint Detection**: Identifies sustained sprints (>7 m/s for >1 second) using convolution techniques
- **Minutes Played**: Calculates actual playing time per player from tracking data
- **Physical Summary Reports**: Comprehensive performance dashboards per player

### Visualizations Created:
1. **Velocity Vector Plot**: Player positions with velocity arrows showing speed and direction
2. **Distance Bar Chart**: Total distance covered by each player
3. **Kick-off Formation**: Tactical positioning analysis
4. **Speed Category Analysis**: Clustered bar chart showing distance by intensity level
5. **Sprint Trajectory Map**: All sprint paths for individual players

### Video Generation:
- **Goal Movies**: Generates `first_goal.mp4` and `home_goal_2.mp4` (20-second clips)
- **Real-time Animation**: 25 FPS professional match analysis videos
- **Tactical Insights**: Shows player movements impossible to see in traditional analysis

### Core Modules Used:
- `Metrica_Velocities`: Advanced velocity calculation with noise filtering
- `Metrica_Viz`: Video generation and advanced plotting
- FFmpeg integration for professional video output

### Technical Implementation:
- Scipy signal processing for velocity smoothing
- Numpy convolution for sprint detection algorithms
- Matplotlib animation for video generation
- Pandas groupby operations for performance statistics

---

## Tutorial 3: Pitch Control Modeling
**YouTube**: https://www.youtube.com/watch?v=5X1cSehLg6s  
**Script**: `Tutorial3_PitchControl.py`

### What It Does:
Implements Will Spearman's pitch control model to evaluate passing options and possession probability across the entire field.

### Key Features:
- **Pitch Control Surfaces**: Calculates probability that each team will gain possession at any field location
- **Pass Success Prediction**: Evaluates likelihood of successful passes to different targets
- **Risk Assessment**: Identifies high-risk vs. safe passing options
- **Mathematical Modeling**: Physics-based player influence zones using position and velocity

### Advanced Analytics:
- **Player Influence Modeling**: Each player's ability to reach field locations based on position, velocity, and physical capabilities
- **Goalkeeper Integration**: Special handling for goalkeeper positioning and offside calculations
- **Real-time Probability Calculation**: Dynamic control surfaces that update as players move
- **Pass Risk Analysis**: Statistical analysis of risky pass outcomes

### Visualizations Created:
1. **Goal Sequence Events**: 3 passing events leading to goal with arrows and markers
2. **Pitch Control Heat Maps**: Color-coded probability surfaces (red=attacking team, blue=defending team)
3. **Pass Success Histogram**: Distribution of pass success probabilities
4. **Risky Pass Analysis**: Visualization of high-risk passes and their outcomes

### Scientific Implementation:
- **Spearman's Model**: Implements the research-grade pitch control algorithm
- **Probability Mathematics**: Gaussian influence functions for player control zones
- **Optimization**: Efficient grid-based calculations across 50x32 field divisions
- **Physics Integration**: Player velocity vectors influence control probability

### Core Modules Used:
- `Metrica_PitchControl`: Complete implementation of Spearman's pitch control model
- Advanced probability calculations and player modeling

### Key Insights:
- Risky passes (16% success probability) often led to ball loss but sometimes created scoring opportunities
- Pitch control reveals optimal passing targets invisible to traditional analysis
- Mathematical validation of tactical decision-making

---

## Tutorial 4: Expected Possession Value (EPV) Analysis
**YouTube**: https://www.youtube.com/watch?v=KXSLKwADXKI  
**Script**: `Tutorial4_EPV.py`

### What It Does:
The ultimate football analytics tutorial combining pitch control with Expected Possession Value to measure decision-making quality and pass value creation.

### Key Features:
- **EPV Surface Loading**: Uses pre-computed EPV grid (`EPV_grid.csv`) from thousands of historical possessions
- **Value-Added Calculation**: Quantifies how much each pass increases goal-scoring probability
- **Decision Quality Measurement**: Objective evaluation of player tactical decisions
- **Combined Analytics**: Integrates pitch control probability with possession value

### Revolutionary Analytics:
- **Pass Value Quantification**: Each pass assigned a numerical value (e.g., +0.093 EPV = 9.3% increase in goal probability)
- **Player Ranking**: Identifies best decision-makers based on EPV creation
- **Tactical Optimization**: Shows optimal passing targets at any match moment
- **Professional-Grade Analysis**: Same methodology used by top football clubs

### Top Performance Results (Sample Game 2):
**Home Team Best Passes:**
1. Event 1753: +0.093 EPV (assist to header)
2. Event 1478: +0.082 EPV 
3. Event 197: +0.063 EPV

**Away Team Best Passes:**
1. Event 1663: +0.073 EPV (assist to blocked shot)
2. Event 961: +0.070 EPV
3. Event 1901: +0.065 EPV

### Advanced Visualizations:
1. **EPV Heat Map Surface**: Complete field probability map for goal-scoring likelihood
2. **EPV + Pitch Control Combined**: Ultimate tactical analysis overlaying control and value
3. **Goal Sequence Analysis**: Value creation during actual goals
4. **Contour Visualizations**: Detailed probability landscapes for key passes
5. **Cross-field Pass Analysis**: Advanced tactical patterns

### Technical Implementation:
- **Pre-computed EPV Grid**: (32x50) probability surface from historical data analysis
- **Real-time Integration**: Combines live pitch control with static EPV values
- **Offside Modeling**: Advanced goalkeeper positioning for accurate calculations
- **Mathematical Rigor**: Research-grade algorithms validated in academic literature

### Core Modules Used:
- `Metrica_EPV`: Complete EPV calculation and grid loading system
- `Metrica_PitchControl`: Enhanced with offside detection
- Combined visualization system for ultimate tactical analysis

### Scientific Achievement:
This tutorial represents the pinnacle of football analytics, combining multiple research papers into a practical implementation that rivals professional club analysis departments.

---

## System Requirements
- Python 3.7+
- pandas, numpy, matplotlib, scipy
- FFmpeg (for video generation in Tutorial 2)

## Data Requirements
- Metrica Sports sample data (tracking + event data)
- EPV_grid.csv (included in repository)