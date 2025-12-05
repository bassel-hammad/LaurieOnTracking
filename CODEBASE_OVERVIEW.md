# LaurieOnTracking Codebase Overview

## 📋 Table of Contents
1. [Project Purpose](#project-purpose)
2. [Architecture Overview](#architecture-overview)
3. [Core Modules](#core-modules)
4. [Tutorial Files](#tutorial-files)
5. [PFF Adaptation Layer](#pff-adaptation-layer)
6. [Data Flow & Workflow](#data-flow--workflow)
7. [File-by-File Breakdown](#file-by-file-breakdown)

---

## 🎯 Project Purpose

**LaurieOnTracking** is a comprehensive football analytics framework created by Laurie Shaw (@EightyFivePoint) for the "Friends of Tracking" educational series. It provides:

- **Player tracking analysis** - Positions, velocities, trajectories
- **Event data processing** - Shots, passes, goals, challenges
- **Pitch control modeling** - Probability surfaces showing possession likelihood
- **Expected Possession Value (EPV)** - Evaluating passing options and decision-making
- **Video generation** - Creating tactical analysis videos from tracking data

The codebase originally supported **Metrica Sports sample data** and has been extended with a **PFF (Pro Football Focus) adapter** to work with FIFA World Cup 2022 Final data (Argentina vs France).

---

## 🏗️ Architecture Overview

The project follows a **modular architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    TUTORIAL SCRIPTS                         │
│  (Tutorial1-4 for Metrica, Tutorial1-4_PFF for World Cup)   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                   CORE MODULES (Metrica_*.py)               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │   I/O    │ │   Viz    │ │Velocities│ │  Pitch   │        │
│  │          │ │          │ │          │ │ Control  │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
│  ┌──────────┐                                               │
│  │   EPV    │                                               │
│  └──────────┘                                               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA ADAPTER LAYER                         │
│              (PFF_to_Metrica_Adapter.py)                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAW DATA                               │
│  Sample Data/   (Metrica CSV)                               │
│  PFF Data/      (JSON/JSONL from World Cup)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Core Modules

### 1. **Metrica_IO.py** - Data Input/Output
**Purpose**: Load and preprocess football tracking and event data

**Key Functions**:
- `read_match_data(DATADIR, gameid)` - Load complete match data (tracking + events)
- `read_event_data(DATADIR, game_id)` - Load event data (passes, shots, etc.)
- `tracking_data(DATADIR, game_id, teamname)` - Load tracking data for one team
- `merge_tracking_data(home, away)` - Combine home and away tracking
- `to_metric_coordinates(data)` - Convert from 0-1 normalized to meters
- `to_single_playing_direction(home, away, events)` - Normalize attack direction
- `find_goalkeeper(team)` - Identify goalkeeper (closest to goal at kickoff)
- `find_playing_direction(team, teamname)` - Determine attack direction

**How It Works**:
1. Reads CSV files with flexible player count support
2. Parses headers to extract player jersey numbers
3. **Auto-detects PFF speed columns** (`pff_speed` in header row 3)
4. Creates dynamic column names (`Home_1_x`, `Home_1_y`, `Home_1_pff_speed`, etc.)
5. Converts coordinates from Metrica's 0-1 scale to meters (106m x 68m pitch)
6. Flips coordinates for second half/extra time to ensure consistent attack direction

**PFF Speed Detection**:
```python
# Automatically detects if tracking CSV includes pff_speed columns
has_pff_speed = any('pff_speed' in str(col) for col in row2)
if has_pff_speed:
    columns.extend([f"{teamname}_{jersey}_x", f"{teamname}_{jersey}_y", 
                   f"{teamname}_{jersey}_visibility", f"{teamname}_{jersey}_pff_speed"])
```

**Data Flow**:
```
CSV File → Pandas DataFrame → Coordinate conversion → Direction normalization
```

---

### 2. **Metrica_Viz.py** - Visualization
**Purpose**: Create professional football pitch visualizations, plots, and videos

**Key Functions**:
- `plot_pitch()` - Draw a professional football pitch
- `plot_frame()` - Visualize single tracking frame with players and ball
- `plot_events()` - Plot event sequences (passes, shots) with markers/arrows
- `plot_pitchcontrol_for_event()` - Overlay pitch control surfaces on events
- `plot_EPV()` - Visualize Expected Possession Value surface
- `plot_EPV_for_event()` - Combine EPV and pitch control for specific events
- `save_match_clip()` - Generate MP4 video clips from tracking data

**Advanced Features**:
- **Velocity backfilling**: Estimates missing player velocities from neighboring frames
- **Video generation**: Uses FFmpeg to create 25 FPS match analysis videos
- **Professional styling**: Arsenal/Barcelona-style team colors, pitch markings
- **Dynamic annotations**: Player numbers, event markers, velocity arrows

**How It Works**:
1. Uses Matplotlib for 2D pitch rendering
2. Plots player positions as circles (colored by team)
3. Adds velocity arrows when available
4. Overlays heatmaps for pitch control/EPV surfaces
5. Animates frames using `matplotlib.animation` for video output
6. Saves videos via FFmpeg integration

**Visualization Types**:
- Static pitch plots (formations, events)
- Trajectory plots (player movement over time)
- Heatmaps (pitch control, EPV surfaces)
- Animated videos (match clips, tactical analysis)

---

### 3. **Metrica_Velocities.py** - Player Velocity Calculation
**Purpose**: Calculate smooth, accurate player velocities from noisy position data

**Key Functions**:
- `calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed=12)`
  - Calculates `vx`, `vy`, `speed` for each player
  - Removes outliers exceeding `maxspeed` (default 12 m/s)
  - Applies smoothing filter to reduce noise
- `calc_player_velocities_hybrid(team, smoothing=True, use_pff_speed=True)` **(NEW)**
  - Hybrid approach using PFF's raw speed values with calculated direction
  - More accurate than standard method for PFF data
- `remove_player_velocities(team)` - Remove existing velocity columns (preserves `_pff_speed`)

**How It Works**:

#### Standard Method (for Metrica data):
1. **Finite difference**: `vx = dx/dt`, `vy = dy/dt`
2. **Outlier removal**: Flags unrealistic speeds (likely position errors)
3. **Savitzky-Golay filtering**: Fits polynomial to velocity within sliding window
   - Default: 7-frame window, linear polynomial (polyorder=1)
   - Smooths noise while preserving acceleration trends
4. **Half-aware processing**: Applies filter separately to each half to avoid boundary issues

#### Hybrid Method (for PFF data):
The hybrid method was introduced to address a key limitation: **PFF exports positions at ~3.75 FPS (after deduplication) but calculates speed internally at ~15 FPS**.

1. **Direction from positions**: Calculate movement direction from position differences
2. **Magnitude from PFF speed**: Use PFF's raw `speed` field (more accurate, higher FPS)
3. **Combine**: `vx = pff_speed * dir_x`, `vy = pff_speed * dir_y`
4. **Scaling**: Convert from m/s to normalized velocity components

**Why the Hybrid Method is More Accurate**:
| Source | FPS | Accuracy |
|--------|-----|----------|
| PFF internal speed | ~15 FPS | High (ground truth) |
| Position differences | ~3.75 FPS | Lower (underestimates by ~10%) |

Test results show:
- **Hybrid mean speed: 1.88 m/s** (matches PFF raw)
- **Standard mean speed: 1.71 m/s** (underestimates)

**Why Smoothing Matters**:
- Raw position data has measurement noise
- Direct differentiation amplifies noise dramatically
- Savitzky-Golay filter preserves true velocity trends while removing noise
- Allows accurate acceleration analysis (gradient of velocity)

**Output**:
```
Player columns added:
- Home_1_vx: X-velocity (m/s)
- Home_1_vy: Y-velocity (m/s)  
- Home_1_speed: Total speed (m/s)
- Home_1_pff_speed: PFF raw speed (m/s) - preserved for reference
```

---

### 4. **Metrica_PitchControl.py** - Pitch Control Model
**Purpose**: Calculate probability of each team gaining possession at any field location

**Mathematical Foundation**:
Based on William Spearman's model: ["Off the Ball Scoring Opportunities"](http://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf)

**Key Concepts**:
- **Pitch Control (PC)**: Probability team will reach ball first at location `(x,y)`
- **Time to Intercept (TTI)**: How long each player takes to reach `(x,y)`
- **Influence Function**: Player's control decreases with distance/time

**Key Functions**:
- `generate_pitch_control_for_event(event_id, ...)` - Full pitch control surface (50x32 grid)
- `calculate_pitch_control_at_target(target_pos, attacking_players, defending_players, ...)` - PC at specific location
- `initialise_players(team, teamname, params, GKid)` - Create player objects with positions/velocities
- `check_offsides(attacking_players, defending_players, ball_position, ...)` - Flag offside players
- `default_model_params()` - Load default model parameters

**How It Works**:

1. **Player State Extraction**:
   - Position `(x, y)` from tracking data
   - Velocity `(vx, vy)` from Metrica_Velocities (or backfilled)
   - Reaction time, max speed parameters

2. **Time-to-Intercept Calculation**:
   ```python
   # For each player, calculate time to reach target location
   TTI = simple_time_to_intercept(player_pos, player_velocity, target_pos, params)
   ```

3. **Influence Calculation**:
   ```python
   # Player's control influence at target location
   influence = 1 / (1 + exp((TTI - tau) / kappa))
   ```
   Where:
   - `tau`: Characteristic time scale
   - `kappa`: Sharpness parameter

4. **Team Pitch Control**:
   ```python
   # Attacking team PC = product of (1 - defender_influence) for all defenders
   PC_attack = attack_influence * product(1 - def_influence for def in defenders)
   PC_defend = 1 - PC_attack
   ```

5. **Offside Handling**:
   - Players in offside positions have zero influence
   - Identified by position relative to second-last defender

**Output**:
- `PPCF`: 2D numpy array (32x50 grid) with values 0-1
- `xgrid, ygrid`: Coordinate grids for plotting
- Each cell = probability attacking team controls that location

**Use Cases**:
- Evaluate pass success probability
- Identify dangerous attacking zones
- Assess defensive coverage
- Tactical decision-making analysis

---

### 5. **Metrica_EPV.py** - Expected Possession Value
**Purpose**: Quantify the value of ball possession at any field location

**Key Concept**:
**EPV = Probability that current possession will end in a goal, given ball location**

**Key Functions**:
- `load_EPV_grid(fname='EPV_grid.csv')` - Load precomputed EPV surface (32x50 grid)
- `get_EPV_at_location(position, EPV, attack_direction)` - EPV value at `(x,y)`
- `calculate_epv_added(event_id, events, tracking_home, tracking_away, GK_numbers, EPV, params)` - Value added by a pass
- `find_max_value_added_target(event_id, ...)` - Find optimal pass target across entire field

**How It Works**:

1. **EPV Surface**:
   - Precomputed grid showing goal probability from each location
   - Higher values near opponent's goal
   - Trained on historical possession data

2. **Expected EPV (EEPV)**:
   ```python
   # Account for success probability using pitch control
   EEPV = Pitch_Control * EPV
   ```

3. **EPV Added**:
   ```python
   # Value added by moving ball from start to target
   EEPV_added = (PC_target * EPV_target) - (PC_start * EPV_start)
   ```
   Where:
   - `PC_target`: Pitch control probability at pass target
   - `EPV_target`: EPV value at pass target
   - `PC_start`: Pitch control probability at pass start
   - `EPV_start`: EPV value at pass start

4. **Pass Evaluation**:
   - Positive EEPV_added = good pass (increases goal probability)
   - Negative EEPV_added = risky pass (decreases goal probability)
   - Magnitude shows how much value was added/lost

**Workflow Example**:
```python
# Load EPV surface
EPV = load_EPV_grid('EPV_grid.csv')

# Calculate value added by pass event 822
EEPV_added, EPV_diff = calculate_epv_added(
    822, events, tracking_home, tracking_away, 
    GK_numbers, EPV, params
)
# EEPV_added might be +0.05 (5% increased goal probability)
```

**Use Cases**:
- Evaluate passing decisions
- Identify best passing options
- Assess risk/reward of passes
- Compare player decision-making quality

---

## 📚 Tutorial Files

### Tutorial 1: Getting Started (`Tutorial1_GettingStarted.py` / `Tutorial1_PFF.py`)
**Purpose**: Introduction to tracking and event data

**What It Does**:
1. Loads event and tracking data
2. Converts coordinates to meters
3. Analyzes shots and goals
4. Plots goal locations with directional arrows
5. Visualizes passing sequences leading to goals
6. Plots player trajectories over time
7. Shows team formations at kickoff

**Key Visualizations**:
- Goal location scatter plot
- Passing sequence diagrams
- Player trajectory trails (5 players, 1500 frames)
- Kick-off formations

**Modules Used**: `Metrica_IO`, `Metrica_Viz`

---

### Tutorial 2: Physical Performance (`Tutorial2_DelvingDeeper.py` / `Tutorial2_PFF.py`)
**Purpose**: Player physical metrics and video generation

**What It Does**:
1. Calculates player velocities using Savitzky-Golay filter
2. Measures distance covered per player
3. Categorizes movement by speed:
   - Walking: < 2 m/s
   - Jogging: 2-4 m/s
   - Running: 4-7 m/s
   - Sprinting: > 7 m/s
4. Detects sustained sprints (> 7 m/s for > 1 second)
5. Generates MP4 video clips of goals (requires FFmpeg)

**Key Visualizations**:
- Velocity vector plots (arrows showing player speeds)
- Distance covered bar charts
- Speed category analysis (clustered bars)
- Sprint trajectory maps
- Animated match videos (25 FPS)

**Modules Used**: `Metrica_IO`, `Metrica_Viz`, `Metrica_Velocities`

**Video Output**: `first_goal.mp4`, `home_goal_2.mp4` (20-second clips)

---

### Tutorial 3: Pitch Control (`Tutorial3_PitchControl.py` / `Tutorial3_PFF.py`)
**Purpose**: Evaluate passing options using pitch control model

**What It Does**:
1. Implements Spearman's pitch control model
2. Normalizes playing direction (both halves attack same direction)
3. Calculates velocity-smoothed tracking data
4. Generates pitch control surfaces for key events
5. Evaluates pass success probability
6. Identifies safe vs. risky passing zones

**Key Visualizations**:
- Pitch control heatmaps (red=attacking, blue=defending)
- Pass event sequences with PC overlays
- Success probability for all passes

**Modules Used**: `Metrica_IO`, `Metrica_Viz`, `Metrica_Velocities`, `Metrica_PitchControl`

**Example Workflow**:
```python
# Get pitch control for pass event 820
PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
    820, events, tracking_home, tracking_away, 
    params, GK_numbers, field_dimen=(106., 68.)
)
# Plot overlay
mviz.plot_pitchcontrol_for_event(820, events, tracking_home, tracking_away, PPCF)
```

---

### Tutorial 4: Expected Possession Value (`Tutorial4_EPV.py` / `Tutorial4_PFF.py`)
**Purpose**: Quantify pass value using EPV surfaces

**What It Does**:
1. Loads precomputed EPV grid (`EPV_grid.csv`)
2. Plots EPV surface (shows goal probability from each location)
3. Calculates EPV-added for every pass
4. Combines pitch control + EPV for decision analysis
5. Finds optimal pass targets across the field
6. Ranks passes by value added

**Key Visualizations**:
- EPV surface heatmaps
- EPV-added for specific passes
- Combined PC + EPV surfaces showing expected value
- Pass value distribution histograms

**Modules Used**: All core modules (`Metrica_IO`, `Metrica_Viz`, `Metrica_Velocities`, `Metrica_PitchControl`, `Metrica_EPV`)

**Example Output**:
```
Pass EPV-added: +0.053
(This pass increased goal probability by 5.3%)
```

---

## 🔄 PFF Adaptation Layer

### **PFF_to_Metrica_Adapter.py**
**Purpose**: Convert PFF World Cup data to Metrica format for tutorial compatibility

**What It Does**:
- Reads PFF JSON/JSONL files (tracking + event data)
- Converts coordinate systems (PFF meters → Metrica 0-1 scale)
- **Flips Y-axis** (PFF: y↑, Metrica: y↓)
- Maps PFF event types to Metrica event types
- **Extracts PFF raw speed values** for hybrid velocity calculation
- Smart sampling (every 4th frame + event frames) for ~7.5 FPS output
- Generates Metrica-compatible CSV files with extended columns

**Key Classes**:
- `PFFToMetricaAdapter` - Main conversion class

**Key Methods**:
- `normalize_coordinates(x_meters, y_meters)` - Convert PFF → Metrica coordinates
- `map_outcome_to_subtype(possession_data, event_type)` - Map PFF outcomes to Metrica subtypes
- `process_tracking_frame(frame_data)` - Extract positions, visibility, and **pff_speed**
- `convert_tracking_data_smart_sampling()` - Convert JSONL tracking with intelligent sampling
- `convert_event_data()` - Convert JSON events to CSV
- `run_conversion()` - Execute complete pipeline

**Coordinate Conversion**:
```python
# PFF: Center origin, meters, y increases upward
# Metrica: Top-left origin, 0-1 scale, y increases downward

x_normalized = (x_meters + field_length/2) / field_length
y_flipped = -y_meters  # FLIP Y-AXIS
y_normalized = (y_flipped + field_width/2) / field_width
```

**PFF Speed Data Extraction**:
PFF tracking data includes raw `speed` values calculated internally at ~15 FPS:
```python
# For each player in frame
pff_speed = player.get('speed', 0.0)  # Raw speed in m/s
frame_info[f'Home_{jersey}_pff_speed'] = pff_speed
```

This speed data is preserved in the CSV and used by `calc_player_velocities_hybrid()` for more accurate velocity calculations than position-based differentiation.

**Event Mapping**:
```
PFF → Metrica
PA (Pass) → PASS
SH (Shot) → SHOT
CR (Cross) → PASS-CROSS
CH (Challenge) → CHALLENGE
CL (Clearance) → CLEARANCE
...
```

**Output Structure**:
```
Sample Data/Sample_Game_10517/
  ├── Sample_Game_10517_RawEventsData.csv
  ├── Sample_Game_10517_RawTrackingData_Home_Team.csv  (includes pff_speed columns)
  └── Sample_Game_10517_RawTrackingData_Away_Team.csv  (includes pff_speed columns)
```

**Tracking CSV Column Format** (per player):
- `x` - Normalized X position (0-1)
- `y` - Normalized Y position (0-1)
- `visibility` - VISIBLE or ESTIMATED
- `pff_speed` - Raw speed from PFF in m/s

---

### **generate_epv_results_csv.py**
**Purpose**: Batch process all events and export EPV-added values to CSV

**What It Does**:
1. Loads match data (tracking + events)
2. Calculates velocities
3. Normalizes playing direction
4. Computes EPV-added for every pass event
5. Exports results to CSV for external analysis

**Output CSV Columns**:
```
Event_ID, Period, Time, Team, Type, From, To, 
Start_X, Start_Y, End_X, End_Y, 
EEPV_Added, EPV_Difference
```

**Use Case**: Batch analytics, statistical analysis, exporting to BI tools

---

## 📊 Data Flow & Workflow

### **Standard Analysis Workflow**

```
1. LOAD DATA
   ↓
   read_match_data() → tracking_home, tracking_away, events
   
2. COORDINATE CONVERSION  
   ↓
   to_metric_coordinates() → Convert 0-1 to meters (106m x 68m)
   
3. DIRECTION NORMALIZATION
   ↓
   to_single_playing_direction() → Both halves attack same direction
   
4. VELOCITY CALCULATION
   ↓
   calc_player_velocities() → Add vx, vy, speed columns
   
5. ANALYSIS (Choose path)
   ↓
   ├─→ PITCH CONTROL
   │   ├─ default_model_params()
   │   ├─ find_goalkeeper()  
   │   ├─ generate_pitch_control_for_event()
   │   └─ plot_pitchcontrol_for_event()
   │
   └─→ EPV ANALYSIS
       ├─ load_EPV_grid()
       ├─ calculate_epv_added()
       ├─ find_max_value_added_target()
       └─ plot_EPV_for_event()
```

### **Data Structures**

**Tracking DataFrame Structure**:
```
Index: Frame (integer)
Columns:
  - Period: 1, 2, 3, 4 (half/extra time)
  - Time [s]: Timestamp (float)
  - Home_1_x, Home_1_y: Player 1 position (meters)
  - Home_1_vx, Home_1_vy: Player 1 velocity (m/s)
  - Home_1_speed: Player 1 total speed (m/s)
  - ... (repeat for all players)
  - ball_x, ball_y: Ball position (meters)
```

**Event DataFrame Structure**:
```
Index: Event ID (integer)
Columns:
  - Type: PASS, SHOT, CHALLENGE, etc.
  - Subtype: PASS-CROSS, SHOT-GOAL, etc.
  - Period: 1, 2, 3, 4
  - Start Frame: Frame index (links to tracking data)
  - Start Time [s]: Event timestamp
  - Team: Home/Away
  - From: Player initiating event
  - To: Player receiving (for passes)
  - Start X, Start Y: Event start position (meters)
  - End X, End Y: Event end position (meters)
```

---

## 📁 File-by-File Breakdown

### Core Library Files

| File | Lines | Purpose | Key Functions |
|------|-------|---------|---------------|
| `Metrica_IO.py` | 200 | Data loading & preprocessing | `read_match_data`, `to_metric_coordinates`, `to_single_playing_direction` |
| `Metrica_Viz.py` | 520 | Visualization & video generation | `plot_pitch`, `plot_frame`, `save_match_clip`, `plot_EPV_for_event` |
| `Metrica_Velocities.py` | 100 | Velocity calculation | `calc_player_velocities` (Savitzky-Golay filtering) |
| `Metrica_PitchControl.py` | 462 | Pitch control model | `generate_pitch_control_for_event`, `calculate_pitch_control_at_target` |
| `Metrica_EPV.py` | 218 | Expected Possession Value | `load_EPV_grid`, `calculate_epv_added`, `find_max_value_added_target` |

### Tutorial Files (Metrica Sample Data)

| File | Lines | Dataset | Topics Covered |
|------|-------|---------|----------------|
| `Tutorial1_GettingStarted.py` | 101 | Sample Game 2 | Event data, shot analysis, trajectories, formations |
| `Tutorial2_DelvingDeeper.py` | ~150 | Sample Game 2 | Velocities, distance, sprints, video generation |
| `Tutorial3_PitchControl.py` | 125 | Sample Game 2 | Pitch control surfaces, pass probability |
| `Tutorial4_EPV.py` | 174 | Sample Game 2 | EPV surfaces, value-added analysis |

### Tutorial Files (PFF World Cup Data)

| File | Lines | Dataset | Topics Covered |
|------|-------|---------|----------------|
| `Tutorial1_PFF.py` | 297 | Game 10517 (WC Final) | Event/tracking analysis adapted for PFF data |
| `Tutorial2_PFF.py` | ~250 | Game 10517 | Physical performance metrics for World Cup match |
| `Tutorial3_PFF.py` | ~200 | Game 10517 | Pitch control for World Cup data |
| `Tutorial4_PFF.py` | ~250 | Game 10517 | EPV analysis for World Cup Final |

### Adapter & Utility Files

| File | Lines | Purpose |
|------|-------|---------|
| `PFF_to_Metrica_Adapter.py` | 914 | Convert PFF JSON/JSONL → Metrica CSV format |
| `generate_epv_results_csv.py` | 221 | Batch EPV calculation → CSV export |

### Data Files

| File/Folder | Type | Content |
|-------------|------|---------|
| `EPV_grid.csv` | CSV | Precomputed EPV surface (32x50 grid) |
| `Sample Data/Sample_Game_1/` | CSV | Metrica sample match 1 (tracking + events) |
| `Sample Data/Sample_Game_2/` | CSV | Metrica sample match 2 (tracking + events) |
| `Sample Data/Sample_Game_10517/` | CSV | Converted World Cup Final data |
| `PFF Data/Tracking Data/*.jsonl` | JSONL | Raw PFF tracking data (29.97 FPS) |
| `PFF Data/Event Data/*.json` | JSON | Raw PFF event data |
| `PFF Data/Meta Data/*.json` | JSON | Match metadata (pitch dims, FPS, teams) |

### Configuration Files

| File | Purpose |
|------|---------|
| `.gitignore` | Ignore Python artifacts, data folders, output files |
| `README.md` | Project overview and tutorial descriptions |
| `LICENSE` | MIT License |

---

## 🔄 Typical Use Cases

### **Use Case 1: Analyze Passing Decisions**
```python
# Load data
tracking_home, tracking_away, events = mio.read_match_data('Sample Data', 2)

# Preprocess
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)
tracking_home, tracking_away, events = mio.to_single_playing_direction(
    tracking_home, tracking_away, events
)

# Calculate velocities
tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

# Setup pitch control
params = mpc.default_model_params()
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]

# Load EPV
EPV = mepv.load_EPV_grid('EPV_grid.csv')

# Analyze specific pass (event 822)
EEPV_added, EPV_diff = mepv.calculate_epv_added(
    822, events, tracking_home, tracking_away, GK_numbers, EPV, params
)
print(f"Pass value added: {EEPV_added:.3f}")

# Visualize
PPCF, xgrid, ygrid = mpc.generate_pitch_control_for_event(
    822, events, tracking_home, tracking_away, params, GK_numbers
)
mviz.plot_EPV_for_event(822, events, tracking_home, tracking_away, PPCF, EPV)
```

### **Use Case 2: Generate Match Video Clip**
```python
# Load and preprocess data (as above)

# Find goal frame
goal_event = events[(events['Type']=='SHOT') & 
                    (events['Subtype'].str.contains('GOAL'))].iloc[0]
goal_frame = goal_event['Start Frame']

# Generate 10-second clip (250 frames at 25 FPS)
mviz.save_match_clip(
    tracking_home, tracking_away,
    DATADIR, 'first_goal.mp4',
    frames_per_second=25,
    team_colors=('r', 'b'),
    include_player_velocities=True,
    PlayerMarkerSize=10,
    PlayerAlpha=0.7
)
```

### **Use Case 3: Convert PFF Data to Metrica Format**
```python
from PFF_to_Metrica_Adapter import PFFToMetricaAdapter

# Initialize adapter
adapter = PFFToMetricaAdapter(
    pff_data_dir='PFF Data',
    output_dir='Sample Data',
    game_id='10517'
)

# Run full conversion
adapter.run_full_conversion()

# Now use converted data with standard tutorials
tracking_home, tracking_away, events = mio.read_match_data('Sample Data', 10517)
```

---

## 🎓 Learning Path Recommendation

**For Beginners**:
1. Start with `Tutorial1_GettingStarted.py` - understand data structures
2. Move to `Tutorial2_DelvingDeeper.py` - learn physical metrics
3. Study `Tutorial3_PitchControl.py` - grasp spatial control concepts
4. Advance to `Tutorial4_EPV.py` - master decision evaluation

**For Advanced Users**:
1. Read `Metrica_PitchControl.py` - understand model implementation
2. Study `Metrica_EPV.py` - learn EPV calculation details
3. Explore `PFF_to_Metrica_Adapter.py` - adapt to new data formats
4. Modify parameters in `default_model_params()` for custom models

**For Data Scientists**:
1. Use `generate_epv_results_csv.py` for batch analytics
2. Export results to Pandas/R for statistical modeling
3. Train custom EPV surfaces using historical data
4. Build predictive models on top of pitch control features

---

## 🚀 Key Takeaways

### **What Makes This Codebase Powerful**:
1. **Modular Design**: Each module has clear responsibility (I/O, viz, velocity, PC, EPV)
2. **Flexible Data Support**: Works with Metrica CSV and PFF JSON via adapter pattern
3. **Research-Grade Models**: Implements published academic models (Spearman's PC)
4. **Production-Ready**: Smooth velocity calculation, outlier handling, video generation
5. **Educational Focus**: Well-commented tutorials with progressive complexity

### **Common Workflows**:
- **Basic Analysis**: Load → Convert → Visualize
- **Physical Performance**: Load → Convert → Velocities → Metrics
- **Tactical Analysis**: Load → Convert → Velocities → Pitch Control → Visualize
- **Decision Evaluation**: Load → Convert → Velocities → Pitch Control → EPV → Rank

### **Extension Points**:
- Add new data adapters (e.g., StatsBomb, Wyscout)
- Implement alternative pitch control models
- Train custom EPV surfaces from your data
- Build real-time analysis pipelines
- Create interactive dashboards (Plotly/Dash)

---

## 📞 Resources

- **GitHub**: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking
- **YouTube**: Friends of Tracking series
- **Data**: https://github.com/metrica-sports/sample-data
- **Author**: Laurie Shaw (@EightyFivePoint)
- **Paper**: [Spearman - Off the Ball Scoring Opportunities](http://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf)

---

**Created**: October 2025  
**Author**: Automated Documentation  
**Version**: 2.0 (PFF Extended)
