# Player Influence Analysis

## Overview

`generate_player_influence_analysis.py` quantifies individual player contributions to pitch control dynamics using a **sequential perturbation method**. This tool reveals which players created or destroyed space during key match sequences, and how much of the effect was due to coordinated team movement versus individual actions.

---

## What It Does

### Core Functionality

The script analyzes a specific time window (default: 12 seconds around Di María's goal in the 2022 World Cup Final) and:

1. **Isolates each player's contribution** to pitch control changes
2. **Ranks players** by their total influence on space creation/destruction
3. **Calculates interaction effects** showing team coordination vs individual impact
4. **Generates spatial heatmaps** visualizing where each player created the most space

### Output Files

- **`player_influence_results.npz`**: Complete analysis data (pitch control surfaces, individual contributions, player rankings)
- **`player_influence_[PlayerName].png`**: Top 5 players' spatial influence heatmaps
- **Console output**: Rankings table and interaction statistics

---

## How It Works: The Sequential Perturbation Method

### The Core Idea

**Question**: When the pitch control changes from frame `t` to `t+1`, how much did *each individual player* contribute?

**Method**: Freeze all players at time `t`, then move *only one player* to their `t+1` position. Measure the change. This isolates that player's individual effect.

### Mathematical Framework

For each frame transition (`t` → `t+1`):

1. **Baseline**: Calculate pitch control with all players at time `t`
   ```
   PC_baseline = PitchControl(all_players_at_t)
   ```

2. **Actual**: Calculate pitch control with all players at time `t+1`
   ```
   PC_actual = PitchControl(all_players_at_t+1)
   ΔPC_actual = PC_actual - PC_baseline
   ```

3. **Individual Attribution**: For each player `i`:
   ```
   PC_i = PitchControl(only_player_i_moves_to_t+1, others_stay_at_t)
   ΔPC_i = PC_i - PC_baseline
   ```

4. **Interaction Term**: The coordinated effect beyond individual contributions
   ```
   ΔPC_sum = sum(ΔPC_i for all players)
   Interaction = ΔPC_actual - ΔPC_sum
   ```

### Why This Method?

**Advantages over alternatives:**

- ✅ **Exact causal attribution**: No approximations or gradients
- ✅ **Simple interpretation**: "If only this player moved, here's the effect"
- ✅ **Reveals synergies**: Interaction term shows coordinated movement effects
- ✅ **No assumptions** about player influence functions or spatial models

**Conceptual analogy**: Like A/B testing in experiments - change one variable, measure the effect, isolate causality.

---

## Interpreting The Results

### Player Rankings

The console output shows:

```
Player                    Total        Positive     Negative     Frames
----------------------------------------------------------------------
Ángel Di María                824.083     437.365    -386.718      33
Alexis Mac Allister           372.875     283.834     -89.041      36
Julián Álvarez                202.814     113.897     -88.916      37
Lionel Messi                   87.875      43.061     -44.814      33
```

#### What Each Column Means:

- **Total**: Sum of absolute changes `sum(|ΔPC_i|)` across all grid cells
  - **High total** = player caused dramatic pitch control changes (good for attackers!)
  - This is the **magnitude** of influence, not direction

- **Positive**: Space created for own team `sum(ΔPC_i where ΔPC_i > 0)`
  - Red zones on heatmap
  - Represents offensive space creation

- **Negative**: Space conceded to opponents `sum(ΔPC_i where ΔPC_i < 0)`
  - Blue zones on heatmap
  - Represents defensive space loss (or tactical withdrawal)

- **Frames**: Number of frames (out of 60) where player was in attacking half
  - Players deeper/wider appear in fewer frames

#### Interpreting Player Roles:

**High Total, Balanced +/-** (e.g., Di María: 824.1 total, +437/-387)
- **Role**: Primary attacking threat
- **Interpretation**: Constantly creating AND destroying space through dynamic movement
- **Tactical meaning**: Forcing defensive reactions, pulling defenders, creating chaos

**High Positive, Low Negative** (e.g., Mac Allister: +284/-89)
- **Role**: Space creator / Support player
- **Interpretation**: Efficient movement into unmarked areas
- **Tactical meaning**: Finding pockets of space, creating passing options without drawing pressure

**Moderate Total** (e.g., Álvarez: 203 total)
- **Role**: Occupying defenders / Target striker
- **Interpretation**: Movement attracts defensive attention (negative) but opens space for others (positive)
- **Tactical meaning**: "Decoy runs" or holding defenders to free teammates

**Low Total** (e.g., Messi: 88 total)
- **Possible interpretations**:
  1. Player was relatively stationary (waiting to receive)
  2. Already tightly marked (defenders stick regardless of movement)
  3. Positioned in areas where movement has less PC impact
  4. Gravitational threat (presence alone controls space, movement less important)

### The Interaction Term (84%)

```
Total actual ΔPC:        3249.447
Total interaction term:  2730.084
Interaction percentage:  84.0%
```

#### What This Means:

**Interaction % = (Actual ΔPC - Sum of Individual ΔPC) / Actual ΔPC × 100**

- **Low interaction (<40%)**: Players moving mostly independently
- **Medium interaction (40-70%)**: Some coordination, some individual brilliance
- **High interaction (>70%)**: Highly coordinated team movement - **THIS IS WHAT WE SEE!**

#### Why 84% Is Remarkable:

**84% means**: The actual pitch control change was **6.3× larger** than what individual movements would predict!

**Physical explanation**:
1. **Defensive overload**: When multiple attackers move simultaneously, defenders must choose who to track → gaps appear
2. **Synergistic runs**: Di María's run pulls defenders → Mac Allister exploits the created gap → Combined effect > sum of parts
3. **Non-linear dynamics**: Pitch control model has sigmoid functions (1/(1+exp(-x))) → small coordinated changes can cause large PC shifts
4. **Tactical choreography**: Pre-planned attacking sequences where timing amplifies individual effects

**Chess analogy**: Moving your queen alone (individual) vs moving queen + rook + bishop simultaneously (coordinated attack) → opponent overwhelmed, interaction effect dominates.

### Heatmaps

Each PNG shows spatial distribution of a player's influence:

- **Red zones**: Where player's movement increased own team's pitch control
- **Blue zones**: Where player's movement decreased own team's pitch control (gave space to opponents)
- **Color intensity**: Magnitude of ΔPC at each location
- **Symmetric colormap**: Centered at zero, equal scaling for gains/losses

**What to look for:**

- **Red streaks**: Path of attacking runs creating forward space
- **Blue areas behind player**: Space vacated by forward movement
- **Red clusters near penalty box**: Prime goal-scoring opportunities created
- **Spatial patterns**: Where on the pitch each player operates and influences

---

## Technical Implementation Details

### Optimization Strategies

**Problem**: Calculating pitch control for every player at every frame is computationally expensive.

**Solutions implemented:**

1. **Temporal sampling**: 5 FPS (not every frame)
   - 61 frames over 12 seconds → 60 transitions
   - Reduces calculations from ~2800 to ~600

2. **Spatial filtering**: Only attacking-half players (x > 0)
   - ~8 players per frame instead of ~11
   - Focuses on offensive contributors

3. **Frame backfilling**: Use forward-fill for NaN positions
   - Handles temporary occlusions in tracking data
   - Prevents calculation failures

### Pitch Control Model

Uses **Spearman's (2018) "Beyond Expected Goals"** framework:

```python
P(t) = 1 / (1 + exp(-π/√3 * (T_arrival - TTI) / σ))
```

Where:
- `T_arrival`: Time for ball to reach grid location
- `TTI`: Player's time to intercept = reaction_time + distance/max_speed
- `σ`: Uncertainty parameter

**Key parameters** (from `Metrica_PitchControl.py`):
- `max_player_speed = 10 m/s` (boosted for attacking analysis)
- `kappa_def = 0.7` (reduced defender advantage)
- `reaction_time_def = 1.0s` (delayed defender reactions)
- `lambda_gk = 1.5×` (reduced goalkeeper influence range)

### Data Pipeline

```
PFF JSON tracking data
    ↓
Metrica CSV format (via PFF_to_Metrica_Adapter.py)
    ↓
Calculate velocities (Metrica_Velocities.py)
    ↓
Identify goal frame and time window
    ↓
For each frame transition:
    ├─ Calculate baseline PC (all at t)
    ├─ Calculate actual PC (all at t+1)
    ├─ For each attacking player:
    │   └─ Calculate PC with only that player moved
    └─ Calculate interaction term
    ↓
Aggregate statistics and generate heatmaps
```

---

## How to Use

### Basic Usage

```bash
python Scripts\generate_player_influence_analysis.py
```

### Modifying Analysis Parameters

Edit the script to change:

**Time window**:
```python
time_before = 6.0  # seconds before goal
time_after = 6.0   # seconds after goal
```

**Frame rate**:
```python
fps_target = 5  # frames per second (lower = faster, higher = more detailed)
```

**Spatial filter**:
```python
attacking_half_threshold = 0  # x-coordinate (0 = halfway line)
```

**Goal selection**:
```python
dimaria_goal = goals.iloc[1]  # 0-indexed: 0=first goal, 1=second, etc.
```

### Analyzing Different Sequences

To analyze a different goal or passage of play:

1. **Find the event**: Check `events` DataFrame for the frame number
2. **Update goal selection**: Change `goals.iloc[index]`
3. **Adjust time window**: Modify `time_before` and `time_after`
4. **Run script**: Results will overwrite previous output

### Loading Saved Results

```python
import numpy as np

# Load the NPZ file
data = np.load('Metrica_Output/player_influence_results.npz', allow_pickle=True)

# Access player names
player_names = data['player_name_map'].item()

# Access rankings
sorted_players = data['sorted_players']

# Access frame-by-frame data
influence_results = data['influence_results']

# Example: Get Di María's influence in frame 10
frame_10 = influence_results[10]
dimaria_influence = frame_10['player_influences']['Home_11']
print(f"Frame 10 - Di María total influence: {dimaria_influence['total_influence']:.3f}")
```

---

## Theoretical Background

### Why Sequential Perturbation?

Other approaches and their limitations:

1. **Gradient-based methods** (∂PC/∂x_i, ∂PC/∂y_i):
   - Problem: Gradients show *where* to move, not *how much* space was created
   - Requires numerical differentiation (approximation errors)
   - Doesn't capture non-linear interactions

2. **Functional derivatives** (δPC/δθ_i):
   - Problem: Complex calculus, assumes smooth differentiability
   - Hard to interpret physically
   - Still doesn't separate individual vs interaction effects

3. **Shapley values** (game theory):
   - Problem: Requires 2^N calculations (exponential in player count)
   - Computationally prohibitive for real-time analysis
   - Assumes coalition structures that may not apply to continuous space

**Sequential perturbation advantages**:
- ✅ Direct measurement, no approximations
- ✅ Linear complexity: O(N) calculations per frame
- ✅ Clean separation: individual effects + interaction term
- ✅ Intuitive interpretation: "freeze all but one, measure change"

### Connection to Partial Differential Equations

The pitch control surface PC(x, y, t) can be viewed as solving a time-dependent PDE:

```
∂PC/∂t = F(PC, player_positions(t), player_velocities(t))
```

Where:
- `PC(x,y,t)`: Pitch control at location (x,y) and time t
- `F`: Non-linear function from Spearman model

**Sequential perturbation** effectively measures:
```
∂PC/∂p_i ≈ [PC(p_i at t+1) - PC(p_i at t)] / Δt
```

Where `p_i` is player i's position vector. This is a **finite difference approximation** of the partial derivative with respect to player position.

The **interaction term** captures higher-order derivatives:
```
Interaction ≈ ∑∑ ∂²PC/∂p_i∂p_j * Δp_i * Δp_j  (cross-terms)
```

When interaction is high (84%), these cross-partial derivatives dominate → player movements are strongly coupled.

---

## Limitations and Future Work

### Current Limitations

1. **Computational cost**: ~1-2 minutes for 60 frame transitions
   - Could parallelize player calculations (independent)
   - Could use GPU-accelerated PC calculations

2. **Only attacking half**: Misses defensive players' contributions
   - Easy to enable by removing `attacking_half_threshold` filter
   - Would double computation time

3. **Linear decomposition**: Interaction term is "everything else" (black box)
   - Doesn't identify *which* player pairs have strongest synergy
   - Could extend to pairwise interaction matrix

4. **Single time window**: Analyzes one goal sequence
   - Could batch-process multiple goals for comparison
   - Could create movies showing influence evolution

### Potential Extensions

**Pairwise Synergy Analysis**:
```python
# Calculate interaction between players i and j
PC_i_and_j = PitchControl(only_i_and_j_move)
synergy_ij = PC_i_and_j - PC_baseline - ΔPC_i - ΔPC_j
```

**Temporal Influence Movies**:
- Create video showing influence heatmaps evolving over time
- Identify "key moments" where influence spikes
- Overlay with actual player positions and ball location

**Defensive Analysis**:
- Run same method on defending team
- Find which defenders best prevented space creation
- Compare attacking influence vs defensive containment

**Optimal Movement Gradients**:
```python
gradient_x = ∂(ΔPC)/∂x_i  # Which direction should player move?
gradient_y = ∂(ΔPC)/∂y_i  # To maximize space creation
```

**Multi-Goal Comparison**:
- Compare interaction % across different goals
- Identify tactical patterns (high interaction = coordinated attacks)
- Cluster goals by influence signatures

---

## Example Interpretation: Di María's Goal

### The Numbers

```
Ángel Di María:      824.1 total (+437.4 / -386.7)
Alexis Mac Allister: 372.9 total (+283.8 / -89.0)
Julián Álvarez:      202.8 total (+113.9 / -88.9)
Lionel Messi:         87.9 total (+43.1 / -44.8)

Interaction: 84.0% (2730.1 / 3249.4)
```

## References

- **Spearman, W.** (2018). "Beyond Expected Goals." *MIT Sloan Sports Analytics Conference*.
- **Fernández, J. & Bornn, L.** (2018). "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer." *MIT Sloan Sports Analytics Conference*.
- **Repository**: [github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)


