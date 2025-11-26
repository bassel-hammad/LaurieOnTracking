# Player Influence Analysis

## Overview

`generate_player_influence_analysis.py` quantifies individual player contributions to pitch control dynamics using a **sequential perturbation method**. This tool reveals which players created or destroyed space during specific match sequences by isolating each player's individual impact on pitch control changes.

---

## What It Does

### Core Functionality

The script analyzes a user-selected sequence of play and:

1. **Prompts for input**:
   - Match ID (e.g., 10517)
   - Sequence number (from event data)
   
2. **Isolates each player's contribution** to pitch control changes at event transitions
3. **Ranks all 11 players** by their total influence on space control
4. **Calculates interaction effects** showing team coordination vs individual impact
5. **Generates an animated video** showing pitch control evolution with player rankings overlaid

### Output

- **Console output**: 
  - Complete rankings table for all 11 home team players
  - Influence metrics (total, positive, negative)
  - Interaction analysis statistics
  - Sequence event details
  
- **Video file** (`sequence_X_pitch_control.mp4`):
  - Animated pitch control surface (red = home control, blue = away control)
  - Player positions with ranking numbers (1-5) on top contributors
  - Bar chart showing top 5 most influential players
  - Plays at 5 FPS matching game time duration

---

## How It Works: The Sequential Perturbation Method

### The Core Idea

**Question**: When the pitch control changes during an event (e.g., a pass), how much did *each individual player* contribute to that change?

**Method**: Freeze all players at event time `t`, then move *only one player* to their position at event time `t+1`. Measure the change. This isolates that player's individual effect.

### Mathematical Framework

For each event transition (event `i` → event `i+1`):

1. **Baseline**: Calculate pitch control with all players at event `i`
   ```
   PC_baseline = PitchControl(all_players_at_event_i)
   ```

2. **Actual**: Calculate pitch control with all players at event `i+1`
   ```
   PC_actual = PitchControl(all_players_at_event_i+1)
   ΔPC_actual = PC_actual - PC_baseline
   ```

3. **Individual Contributions**: For each player `k`:
   ```
   # Move ONLY player k to event i+1, keep all others at event i
   PC_only_k_moved = PitchControl(player_k_at_event_i+1, others_at_event_i)
   
   # Player k's isolated contribution
   ΔPC_player_k = PC_only_k_moved - PC_baseline
   ```

4. **Aggregate across all events** in the sequence:
   ```
   Total_Influence_k = Σ |ΔPC_player_k| across all event transitions
   ```

### Key Metrics

**Total Influence**: Sum of absolute pitch control changes caused by a player's movements
- Higher values = player had more impact on space control
- Captures both space creation (positive ΔPC) and space destruction (negative ΔPC)

**Positive Influence**: Sum of pitch control increases
- Space the player created for their team

**Negative Influence**: Sum of pitch control decreases  
- Space the player gave away or failed to control

**Interaction Term**: Difference between actual change and sum of individual contributions
```
Interaction = ΔPC_actual - Σ(ΔPC_player_k)
```
- Low interaction (<10%) = changes explained by individual movements
- High interaction (>20%) = significant coordination effects between players

### Why This Method?

**Advantages over alternatives:**

- ✅ **Exact causal attribution**: Directly measures each player's impact by isolation
- ✅ **Simple interpretation**: "If only this player moved, here's what changed"
- ✅ **Reveals synergies**: Interaction term quantifies coordinated movement effects
- ✅ **No assumptions** about influence functions beyond the pitch control model itself
- ✅ **Event-based**: Analyzes actual game events rather than arbitrary time intervals

**Conceptual analogy**: Like A/B testing - change one variable at a time, measure the effect, isolate causality.

### Analysis Granularity

The script analyzes transitions between consecutive event frames:
- If a sequence has 10 events, it analyzes 9 transitions
- Each transition shows how pitch control evolved during that specific play
- Aggregating across all transitions reveals cumulative player impact throughout the sequence

---

## Interpreting The Results

### Player Rankings

The console output shows all 11 players ranked by influence:

```
Rank   Player                    Total        Positive     Negative     Frames
--------------------------------------------------------------------------------
1      Ángel Di María           824.083      437.365     -386.718         7
2      Alexis Mac Allister      372.875      283.834      -89.041         8
3      Julián Álvarez           202.814      113.897      -88.916         9
4      Lionel Messi              87.875       43.061      -44.814         7
...
11     Emiliano Martínez          2.145        1.234       -0.911         9
```

#### What Each Column Means:

- **Rank**: Position in influence hierarchy (1 = most influential)

- **Total**: Sum of absolute pitch control changes `Σ|ΔPC|` across all grid cells and events
  - **High value** = player's movements caused dramatic spatial changes
  - This is the **magnitude** of influence regardless of direction
  - Attackers making runs typically have high values

- **Positive**: Total pitch control gained for own team `Σ(ΔPC where ΔPC > 0)`
  - Represents space creation and offensive contribution
  - Shows areas where the player increased team's territorial dominance

- **Negative**: Total pitch control lost `Σ(ΔPC where ΔPC < 0)`  
  - Represents space conceded or defensive vulnerability
  - Can indicate tactical withdrawal or pressing failures

- **Frames**: Number of event transitions where player was present
  - Players appearing in more frames have more opportunities to influence
  - Useful for normalizing influence (influence per frame)
  - Players deeper/wider appear in fewer frames


#### What This Means:

**Interaction % = |Actual ΔPC - Sum of Individual ΔPC| / Actual ΔPC × 100**

- **Low interaction (<10%)**: Changes fully explained by individual movements (linear approximation valid)
- **Medium interaction (10-20%)**: Some nonlinear coordination effects
- **High interaction (>20%)**: Significant player coupling and coordinated effects

#### Interpretation:

**High interaction** indicates:
1. **Defensive overload**: Multiple attackers moving forces defenders to choose who to track → gaps appear
2. **Synergistic runs**: One player's movement creates space that another exploits
3. **Non-linear dynamics**: Pitch control model's sigmoid functions amplify coordinated movements
4. **Tactical choreography**: Pre-planned sequences where timing multiplies individual effects

**Low interaction** suggests:
- Players moving independently
- Changes predictable from individual actions
- Less tactical coordination in the sequence

### Video Output

The generated MP4 file shows:

**Top Panel - Pitch View**:
- **Red/blue surface**: Pitch control probability (red = home team controls, blue = away team)
- **Red dots**: Home team player positions
- **Blue dots**: Away team player positions  
- **Black dot**: Ball position
- **White numbers (1-5)**: Ranking labels on top 5 most influential players

**Bottom Panel - Bar Chart**:
- Horizontal bars showing total influence for top 5 players
- Ranked from most to least influential
- Values show magnitude of pitch control changes caused

---

## Technical Implementation Details

### Frame Synchronization

The script uses **frame-based analysis** rather than time-based to ensure perfect synchronization:

1. **Event frames**: Extracts Start Frame and End Frame from event data
2. **Tracking alignment**: Uses these exact frame numbers to query tracking data
3. **Event transitions**: Analyzes pitch control changes between consecutive event frames
4. **Video generation**: Samples frames uniformly at 5 FPS within the frame range

This ensures the analysis and video correspond to the actual game footage.

### Computational Approach

**For each event transition**:
```
Baseline PC calculation: 1× full pitch control (50×32 grid)
Actual PC calculation: 1× full pitch control
Individual attributions: 11× full pitch control (one per player)
Total: 13 pitch control calculations per transition
```

**Optimization**: Analyzes only event transitions (typically 5-10) rather than every tracking frame (hundreds)

## How to Use

### Running the Script

```bash
python Scripts/generate_player_influence_analysis.py
```

### Interactive Prompts

1. **Enter match ID**: e.g., `10517` (World Cup Final 2022)
2. **Select sequence**: Choose from available sequences shown
   - Sequences are possession chains or tactical phases
   - Each sequence contains multiple related events

### Example Session

```
Enter match ID (e.g., 10517): 10517
Available sequences: [1.0, 2.0, 3.0, 4.0, ...]
Enter sequence number (e.g., 1): 45

Analyzing sequence 45.0
  Events in sequence: 8
  Frame range: 15234 to 15567
  Time window: 508.3s to 520.8s
```
## References

- **Spearman, W.** (2018). "Beyond Expected Goals." *MIT Sloan Sports Analytics Conference*.
- **Fernández, J. & Bornn, L.** (2018). "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer." *MIT Sloan Sports Analytics Conference*.
- **Repository**: [github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)


