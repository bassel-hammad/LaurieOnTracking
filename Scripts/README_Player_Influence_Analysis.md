# Player Influence Analysis

## Overview

`generate_player_influence_analysis.py` quantifies individual player contributions to pitch control dynamics using a **sequential perturbation method**. This tool reveals which players created or destroyed space during specific match sequences by isolating each player's individual impact on pitch control changes.

---

## What It Does

### Core Functionality

The script analyzes a user-selected sequence of play and:

1. **Prompts for input**: Match ID and sequence number
2. **Automatically detects attacking team** from sequence events
3. **Isolates each player's contribution** to pitch control changes at event transitions
4. **Generates 4 ranking tables** by different metrics (total, positive, negative, net)
5. **Calculates interaction effects** showing team coordination vs individual impact
6. **Generates an animated video** with dynamic player rankings and stacked bar charts

### Output

- **Console output**: 
  - **Table 1**: Total Influence Rankings (magnitude of impact)
  - **Table 2**: Positive Influence Rankings (space creation)
  - **Table 3**: Negative Influence Rankings (space concession)
  - **Table 4**: Net Influence Rankings (net gain = positive + negative)
  - Interaction analysis statistics
  
- **Video file** (`sequence_X_pitch_control.mp4`):
  - Green pitch with white markings
  - Dynamic team colors (attacking team highlighted)
  - Player positions with ranking numbers (1-5) on top contributors
  - Vertical stacked bar chart showing all 11 players with rainbow-colored event segments
  - Player names below chart at 45° angle

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

**Net Influence**: Positive Influence + Negative Influence
- The algebraic sum showing net space impact
- Positive net = player created more space than they conceded
- Negative net = player conceded more space than they created
- Reveals whether a player was a net contributor or detractor to space control

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


