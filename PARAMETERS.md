# Pitch Control Model Parameters

This document explains all parameters used in the pitch control model (`Metrica_PitchControl.py`) and how they affect the calculations.

## Overview

The pitch control model calculates the probability that each team would win possession if the ball arrived at any given location on the field. It's based on William Spearman's 2018 paper ["Beyond Expected Goals"](http://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf).

---

## Player Movement Parameters

### `max_player_speed`
- **Current Value:** `10.0 m/s` (36 km/h)
- **Description:** Maximum sprint speed players can achieve
- **Effect on Pitch Control:**
  - **Higher values** → Larger control zones (players can reach more space)
  - **Lower values** → Smaller control zones (players cover less ground)
- **Realistic Values:**
  - Elite speedsters (Mbappé, Haaland): 10 m/s (36 km/h)
  - Fast players: 9 m/s (32 km/h)
  - Average professionals: 8 m/s (29 km/h)
  - Youth/amateur: 7 m/s (25 km/h)
- **Notes:** Set to 10 m/s for World Cup Final elite-level analysis

### `reaction_time`
- **Current Value:** `0.7 seconds`
- **Description:** Time for **attacking players** to react and change trajectory
- **Effect on Pitch Control:**
  - **Higher values** → Slower reactions, smaller attacking control zones
  - **Lower values** → Faster reactions, larger attacking control zones
- **How It Works:** Players continue at current velocity for this duration before sprinting to target
- **Realistic Values:**
  - Elite reactions: 0.5-0.6 s
  - Average professional: 0.7 s
  - Slower reactions: 0.8-1.0 s

### `reaction_time_def`
- **Current Value:** `1.0 seconds`
- **Description:** Time for **defending players** to react and change trajectory
- **Effect on Pitch Control:**
  - **Higher than `reaction_time`** → Defenders disadvantaged (more realistic)
  - **Equal to `reaction_time`** → Neutral (equal reactions)
  - **Lower than `reaction_time`** → Defenders advantaged (unrealistic)
- **Why Slower:** Defenders must react to unpredictable attacking movements
- **Notes:** Set 0.3s slower than attackers (1.0s vs 0.7s) to reflect defensive disadvantage

### `max_player_accel`
- **Current Value:** `7.0 m/s²`
- **Description:** Maximum player acceleration
- **Effect on Pitch Control:** **Not currently used** in the model
- **Notes:** Included for potential future enhancements

### `tti_sigma`
- **Current Value:** `0.45 seconds`
- **Description:** Standard deviation of the sigmoid function that determines uncertainty in player arrival time
- **Effect on Pitch Control:**
  - **Higher values** → More gradual transition between control zones (more uncertainty)
  - **Lower values** → Sharper boundaries between control zones (more certainty)
- **How It Works:** Controls the steepness of the probability curve: `P(T) = 1/(1 + exp(-π/√3/σ * (T - TTI)))`
- **Notes:** From Spearman 2018, represents inherent uncertainty in player movements

---

## Ball Control Parameters

### `lambda_att`
- **Current Value:** `4.3`
- **Description:** Ball control rate for **attacking team** players
- **Effect on Pitch Control:**
  - **Higher values** → Attackers gain control faster once they arrive
  - **Lower values** → Attackers slower to secure possession
- **How It Works:** Determines how quickly probability increases after player arrival
- **Notes:** From Spearman 2018 paper

### `lambda_def`
- **Current Value:** `3.01` (calculated as `4.3 × kappa_def`)
- **Description:** Ball control rate for **defending team** players
- **Effect on Pitch Control:**
  - **Higher values** → Defenders gain control faster
  - **Lower values** → Defenders slower to secure possession
- **Notes:** Currently lower than `lambda_att` due to `kappa_def = 0.7`

### `kappa_def`
- **Current Value:** `0.7`
- **Description:** Defensive advantage/disadvantage multiplier
- **Effect on Pitch Control:**
  - **> 1.0** → Defenders advantaged (control ball faster than attackers)
  - **= 1.0** → Neutral (equal ball control rates)
  - **< 1.0** → Attackers advantaged (defenders slower to control ball)
- **Realistic Values:**
  - Spearman's original: 1.72 (strong defensive advantage)
  - Neutral: 1.0
  - Current setting: 0.7 (attacking advantage)
- **Why < 1.0:** 
  - Attackers know where ball is going
  - Attackers have momentum toward ball
  - Defenders reacting to unpredictable movements

### `lambda_gk`
- **Current Value:** `4.52` (calculated as `lambda_def × 1.5`)
- **Description:** Ball control rate for **goalkeepers**
- **Effect on Pitch Control:**
  - Goalkeepers control space near them faster than outfield players, but not overwhelmingly
  - Reflects ability to catch/claim the ball with minimal advantage
- **Notes:** 1.5× the defending player rate (reduced from 3×), minimal GK advantage - more contested penalty area

### `average_ball_speed`
- **Current Value:** `15.0 m/s` (54 km/h)
- **Description:** Average speed of passes/ball movement
- **Effect on Pitch Control:**
  - **Higher values** → Ball arrives faster, reduces travel time advantage
  - **Lower values** → Ball slower, increases importance of positioning
- **How It Works:** `ball_travel_time = distance / average_ball_speed`
- **Realistic Values:**
  - Slow pass: 10-12 m/s
  - Average pass: 15 m/s (current)
  - Hard pass/shot: 20-25 m/s
- **Notes:** Constant across all passes (simplification)

---

## Numerical Parameters

These parameters control the mathematical computation of the model.

### `int_dt`
- **Current Value:** `0.04 seconds`
- **Description:** Integration timestep for probability calculations
- **Effect on Pitch Control:**
  - **Smaller values** → More accurate but slower computation
  - **Larger values** → Faster computation but less accurate
- **Notes:** 0.04s = 25 FPS, matches typical tracking data frame rate

### `max_int_time`
- **Current Value:** `10 seconds`
- **Description:** Maximum time to integrate equation (computational limit)
- **Effect on Pitch Control:**
  - **Higher values** → More complete calculations for long-distance scenarios
  - **Lower values** → Faster computation, may miss distant player contributions
- **Notes:** 10 seconds is typically sufficient for most on-field scenarios

### `model_converge_tol`
- **Current Value:** `0.01` (1%)
- **Description:** Stop integration when total probability exceeds this threshold
- **Effect on Pitch Control:**
  - **Lower values** → More precise but slower (e.g., 0.99 → 0.999)
  - **Higher values** → Faster but less precise
- **How It Works:** Integration stops when `PPCF_att + PPCF_def > (1 - tolerance)`
- **Notes:** 0.01 means stop when total probability > 99%

### `time_to_control_veto`
- **Current Value:** `3` (represents 10⁻³ = 0.001 probability threshold)
- **Description:** Ignore players with very low control probability
- **Effect on Pitch Control:**
  - **Higher values** → Stricter filtering, faster computation
  - **Lower values** → Include more distant players, slower computation
- **How It Works:** Players with arrival time advantage < `time_to_control_veto × log(10) × ...` are filtered out
- **Notes:** Computational optimization - removes negligible contributions

### `time_to_control_att`
- **Current Value:** Calculated as `3 × log(10) × (√3 × tti_sigma / π + 1 / lambda_att)`
- **Description:** Minimum time advantage needed for attacking team to shortcut calculation
- **Effect on Pitch Control:**
  - If attackers arrive this much earlier than defenders → instant 100% attacking control
- **Notes:** Automatically calculated from other parameters

### `time_to_control_def`
- **Current Value:** Calculated as `3 × log(10) × (√3 × tti_sigma / π + 1 / lambda_def)`
- **Description:** Minimum time advantage needed for defending team to shortcut calculation
- **Effect on Pitch Control:**
  - If defenders arrive this much earlier than attackers → instant 100% defending control
- **Notes:** Automatically calculated from other parameters

---

## Summary of Current Configuration

| Parameter | Value | Effect |
|-----------|-------|--------|
| **Speed & Movement** |
| `max_player_speed` | 10 m/s | Elite World Cup level |
| `reaction_time` | 0.7 s | Attacking players reaction |
| `reaction_time_def` | 1.0 s | Defending players (slower) |
| `tti_sigma` | 0.45 s | Moderate uncertainty |
| **Ball Control** |
| `lambda_att` | 4.3 | Attacking control rate |
| `lambda_def` | 3.01 | Defending control rate (0.7× att) |
| `kappa_def` | 0.7 | **Attacking advantage** |
| `lambda_gk` | 4.52 | GK control rate (1.5× def) |
| `average_ball_speed` | 15 m/s | Typical pass speed |
| **Numerical** |
| `int_dt` | 0.04 s | 25 FPS integration |
| `max_int_time` | 10 s | Max calculation time |
| `model_converge_tol` | 0.01 | 99% convergence |
| `time_to_control_veto` | 3 | Filter at 0.1% probability |

---

## Current Model Behavior

With the current parameter configuration:

### Attacking Advantages
1. **Faster reactions** (0.7s vs 1.0s) → Attackers change direction 30% faster
2. **Better ball control** (λ=4.3 vs 3.01) → Attackers secure possession 43% faster
3. **Combined effect** → More red zones in pitch control maps

### What You'll See
- **Large red areas** → Attacking team (Argentina) controls most forward space
- **Blue zones near defenders** → Defending team (France) controls defensive positions
- **Purple/contested areas** → Transitional zones where both teams ~50% control
- **Bright colors** → High certainty (>80% control)
- **Faded colors** → Lower certainty (50-70% control)

### Realistic for:
- ✅ Elite-level football (World Cup Final)
- ✅ Attacking-minded analysis
- ✅ Fast, technical players (Mbappé, Messi, Di María)

---

## Modifying Parameters

To change parameters, edit `Metrica_PitchControl.py`, function `default_model_params()` (around line 249):

```python
def default_model_params(time_to_control_veto=3):
    params = {}
    params['max_player_speed'] = 10.  # Change this
    params['reaction_time'] = 0.7
    params['reaction_time_def'] = 1.0
    params['kappa_def'] = 0.7
    # ... etc
    return params
```

Then re-run your analysis scripts (e.g., `Tutorial3_PFF.py`) to see updated pitch control maps.

---

## References

- Spearman, W. (2018). "Beyond Expected Goals". MIT Sloan Sports Analytics Conference.
  - Paper: http://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf
- Laurie Shaw's implementation: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

---

**Last Updated:** October 31, 2025
