#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create presentation images for Angel Di Maria's goal.

The script generates four still images:
1. Baseline pitch control at time t.
2. Counterfactual pitch control after moving Di Maria to his t+1 position.
3. A position-change image showing Di Maria's movement.
4. A delta image showing the pitch-control change caused by that movement.

Default behavior targets match 10517 and the first Di Maria shot event.
"""

from __future__ import annotations

import argparse
import os
import sys
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Metrica_IO as mio
import Metrica_PitchControl as mpc
import Metrica_Velocities as mvel
import Metrica_Viz as mviz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATADIR = PROJECT_ROOT / "Sample Data"
OUTPUT_DIR = PROJECT_ROOT / "Metrica_Output" / "Presentation" / "Di_Maria_Goal"
TARGET_PLAYER_NAME = "Di Maria"
TARGET_PLAYER_ID = 3868


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate presentation images for Angel Di Maria's goal."
    )
    parser.add_argument("--match-id", type=int, default=10517, help="Match ID to analyze.")
    parser.add_argument(
        "--event-index",
        type=int,
        default=None,
        help="Exact event row index to use. If omitted, the first Di Maria shot is used.",
    )
    parser.add_argument(
        "--sequence",
        type=int,
        default=None,
        help="Sequence number to anchor the images on. Uses the first event in that sequence.",
    )
    parser.add_argument(
        "--player-id",
        dest="player_id",
        type=int,
        default=TARGET_PLAYER_ID,
        help="PFF playerId for Di Maria.",
    )
    parser.add_argument(
        "--player-name",
        type=str,
        default=TARGET_PLAYER_NAME,
        help="Name fragment used to find the event row.",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=1,
        help="How many tracking frames to move Di Maria forward for the counterfactual when no time offset is used.",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=None,
        help="How many seconds to move forward for the counterfactual snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory for the generated images.",
    )
    return parser.parse_args()


def load_match_data(match_id: int):
    print("Loading data...")
    events = mio.read_event_data(str(DATADIR), match_id)
    tracking_home = mio.tracking_data(str(DATADIR), match_id, "Home")
    tracking_away = mio.tracking_data(str(DATADIR), match_id, "Away")

    tracking_home = mio.to_metric_coordinates(tracking_home)
    tracking_away = mio.to_metric_coordinates(tracking_away)
    events = mio.to_metric_coordinates(events)
    tracking_home, tracking_away, events = mio.to_single_playing_direction(
        tracking_home, tracking_away, events
    )

    pff_speed_cols = [c for c in tracking_home.columns if c.endswith("_pff_speed")]
    if pff_speed_cols:
        print("Calculating player velocities using HYBRID method (PFF speed + calculated direction)...")
        tracking_home = mvel.calc_player_velocities_hybrid(
            tracking_home, smoothing=True, use_pff_speed=True
        )
        tracking_away = mvel.calc_player_velocities_hybrid(
            tracking_away, smoothing=True, use_pff_speed=True
        )
    else:
        print("Calculating player velocities from position differences...")
        tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
        tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

    print(f"Data loaded: {len(events)} events, {len(tracking_home):,} tracking frames")
    return events, tracking_home, tracking_away


def find_event(events: pd.DataFrame, event_index: int | None, player_name: str) -> int:
    if event_index is not None:
        if event_index not in events.index:
            raise ValueError(f"Event index {event_index} was not found in the match data.")
        return event_index

    fragments = []
    for token in re.split(r"[^A-Za-z0-9]+", player_name):
        token = token.strip()
        if len(token) >= 3:
            fragments.append(token[:3])
    if not fragments:
        fragments = [player_name]

    mask = events["Type"].astype(str).str.upper().eq("SHOT")
    from_series = events["From"].astype(str)
    fragment_mask = np.zeros(len(events), dtype=bool)
    for fragment in fragments:
        fragment_mask |= from_series.str.contains(fragment, case=False, na=False)
    mask &= fragment_mask
    candidates = events.loc[mask].copy()
    if candidates.empty:
        mask = events["From"].astype(str).str.contains(player_name, case=False, na=False)
        candidates = events.loc[mask].copy()

    if candidates.empty:
        raise ValueError(f"No event rows found for '{player_name}'.")

    print("Candidate events for Di Maria:")
    display_cols = [c for c in ["Sequence", "Team", "Type", "From", "Start Frame", "Start Time [s]"] if c in candidates.columns]
    print(candidates[display_cols].head(10).to_string())
    chosen_index = int(candidates.index[0])
    print(f"Using event row {chosen_index}.")
    return chosen_index


def find_sequence_event(events: pd.DataFrame, sequence: int) -> int:
    sequence_events = events.loc[events["Sequence"] == sequence].copy()
    if sequence_events.empty:
        raise ValueError(f"No event rows found for sequence {sequence}.")

    sequence_events = sequence_events.sort_values(["Start Frame", "Start Time [s]"], kind="mergesort")
    chosen_index = int(sequence_events.index[0])
    print(f"Using sequence {sequence}, event row {chosen_index}.")
    display_cols = [c for c in ["Sequence", "Team", "Type", "From", "Start Frame", "Start Time [s]"] if c in sequence_events.columns]
    print(sequence_events[display_cols].head(10).to_string())
    return chosen_index


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    return normalized.lower()


def filename_prefix_from_player(player_name: str) -> str:
    normalized = normalize_text(player_name)
    safe = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return safe or "selected_player"


def choose_player_from_home(tracking_home: pd.DataFrame) -> str:
    cols = [c for c in tracking_home.columns if c.startswith("Home_") and c.endswith("_x")]
    players = [c[len("Home_"):-2] for c in cols]
    if not players:
        raise ValueError("No Home players found in tracking data to choose from.")

    print("\nChoose a player from Argentina (Home team):")
    for i, p in enumerate(players, start=1):
        print(f"  {i:2d}: {p}")

    while True:
        choice = input(f"Enter number (1-{len(players)}): ").strip()
        if not choice:
            print("No selection made — defaulting to first player.")
            return players[0]
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(choice) - 1
        if 0 <= idx < len(players):
            return players[idx]
        print("Choice out of range — try again.")


def choose_player_interactive(tracking_home: pd.DataFrame, tracking_away: pd.DataFrame) -> str:
    home_cols = [c for c in tracking_home.columns if c.startswith("Home_") and c.endswith("_x")]
    away_cols = [c for c in tracking_away.columns if c.startswith("Away_") and c.endswith("_x")]
    home_players = [c[len("Home_"):-2] for c in home_cols]
    away_players = [c[len("Away_"):-2] for c in away_cols]

    choices = []
    print("\nChoose a player to highlight (Home = Argentina, Away = Opponent):")
    idx = 1
    for p in home_players:
        print(f"  {idx:2d}: Home - {p}")
        choices.append(("Home", p))
        idx += 1
    for p in away_players:
        print(f"  {idx:2d}: Away - {p}")
        choices.append(("Away", p))
        idx += 1

    if not choices:
        raise ValueError("No players found in tracking data to choose from.")

    while True:
        choice = input(f"Enter number (1-{len(choices)}): ").strip()
        if not choice:
            print("No selection made — defaulting to first player.")
            return choices[0][1]
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        sel = int(choice) - 1
        if 0 <= sel < len(choices):
            team, player = choices[sel]
            print(f"Selected: {team} - {player}")
            return player
        print("Choice out of range — try again.")


def resolve_player_column(tracking: pd.DataFrame, team: str, player_name: str) -> str:
    target = normalize_text(player_name)
    candidate_columns = [c for c in tracking.columns if c.startswith(f"{team}_") and c.endswith("_x")]

    exact_matches = [c for c in candidate_columns if target in normalize_text(c)]
    if exact_matches:
        return exact_matches[0][:-2]

    fragments = [frag for frag in re.split(r"[^A-Za-z0-9]+", target) if len(frag) >= 3]
    for column in candidate_columns:
        normalized_column = normalize_text(column)
        if all(fragment in normalized_column for fragment in fragments):
            return column[:-2]

    raise ValueError(f"Could not resolve '{player_name}' in {team} tracking columns.")


def find_player_team_and_column(tracking_home: pd.DataFrame, tracking_away: pd.DataFrame, player_name: str) -> tuple[str, str]:
    for team, tracking in (("Home", tracking_home), ("Away", tracking_away)):
        try:
            player_column = resolve_player_column(tracking, team, player_name)
            return team, player_column
        except ValueError:
            continue
    raise ValueError(f"Player '{player_name}' was not found in the tracking data.")


def get_next_frame(tracking: pd.DataFrame, frame: int, frame_offset: int) -> int:
    target_frame = frame + frame_offset
    if target_frame in tracking.index:
        return int(target_frame)

    later_frames = tracking.index[tracking.index > frame]
    if len(later_frames) == 0:
        raise ValueError(f"No tracking frame exists after {frame}.")
    return int(later_frames[min(frame_offset - 1, len(later_frames) - 1)])


def get_frame_at_time(tracking: pd.DataFrame, time_s: float) -> int:
    if "Time [s]" not in tracking.columns:
        raise ValueError("Tracking data does not contain a 'Time [s]' column.")
    times = tracking["Time [s]"].astype(float)
    nearest_idx = (times - time_s).abs().idxmin()
    return int(nearest_idx)


def choose_sequence_interactive(events: pd.DataFrame) -> int | None:
    seqs = sorted(events['Sequence'].dropna().unique())
    if not seqs:
        return None
    print('\nAvailable sequences (first few shown):')
    sample = []
    for s in seqs[:30]:
        subset = events[events['Sequence'] == s]
        first = subset.sort_values(['Start Frame', 'Start Time [s]']).iloc[0]
        sample.append((int(s), first['Type'], first['From'], int(first['Start Frame'])))
    for i, (s, t, frm, sf) in enumerate(sample, start=1):
        print(f"  {i:2d}: Sequence {s} — {t} by {frm} @ frame {sf}")

    print("  Press Enter to skip sequence selection and search by player name.")
    choice = input(f"Enter number (1-{len(sample)}) to choose sequence, or Enter to skip: ").strip()
    if not choice:
        return None
    if not choice.isdigit():
        print("Invalid selection; skipping sequence selection.")
        return None
    idx = int(choice) - 1
    if 0 <= idx < len(sample):
        return sample[idx][0]
    print("Choice out of range; skipping sequence selection.")
    return None


def copy_player_state(source: pd.DataFrame, target: pd.DataFrame, frame: int, replacement_frame: int, player_column: str) -> pd.DataFrame:
    result = target.copy()
    source_row = source.loc[replacement_frame]

    if isinstance(source_row, pd.DataFrame):
        source_row = source_row.iloc[0]

    # Only move the player's position/visibility; leave other tracked values at the sequence-start frame.
    columns_to_copy = [f"{player_column}_x", f"{player_column}_y", f"{player_column}_visibility"]

    for column in columns_to_copy:
        if column in result.columns and column in source_row.index:
            result.loc[frame, column] = source_row[column]

    return result


def get_player_position(tracking: pd.DataFrame, frame: int, player_column: str) -> tuple[float, float]:
    row = tracking.loc[frame]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return float(row[f"{player_column}_x"]), float(row[f"{player_column}_y"])


def get_ball_position(tracking: pd.DataFrame, frame: int) -> tuple[float, float]:
    row = tracking.loc[frame]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return float(row["ball_x"]), float(row["ball_y"])


def generate_pitch_control(events, tracking_home, tracking_away, event_index):
    params = mpc.default_model_params()
    gk_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
    ppcf, xgrid, ygrid = mpc.generate_pitch_control_for_event(
        event_index, events, tracking_home, tracking_away, params, gk_numbers
    )
    attacking_team = events.loc[event_index, "Team"]
    return ppcf, xgrid, ygrid, attacking_team


def surface_cmap(attacking_team: str) -> str:
    return "bwr" if attacking_team == "Home" else "bwr_r"


def plot_surface(ax, surface: np.ndarray, attacking_team: str, alpha: float = 0.55):
    ax.imshow(
        surface,
        extent=(-53, 53, -34, 34),
        interpolation="spline36",
        vmin=0.0,
        vmax=1.0,
        cmap=surface_cmap(attacking_team),
        alpha=alpha,
        origin="lower",
    )


def highlight_player(ax, row: pd.Series, player_column: str, color: str, label: str | None = None):
    x = row[f"{player_column}_x"]
    y = row[f"{player_column}_y"]
    ax.scatter([x], [y], s=160, facecolors="none", edgecolors=color, linewidths=2.5, zorder=20)
    ax.scatter([x], [y], s=45, color=color, zorder=21)
    if label:
        ax.text(x + 1.0, y + 1.0, label, fontsize=11, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9))


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("DI MARIA GOAL PRESENTATION IMAGE GENERATOR")
    print("=" * 70)
    print()

    events, tracking_home, tracking_away = load_match_data(args.match_id)

    # Require a sequence: prompt until a valid sequence number is provided (either via CLI or interactively).
    if args.sequence is None:
        available_seqs = sorted([int(s) for s in events['Sequence'].dropna().unique()])
        print("\nAvailable sequences (sample):", available_seqs[:20])
        while args.sequence is None:
            resp = input("Enter sequence number to anchor on: ").strip()
            if not resp:
                print("Please enter a sequence number.")
                continue
            try:
                seq_val = int(resp)
            except ValueError:
                print("Invalid number — try again.")
                continue
            if seq_val not in available_seqs:
                print("Sequence not found in this match — try another.")
                continue
            args.sequence = seq_val
            print(f"Using sequence {args.sequence}")

    # Prompt for player if not provided.
    if args.player_name == TARGET_PLAYER_NAME or not args.player_name:
        try:
            chosen = choose_player_interactive(tracking_home, tracking_away)
            print(f"Selected player: {chosen}")
            args.player_name = chosen
        except Exception as e:
            print(f"Warning: could not prompt for player selection: {e}")

    # If no time offset provided, prompt the user for one.
    if args.time_offset is None:
        try:
            resp = input("Enter time offset in seconds for counterfactual (default 3.0): ").strip()
            args.time_offset = float(resp) if resp else 3.0
            print(f"Using time offset: {args.time_offset} s")
        except Exception:
            print("Invalid input — defaulting to 3.0s")
            args.time_offset = 3.0
    event_index = find_sequence_event(events, args.sequence)

    event_row = events.loc[event_index]
    if isinstance(event_row, pd.DataFrame):
        event_row = event_row.iloc[0]

    base_frame = int(event_row["Start Frame"])
    event_start_time = float(event_row["Start Time [s]"])
    player_team, player_column_base = find_player_team_and_column(tracking_home, tracking_away, args.player_name)
    filename_prefix = filename_prefix_from_player(args.player_name)
    player_tracking = tracking_home if player_team == "Home" else tracking_away
    ball_x, ball_y = get_ball_position(tracking_home, base_frame)
    if args.time_offset is not None:
        base_tracking_time = player_tracking.loc[base_frame, "Time [s]"]
        if isinstance(base_tracking_time, pd.Series):
            base_tracking_time = float(base_tracking_time.iloc[0])
        else:
            base_tracking_time = float(base_tracking_time)
        replacement_time = base_tracking_time + float(args.time_offset)
        replacement_frame = get_frame_at_time(player_tracking, replacement_time)
    else:
        replacement_time = None
        base_tracking_time = None
        replacement_frame = get_next_frame(player_tracking, base_frame, args.frame_offset)
    player_column_counterfactual = resolve_player_column(player_tracking, player_team, args.player_name)

    print()
    print(f"Selected sequence anchor row: {event_index}")
    print(f"Event details: {event_row['Type']} by {event_row['From']} for {event_row['Team']}")
    print(f"Sequence start frame: {base_frame}")
    print(f"Event start time [events]: {event_start_time:.3f}s")
    print(f"Ball position used for PPCF: ({ball_x:.3f}, {ball_y:.3f})")
    if replacement_time is not None:
        print(f"Sequence start time [tracking@frame]: {base_tracking_time:.3f}s")
        print(f"Counterfactual time: {replacement_time:.3f}s")
        print(f"Counterfactual frame: {replacement_frame} (offset {args.time_offset:.1f}s)")
    else:
        print(f"Counterfactual frame: {replacement_frame} (offset {args.frame_offset})")
    print(f"Di Maria tracking column: {player_column_base}")
    start_x, start_y = get_player_position(player_tracking, base_frame, player_column_base)
    end_x, end_y = get_player_position(player_tracking, replacement_frame, player_column_base)
    print(f"Chosen player start position: ({start_x:.3f}, {start_y:.3f})")
    print(f"Chosen player offset position: ({end_x:.3f}, {end_y:.3f})")

    if player_team == "Home":
        counterfactual_home = copy_player_state(tracking_home, tracking_home, base_frame, replacement_frame, player_column_base)
        counterfactual_away = tracking_away.copy()
    else:
        counterfactual_home = tracking_home.copy()
        counterfactual_away = copy_player_state(tracking_away, tracking_away, base_frame, replacement_frame, player_column_base)

    events_for_ppcf = events.copy()
    events_for_ppcf.loc[event_index, "Start Frame"] = base_frame
    ball_x, ball_y = get_ball_position(tracking_home, base_frame)
    events_for_ppcf.loc[event_index, "Start X"] = ball_x
    events_for_ppcf.loc[event_index, "Start Y"] = ball_y
    baseline_ppcf, _, _, attacking_team = generate_pitch_control(events_for_ppcf, tracking_home, tracking_away, event_index)
    counterfactual_ppcf, _, _, _ = generate_pitch_control(events_for_ppcf, counterfactual_home, counterfactual_away, event_index)
    delta_ppcf = counterfactual_ppcf - baseline_ppcf

    base_home_row = tracking_home.loc[base_frame]
    base_away_row = tracking_away.loc[base_frame]
    cf_home_row = counterfactual_home.loc[base_frame]
    cf_away_row = counterfactual_away.loc[base_frame]

    if isinstance(base_home_row, pd.DataFrame):
        base_home_row = base_home_row.iloc[0]
    if isinstance(base_away_row, pd.DataFrame):
        base_away_row = base_away_row.iloc[0]
    if isinstance(cf_home_row, pd.DataFrame):
        cf_home_row = cf_home_row.iloc[0]
    if isinstance(cf_away_row, pd.DataFrame):
        cf_away_row = cf_away_row.iloc[0]

    team_row = base_home_row if player_team == "Home" else base_away_row
    counterfactual_team_row = cf_home_row if player_team == "Home" else cf_away_row
    player_color = "gold"

    goal_label = f"Di Maria ({args.player_id})"
    frame_time = float(base_tracking_time) if base_tracking_time is not None else float(event_start_time)

    baseline_path = output_dir / f"{filename_prefix}_baseline_pitch_control.png"
    counterfactual_path = output_dir / f"{filename_prefix}_counterfactual_pitch_control.png"
    position_change_path = output_dir / f"{filename_prefix}_position_change.png"
    delta_path = output_dir / f"{filename_prefix}_pitch_control_delta.png"

    # Baseline surface.
    fig, ax = mviz.plot_pitch(field_color="white")
    plot_surface(ax, baseline_ppcf, attacking_team, alpha=0.60)
    mviz.plot_frame(base_home_row, base_away_row, figax=(fig, ax), include_player_velocities=True, PlayerAlpha=0.55)
    highlight_player(ax, team_row, player_column_base, player_color, label=goal_label)
    fig.suptitle(f"Baseline pitch control at t = {frame_time:.1f}s", fontsize=16, y=0.98)
    save_figure(fig, baseline_path)

    # Counterfactual surface.
    fig, ax = mviz.plot_pitch(field_color="white")
    plot_surface(ax, counterfactual_ppcf, attacking_team, alpha=0.60)
    mviz.plot_frame(cf_home_row, cf_away_row, figax=(fig, ax), include_player_velocities=True, PlayerAlpha=0.55)
    counterfactual_label = f"Di Maria moved to t+{args.time_offset:.1f}s" if args.time_offset is not None else f"Di Maria moved to t+{args.frame_offset}"
    highlight_player(ax, counterfactual_team_row, player_column_counterfactual, player_color, label=counterfactual_label)
    fig.suptitle(f"Counterfactual pitch control with Di Maria moved forward", fontsize=16, y=0.98)
    save_figure(fig, counterfactual_path)

    # Position change image.
    fig, ax = mviz.plot_pitch(field_color="white")
    mviz.plot_frame(base_home_row, base_away_row, figax=(fig, ax), include_player_velocities=False, PlayerAlpha=0.35)
    start_x = team_row[f"{player_column_base}_x"]
    start_y = team_row[f"{player_column_base}_y"]
    end_x = counterfactual_team_row[f"{player_column_counterfactual}_x"]
    end_y = counterfactual_team_row[f"{player_column_counterfactual}_y"]
    ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops=dict(arrowstyle="->", color="black", lw=2.5),
        zorder=25,
    )
    ax.scatter([start_x], [start_y], s=180, facecolors="none", edgecolors="black", linewidths=2, zorder=26)
    ax.scatter([end_x], [end_y], s=180, facecolors="none", edgecolors="goldenrod", linewidths=2, zorder=26)
    ax.scatter([start_x], [start_y], s=55, color="black", zorder=27)
    ax.scatter([end_x], [end_y], s=55, color="goldenrod", zorder=27)
    ax.text(start_x + 1.0, start_y + 1.0, "t", fontsize=11, fontweight="bold", color="black")
    end_label = f"t+{args.time_offset:.1f}s" if args.time_offset is not None else f"t+{args.frame_offset}"
    ax.text(end_x + 1.0, end_y + 1.0, end_label, fontsize=11, fontweight="bold", color="goldenrod")
    fig.suptitle("Di Maria position change", fontsize=16, y=0.98)
    save_figure(fig, position_change_path)

    # Pitch control delta image.
    fig, ax = mviz.plot_pitch(field_color="white")
    vmax = float(np.nanmax(np.abs(delta_ppcf))) if np.isfinite(delta_ppcf).any() else 0.1
    vmax = max(vmax, 0.05)
    ax.imshow(
        delta_ppcf,
        extent=(-53, 53, -34, 34),
        interpolation="spline36",
        vmin=-vmax,
        vmax=vmax,
        cmap="RdBu_r",
        alpha=0.75,
        origin="lower",
    )
    mviz.plot_frame(base_home_row, base_away_row, figax=(fig, ax), include_player_velocities=False, PlayerAlpha=0.35)
    highlight_player(ax, team_row, player_column_base, player_color, label="movement impact")
    fig.suptitle("Change in pitch control from the counterfactual", fontsize=16, y=0.98)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax)),
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        shrink=0.8,
    )
    cbar.set_label("Counterfactual minus baseline", fontsize=10)
    save_figure(fig, delta_path)

    print()
    print("Generated images:")
    print(f"  {baseline_path}")
    print(f"  {counterfactual_path}")
    print(f"  {position_change_path}")
    print(f"  {delta_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()