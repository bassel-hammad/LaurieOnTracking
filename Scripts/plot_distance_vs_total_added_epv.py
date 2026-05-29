#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance vs Total Added EPV Correlation (Whole Match)

This script asks for a match ID, loads:
1) Metrica_Output/Distances/game_<match_id>_player_distances.xlsx
2) Metrica_Output/Full Match Analysis/player_influence_analysis_additive_epv_match_<match_id>.xlsx

It then:
- Builds full-match total added EPV per player by summing period totals.
- Maps influence players to Player ID format (Home_<jersey>, Away_<jersey>).
- Merges with distance data.
- Creates a correlation scatter plot (distance vs total added EPV).

Usage:
    python Scripts/plot_distance_vs_total_added_epv.py
"""

import json
import os
import re
import sys
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 110


def normalize_name(name):
    """Normalize names for matching across different sources."""
    if not isinstance(name, str):
        return ""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_name = ascii_name.replace("\xad", "")
    ascii_name = re.sub(r"\s+", " ", ascii_name).strip().lower()
    return ascii_name


def load_distance_data(distance_file):
    """Load and combine distance sheets for both teams."""
    print(f"Loading distances: {distance_file}")

    try:
        home_df = pd.read_excel(distance_file, sheet_name="Home Team")
        away_df = pd.read_excel(distance_file, sheet_name="Away Team")
    except Exception:
        # Fallback for older naming
        home_df = pd.read_excel(distance_file, sheet_name="Home")
        away_df = pd.read_excel(distance_file, sheet_name="Away")

    distances = pd.concat([home_df, away_df], ignore_index=True)

    required_cols = {"Player ID", "Distance [km]"}
    missing = required_cols - set(distances.columns)
    if missing:
        raise ValueError(f"Missing required columns in distance file: {sorted(missing)}")

    distances = distances[["Player ID", "Distance [km]"]].copy()
    distances.rename(columns={"Distance [km]": "Total Distance (km)"}, inplace=True)
    distances["Total Distance (m)"] = distances["Total Distance (km)"] * 1000.0

    print(f"  Loaded {len(distances)} players with distance data")
    return distances


def load_name_to_player_id_map(match_id):
    """Build name -> Player ID map from PFF tracking JSONL game_event entries."""
    tracking_file = os.path.join("PFF Data", "Tracking Data", f"{match_id}.jsonl")
    mapping = {"Home": {}, "Away": {}}

    if not os.path.exists(tracking_file):
        print(f"Warning: tracking file not found for name mapping: {tracking_file}")
        return mapping

    with open(tracking_file, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue

            event = frame.get("game_event")
            if not event:
                continue

            shirt_number = event.get("shirt_number")
            player_name = event.get("player_name")
            is_home = event.get("home_team", 0)

            if shirt_number is None or not player_name:
                continue

            team = "Home" if is_home else "Away"
            player_id = f"{team}_{shirt_number}"

            mapping[team][normalize_name(player_name)] = player_id

    print(
        "  Name map entries: "
        f"Home={len(mapping['Home'])}, Away={len(mapping['Away'])}"
    )
    return mapping


def parse_player_to_id(player_value, team, name_map):
    """Convert a player label from influence sheets to Player ID format."""
    if not isinstance(player_value, str):
        return None

    player_text = player_value.strip()

    # Cases like "#10" or "10"
    if player_text.startswith("#") and player_text[1:].isdigit():
        return f"{team}_{player_text[1:]}"
    if player_text.isdigit():
        return f"{team}_{player_text}"

    # Fallback to name lookup
    normalized = normalize_name(player_text)
    return name_map.get(team, {}).get(normalized)


def load_influence_totals(influence_file, match_id):
    """Load and sum whole-match additive influence (period 1 + period 2)."""
    print(f"Loading influence totals: {influence_file}")

    expected_sheets = {
        "Home": ["Home Period 1 Total", "Home Period 2 Total"],
        "Away": ["Away Period 1 Total", "Away Period 2 Total"],
    }

    xl = pd.ExcelFile(influence_file)
    available = set(xl.sheet_names)

    name_map = load_name_to_player_id_map(match_id)

    rows = []
    for team, sheets in expected_sheets.items():
        for sheet in sheets:
            if sheet not in available:
                print(f"  Warning: missing sheet '{sheet}' in influence file")
                continue

            df = pd.read_excel(influence_file, sheet_name=sheet)
            if "Player" not in df.columns or "Total Influence" not in df.columns:
                print(f"  Warning: sheet '{sheet}' missing Player/Total Influence columns")
                continue

            for _, rec in df.iterrows():
                player_name = rec["Player"]
                total_influence = rec["Total Influence"]
                player_id = parse_player_to_id(player_name, team, name_map)

                rows.append(
                    {
                        "Team": team,
                        "Player": player_name,
                        "Player ID": player_id,
                        "Period Total Added EPV": float(total_influence) if pd.notna(total_influence) else 0.0,
                    }
                )

    if not rows:
        raise ValueError("No influence rows could be loaded from expected sheets")

    influence = pd.DataFrame(rows)

    # Sum period totals to get whole-match value per player.
    influence_totals = (
        influence.groupby(["Team", "Player", "Player ID"], dropna=False, as_index=False)[
            "Period Total Added EPV"
        ]
        .sum()
        .rename(columns={"Period Total Added EPV": "Total Added EPV"})
    )

    unresolved = influence_totals["Player ID"].isna().sum()
    if unresolved > 0:
        print(f"  Warning: {unresolved} players could not be mapped to Player ID")

    print(f"  Loaded {len(influence_totals)} whole-match influence totals")
    return influence_totals


def calculate_stats(x, y):
    """Compute correlation/regression stats after dropping NaNs."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        raise ValueError("Not enough valid points for correlation (need at least 2)")

    pearson_r, pearson_p = pearsonr(x_clean, y_clean)
    spearman_r, spearman_p = spearmanr(x_clean, y_clean)
    slope, intercept, r_val, p_val, std_err = linregress(x_clean, y_clean)

    return {
        "n": len(x_clean),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_val ** 2,
        "regression_p": p_val,
        "std_err": std_err,
    }


def make_plot(merged, stats_dict, output_path, match_id):
    """Create and save distance vs total added EPV scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 9))

    home = merged[merged["Team"] == "Home"]
    away = merged[merged["Team"] == "Away"]

    ax.scatter(
        home["Total Distance (m)"],
        home["Total Added EPV"],
        c="#d1495b",
        s=180,
        alpha=0.75,
        edgecolors="black",
        linewidth=0.7,
        label="Home",
        zorder=3,
    )
    ax.scatter(
        away["Total Distance (m)"],
        away["Total Added EPV"],
        c="#2e86ab",
        s=180,
        alpha=0.75,
        edgecolors="black",
        linewidth=0.7,
        label="Away",
        zorder=3,
    )

    # Annotate with jersey numbers for readability.
    for _, row in merged.iterrows():
        player_id = str(row["Player ID"])
        jersey = player_id.split("_")[-1] if "_" in player_id else "?"
        ax.annotate(
            jersey,
            (row["Total Distance (m)"], row["Total Added EPV"]),
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            zorder=4,
        )

    x = merged["Total Distance (m)"].values
    x_line = np.array([x.min(), x.max()])
    y_line = stats_dict["slope"] * x_line + stats_dict["intercept"]
    ax.plot(
        x_line,
        y_line,
        "k--",
        linewidth=2,
        alpha=0.8,
        label=f"Trend: y={stats_dict['slope']:.6f}x+{stats_dict['intercept']:.4f}",
        zorder=2,
    )

    ax.set_xlabel("Total Distance Covered (m)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Added EPV (Whole Match)", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Distance Covered vs Total Added EPV (Match {match_id})",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, frameon=True)

    stats_text = f"R = {stats_dict['pearson_r']:.3f}"
    ax.text(
        0.03,
        0.97,
        stats_text,
        transform=ax.transAxes,
        va="top",
        fontsize=24,
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.9},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    print("=" * 72)
    print("DISTANCE vs TOTAL ADDED EPV CORRELATION (WHOLE MATCH)")
    print("=" * 72)

    match_id = input("Enter match ID (e.g., 10517, 3822): ").strip()
    if not match_id:
        print("ERROR: Match ID is required")
        sys.exit(1)

    distance_file = os.path.join(
        "Metrica_Output", "Distances", f"game_{match_id}_player_distances.xlsx"
    )
    influence_file = os.path.join(
        "Metrica_Output",
        "Full Match Analysis",
        f"player_influence_analysis_additive_epv_match_{match_id}.xlsx",
    )

    if not os.path.exists(distance_file):
        print(f"ERROR: Distance file not found: {distance_file}")
        sys.exit(1)
    if not os.path.exists(influence_file):
        print(f"ERROR: Influence file not found: {influence_file}")
        sys.exit(1)

    distances_df = load_distance_data(distance_file)
    influence_df = load_influence_totals(influence_file, match_id)

    merged = pd.merge(influence_df, distances_df, on="Player ID", how="inner")
    merged = merged[merged["Total Distance (m)"].notna() & merged["Total Added EPV"].notna()].copy()

    if len(merged) < 2:
        print("ERROR: Not enough merged players to compute correlation")
        print("Tip: verify Player ID mapping and source files for this match")
        sys.exit(1)

    print(f"Merged players: {len(merged)}")
    print(f"  Home: {(merged['Team'] == 'Home').sum()}")
    print(f"  Away: {(merged['Team'] == 'Away').sum()}")

    stats_dict = calculate_stats(merged["Total Distance (m)"].values, merged["Total Added EPV"].values)

    output_dir = os.path.join("Metrica_Output", "Plots")
    os.makedirs(output_dir, exist_ok=True)

    plot_file = os.path.join(output_dir, f"distance_vs_total_added_epv_match_{match_id}.png")
    make_plot(merged, stats_dict, plot_file, match_id)

    print("\nCorrelation Summary")
    print("-" * 72)
    print(f"Pearson r:   {stats_dict['pearson_r']:.4f} (p = {stats_dict['pearson_p']:.4e})")
    print(f"Spearman rho:{stats_dict['spearman_r']:.4f} (p = {stats_dict['spearman_p']:.4e})")
    print(f"R^2:         {stats_dict['r_squared']:.4f}")
    print(f"Slope:       {stats_dict['slope']:.6f} +/- {stats_dict['std_err']:.6f}")

    print("\nOutputs")
    print("-" * 72)
    print(f"Plot:   {plot_file}")


if __name__ == "__main__":
    main()
