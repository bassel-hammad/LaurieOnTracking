"""
Visualization module for Player Influence Analysis.

Handles all plotting, animation, and movie generation.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Arc, Circle

from .config import Config


class Visualizer:
    """Handles visualization and movie generation."""
    
    def __init__(self, data_loader, influence_calculator):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        data_loader : DataLoader
            Data loader instance
        influence_calculator : InfluenceCalculator
            Influence calculator with results
        """
        self.data_loader = data_loader
        self.influence_calc = influence_calculator
        
        self.fig = None
        self.ax_pitch = None
        self.ax_bar_attack = None
        self.ax_bar_defend = None
        self.time_text = None
        self.player_artists = []
    
    def generate_correlation_plot(self, sequence_number, save_path=None):
        """
        Generate a scatter plot showing correlation between Positive and Net influence.
        Each point represents one player in one 1-second interval.
        
        Parameters
        ----------
        sequence_number : int
            Sequence number for title
        save_path : str, optional
            Path to save the plot. If None, displays interactively.
        """
        # Collect data from individual frame transitions
        positive_vals = []
        net_vals = []
        player_names = []
        time_diffs = []
        event_indices = []
        
        # Determine analysis mode
        analysis_mode = self.influence_calc.analysis_mode
        
        for event_idx, result in enumerate(self.influence_calc.influence_results):
            time_t = result['time_t']
            time_t1 = result['time_t1']
            delta_t = time_t1 - time_t
            
            for player_id, influence_data in result['player_influences'].items():
                name = self.data_loader.get_player_display_name(player_id)
                player_names.append(name)
                
                if analysis_mode == 'attacking':
                    # For attacking mode: use additive approach
                    pos = influence_data.get('positive_additive', influence_data.get('positive_influence', 0))
                    neg = influence_data.get('negative_additive', influence_data.get('negative_influence', 0))
                    net = pos + neg
                    positive_vals.append(pos)
                else:
                    # For defending mode: use necessity approach
                    net = influence_data.get('net_necessity', 0)
                    positive_vals.append(net)  # Use net as "positive" for correlation
                
                net_vals.append(net)
                time_diffs.append(delta_t)
                event_indices.append(event_idx)
        
        if not positive_vals:
            print("No player data available for correlation plot.")
            return
        
        positive_vals = np.array(positive_vals)
        net_vals = np.array(net_vals)
        time_diffs = np.array(time_diffs)
        
        # Calculate correlation coefficient
        if len(positive_vals) > 1:
            correlation = np.corrcoef(positive_vals, net_vals)[0, 1]
        else:
            correlation = 0
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Create colormap for time differences
        norm = plt.Normalize(vmin=min(time_diffs), vmax=max(time_diffs))
        cmap = cm.viridis
        
        # Scatter plot colored by time difference
        scatter = ax.scatter(positive_vals, net_vals, s=80, c=time_diffs,
                            cmap=cmap, norm=norm, edgecolors='black', 
                            linewidth=1, alpha=0.7, zorder=5)
        
        # Add colorbar for time differences
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time Δt (seconds)', fontsize=10, fontweight='bold')
        
        # Add reference lines
        all_vals = np.concatenate([positive_vals, net_vals])
        if len(all_vals) > 0:
            min_val = min(all_vals)
            max_val = max(all_vals)
            margin = max(abs(max_val - min_val) * 0.1, 0.1)
            min_val -= margin
            max_val += margin
        else:
            min_val, max_val = -1, 1
        
        # Add horizontal line at y=0 (net=0)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
        
        # Add regression line
        if len(positive_vals) > 1:
            z = np.polyfit(positive_vals, net_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(positive_vals), max(positive_vals), 100)
            ax.plot(x_line, p(x_line), 'r-', alpha=0.7, linewidth=2,
                   label=f'Regression (r = {correlation:.3f})', zorder=2)
        
        # Formatting based on mode
        if analysis_mode == 'attacking':
            ax.set_xlabel('Positive Influence', fontsize=12, fontweight='bold')
            ax.set_ylabel('Net Influence', fontsize=12, fontweight='bold')
            plot_title = 'Correlation: Positive vs Net Influence'
        else:
            ax.set_xlabel('Net Influence (Necessity)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Net Influence', fontsize=12, fontweight='bold')
            plot_title = 'Net Influence Distribution (Defending)'
        
        num_events = len(self.influence_calc.influence_results)
        num_points = len(positive_vals)
        title = f'{plot_title}\n'
        title += f'Sequence {int(sequence_number)} | {num_events} Intervals | {num_points} Player-Interval Points | Mode: {analysis_mode.upper()}'
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add correlation annotation
        if analysis_mode == 'attacking':
            stats_text = f'Pearson r = {correlation:.3f}\n'
            stats_text += f'Δt = {np.mean(time_diffs):.2f}s (fixed interval)'
        else:
            stats_text = f'Mean influence = {np.mean(net_vals):.3f}\n'
            stats_text += f'Δt = {np.mean(time_diffs):.2f}s (fixed interval)'
        
        ax.text(0.05, 0.95, stats_text, 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add interpretation text (only for attacking mode with correlation)
        if analysis_mode == 'attacking':
            if correlation > 0.9:
                interp = "Very strong correlation - approaches agree closely"
            elif correlation > 0.7:
                interp = "Strong correlation - approaches mostly agree"
            elif correlation > 0.5:
                interp = "Moderate correlation - some differences"
            else:
                interp = "Weak correlation - approaches give different insights"
            
            ax.text(0.05, 0.80, interp, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', style='italic',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Correlation plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return correlation

    def generate_movie(self, movie_frames, output_path, sequence_number, verbose=True):
        """
        Generate the player influence movie.
        
        Parameters
        ----------
        movie_frames : list
            List of frame numbers for the movie
        output_path : str
            Output file path
        sequence_number : int
            Sequence number for title
        verbose : bool
            Print progress messages
        """
        if verbose:
            print("Creating animation...")
        
        # Prepare pitch control data for each frame
        pitch_control_data = self._prepare_pitch_control_data(movie_frames, verbose)
        
        # Set up figure and axes
        self._setup_figure()
        
        # Prepare bar chart data
        bar_data = self._prepare_bar_chart_data()
        
        # Create animation
        if verbose:
            print(f"Generating animation with {len(pitch_control_data)} frames...")
        
        # Calculate interval
        if len(pitch_control_data) > 1:
            time_diffs = [
                pitch_control_data[i+1]['time'] - pitch_control_data[i]['time']
                for i in range(len(pitch_control_data)-1)
            ]
            avg_time_diff = np.mean(time_diffs)
            interval_ms = int(avg_time_diff * 1000)
            interval_ms = max(100, min(500, interval_ms))
        else:
            interval_ms = int(1000 / Config.TARGET_FPS)
        
        if verbose:
            print(f"Using interval: {interval_ms}ms")
        
        # Create animation function
        def animate(frame_idx):
            return self._animate_frame(frame_idx, pitch_control_data, bar_data)
        
        anim = animation.FuncAnimation(
            self.fig,
            animate,
            frames=len(pitch_control_data),
            interval=interval_ms,
            blit=False,
            repeat=True
        )
        
        # Add title
        attacking_team = self.influence_calc.attacking_team
        defending_team = self.influence_calc.defending_team
        self.fig.suptitle(
            f"Player Influence Analysis - Sequence {int(sequence_number)} "
            f"({attacking_team} vs {defending_team})",
            fontsize=15, y=0.98
        )
        
        # Save
        if verbose:
            print(f"Saving movie to: {output_path}")
        
        actual_fps = 1000 / interval_ms
        writer = animation.FFMpegWriter(
            fps=actual_fps,
            metadata=dict(artist='LaurieOnTracking'),
            bitrate=Config.BITRATE
        )
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                anim.save(output_path, writer=writer, dpi=Config.DPI)
            
            if verbose:
                print()
                print("=" * 70)
                print("MOVIE GENERATION COMPLETE!")
                print("=" * 70)
                print(f"Movie saved to: {output_path}")
                print(f"Frames: {len(pitch_control_data)}")
                print(f"Playback FPS: {actual_fps:.1f}")
                print()
                
        except Exception as e:
            print(f"\nERROR: Failed to save movie: {e}")
            print("\nNote: This script requires FFMpeg to be installed.")
            self._try_save_gif(anim, output_path)
        
        plt.close()
    
    def _prepare_pitch_control_data(self, movie_frames, verbose=True):
        """Prepare pitch control data for each frame."""
        pitch_control_data = []
        
        for i, frame in enumerate(movie_frames):
            if verbose and i % 10 == 0:
                print(f"  Frame {i+1}/{len(movie_frames)}...")
            
            if frame not in self.data_loader.tracking_home.index:
                continue
            if frame not in self.data_loader.tracking_away.index:
                continue
            
            home_row = self.data_loader.tracking_home.loc[frame].copy()
            away_row = self.data_loader.tracking_away.loc[frame].copy()
            
            pitch_control_data.append({
                'home_row': home_row,
                'away_row': away_row,
                'ball_pos': np.array([home_row['ball_x'], home_row['ball_y']]),
                'time': home_row['Time [s]'],
                'frame': frame
            })
        
        return pitch_control_data
    
    def _prepare_bar_chart_data(self):
        """Prepare data for bar charts based on analysis mode."""
        analysis_mode = self.influence_calc.analysis_mode
        
        if analysis_mode == 'attacking':
            # Get sorted attackers for each metric
            sorted_by_positive = self.influence_calc.get_sorted_attackers(by='positive_additive', reverse=True)
            sorted_by_net = self.influence_calc.get_sorted_attackers(by='net_additive', reverse=True)
            
            # Calculate total (absolute) = positive + |negative|
            attackers_with_total = []
            for player_id, stats in self.influence_calc.attacker_influences.items():
                pos = stats.get('positive_additive', stats.get('positive_influence', 0))
                neg = stats.get('negative_additive', stats.get('negative_influence', 0))
                total = pos + abs(neg)
                attackers_with_total.append((player_id, total, stats))
            sorted_by_total = sorted(attackers_with_total, key=lambda x: x[1], reverse=True)
            
            # Prepare data for each metric
            # Positive influence
            positive_names = [self.data_loader.get_player_display_name(p[0]) for p in sorted_by_positive]
            positive_ids = [p[0] for p in sorted_by_positive]
            positive_vals = [p[1].get('positive_additive', p[1].get('positive_influence', 0)) for p in sorted_by_positive]
            positive_rankings = {p[0]: rank for rank, p in enumerate(sorted_by_positive, 1)}
            top5_positive_ids = [p[0] for p in sorted_by_positive[:5]]
            
            # Total influence (positive + |negative|)
            total_names = [self.data_loader.get_player_display_name(p[0]) for p in sorted_by_total]
            total_ids = [p[0] for p in sorted_by_total]
            total_vals = [p[1] for p in sorted_by_total]
            total_rankings = {p[0]: rank for rank, p in enumerate(sorted_by_total, 1)}
            top5_total_ids = [p[0] for p in sorted_by_total[:5]]
            
            # Net influence (positive + negative, can be positive or negative)
            net_names = [self.data_loader.get_player_display_name(p[0]) for p in sorted_by_net]
            net_ids = [p[0] for p in sorted_by_net]
            net_vals = [p[1].get('net_additive', p[1].get('net', 0)) for p in sorted_by_net]
            net_rankings = {p[0]: rank for rank, p in enumerate(sorted_by_net, 1)}
            top5_net_ids = [p[0] for p in sorted_by_net[:5]]
            
            # No defender data in attacking mode
            defender_names = []
            defender_ids = []
            defender_vals = []
            defender_rankings = {}
            top5_defender_ids = []
        else:
            # Defending mode - use defender data
            sorted_defenders = self.influence_calc.get_sorted_players(reverse=True)
            
            # Populate defender data
            defender_names = [self.data_loader.get_player_display_name(p[0]) for p in sorted_defenders]
            defender_ids = [p[0] for p in sorted_defenders]
            defender_vals = [p[1].get('net_necessity', 0) for p in sorted_defenders]
            defender_rankings = {p[0]: rank for rank, p in enumerate(sorted_defenders, 1)}
            top5_defender_ids = [p[0] for p in sorted_defenders[:5]]
            
            # No attacker data in defending mode
            positive_names = []
            positive_ids = []
            positive_vals = []
            positive_rankings = {}
            top5_positive_ids = []
            
            total_names = []
            total_ids = []
            total_vals = []
            total_rankings = {}
            top5_total_ids = []
            
            net_names = []
            net_ids = []
            net_vals = []
            net_rankings = {}
            top5_net_ids = []
        
        # Event colors
        num_events = len(self.influence_calc.influence_results)
        event_colors = cm.rainbow(np.linspace(0, 1, num_events))
        
        return {
            # Attackers - Positive influence
            'positive_names': positive_names,
            'positive_ids': positive_ids,
            'positive_vals': positive_vals,
            'positive_rankings': positive_rankings,
            'top5_positive_ids': top5_positive_ids,
            
            # Attackers - Total influence
            'total_names': total_names,
            'total_ids': total_ids,
            'total_vals': total_vals,
            'total_rankings': total_rankings,
            'top5_total_ids': top5_total_ids,
            
            # Attackers - Net influence
            'net_names': net_names,
            'net_ids': net_ids,
            'net_vals': net_vals,
            'net_rankings': net_rankings,
            'top5_net_ids': top5_net_ids,
            
            # Defenders
            'defender_names': defender_names,
            'defender_ids': defender_ids,
            'defender_vals': defender_vals,
            'defender_rankings': defender_rankings,
            'top5_defender_ids': top5_defender_ids,
            
            'event_colors': event_colors,
        }
    
    def _setup_figure(self):
        """Set up the figure with 1 pitch and bar chart(s) based on available data."""
        # Check what data is available
        has_attackers = bool(self.influence_calc.attacker_influences)
        has_defenders = bool(self.influence_calc.defender_influences)
        
        if has_attackers and has_defenders:
            # Both teams analyzed - show 2 bar charts
            self.fig = plt.figure(figsize=Config.FIGURE_SIZE)
            gs = self.fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], hspace=0.3, wspace=0.4)
            
            # Pitch (left side)
            self.ax_pitch = self.fig.add_subplot(gs[0, 0])
            self._draw_pitch_on_ax(self.ax_pitch)
            
            # Bar chart for attackers (middle)
            self.ax_bar_attackers = self.fig.add_subplot(gs[0, 1])
            
            # Bar chart for defenders (right side)
            self.ax_bar_defenders = self.fig.add_subplot(gs[0, 2])
        else:
            # Only one team analyzed - show 1 bar chart
            self.fig = plt.figure(figsize=Config.FIGURE_SIZE)
            gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.3)
            
            # Pitch (left side)
            self.ax_pitch = self.fig.add_subplot(gs[0, 0])
            self._draw_pitch_on_ax(self.ax_pitch)
            
            # Single bar chart (right side) - use universal ax_bar
            self.ax_bar = self.fig.add_subplot(gs[0, 1])
            self.ax_bar_attackers = None
            self.ax_bar_defenders = None
        
        # Time text
        self.time_text = self.ax_pitch.text(
            0.02, 0.98, '',
            transform=self.ax_pitch.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    def _draw_pitch_on_ax(self, ax):
        """Draw the football pitch on a specific axis."""
        ax.set_xlim(-55, 55)
        ax.set_ylim(-36, 36)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Green pitch
        pitch_rect = Rectangle((-52.5, -34), 105, 68,
                               facecolor='#2E8B57', edgecolor='white', linewidth=2, zorder=0)
        ax.add_patch(pitch_rect)
        
        # Pitch lines
        ax.plot([0, 0], [-34, 34], 'white', linewidth=2)  # Halfway line
        ax.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor='white', linewidth=2))
        
        # Penalty areas
        for x_sign in [-1, 1]:
            x_base = x_sign * 52.5
            x_box = x_sign * 36
            x_6yard = x_sign * 47
            
            # Penalty area
            ax.plot([x_base, x_box], [-20.16, -20.16], 'white', linewidth=2)
            ax.plot([x_box, x_box], [-20.16, 20.16], 'white', linewidth=2)
            ax.plot([x_base, x_box], [20.16, 20.16], 'white', linewidth=2)
            
            # Goal area
            ax.plot([x_base, x_6yard], [-9.16, -9.16], 'white', linewidth=2)
            ax.plot([x_6yard, x_6yard], [-9.16, 9.16], 'white', linewidth=2)
            ax.plot([x_base, x_6yard], [9.16, 9.16], 'white', linewidth=2)
            
            # Penalty spot
            ax.plot(x_sign * 41.5, 0, 'o', color='white', markersize=4)
        
        # Corner arcs
        for x, y, t1, t2 in [(-52.5, -34, 0, 90), (-52.5, 34, 270, 360),
                             (52.5, -34, 90, 180), (52.5, 34, 180, 270)]:
            ax.add_patch(Arc((x, y), 2, 2, angle=0, theta1=t1, theta2=t2,
                            color='white', linewidth=2))
    
    def _draw_pitch(self):
        """Draw the football pitch (legacy - uses first pitch)."""
        self._draw_pitch_on_ax(self.ax_pitch)
    
    def _animate_frame(self, frame_idx, pitch_control_data, bar_data):
        """Animate a single frame - updates 1 pitch and 2 bar charts (Attackers and Defenders)."""
        data = pitch_control_data[frame_idx]
        
        # Clear previous player positions
        for artist in self.player_artists:
            artist.remove()
        self.player_artists = []
        
        # Clear text annotations
        for txt in self.ax_pitch.texts[:]:
            if txt != self.time_text:
                txt.remove()
        
        # Draw players with both attacker and defender rankings
        self._draw_players_on_pitch(
            self.ax_pitch, data,
            bar_data['top5_net_ids'], bar_data['net_rankings'],
            bar_data['top5_defender_ids'], bar_data['defender_rankings']
        )
        
        # Draw ball
        ball_artist = self.ax_pitch.plot(
            data['ball_pos'][0], data['ball_pos'][1],
            'ko', markersize=Config.BALL_MARKER_SIZE, alpha=1.0, zorder=10
        )[0]
        self.player_artists.append(ball_artist)
        
        # Update time text
        time_str = f"Time: {data['time']:.1f}s | Frame: {frame_idx+1}/{len(pitch_control_data)}"
        self.time_text.set_text(time_str)
        
        # Update bar charts
        self._update_bar_charts(data['time'], bar_data)
        
        return [self.time_text]
    
    def _draw_players_on_pitch(self, ax, data, top5_attacker_ids, attacker_rankings, 
                                top5_defender_ids, defender_rankings):
        """Draw players on a specific pitch with attacker and/or defender top 5 rankings."""
        attacking_team = self.influence_calc.attacking_team
        has_attackers = bool(attacker_rankings)
        has_defenders = bool(defender_rankings)
        
        if attacking_team == 'Home':
            # Home attacking, Away defending
            if has_attackers:
                self._draw_team_players_on_ax(
                    ax, data['home_row'], top5_attacker_ids, attacker_rankings,
                    Config.HOME_COLOR_TOP5, Config.HOME_COLOR_OTHERS, show_rankings=True
                )
            else:
                self._draw_team_players_plain_on_ax(ax, data['home_row'], Config.HOME_COLOR_OTHERS)
            
            if has_defenders:
                self._draw_team_players_on_ax(
                    ax, data['away_row'], top5_defender_ids, defender_rankings,
                    Config.AWAY_COLOR_TOP5, Config.AWAY_COLOR_OTHERS, show_rankings=True
                )
            else:
                self._draw_team_players_plain_on_ax(ax, data['away_row'], Config.AWAY_COLOR_OTHERS)
        else:
            # Away attacking, Home defending
            if has_attackers:
                self._draw_team_players_on_ax(
                    ax, data['away_row'], top5_attacker_ids, attacker_rankings,
                    Config.AWAY_COLOR_TOP5, Config.AWAY_COLOR_OTHERS, show_rankings=True
                )
            else:
                self._draw_team_players_plain_on_ax(ax, data['away_row'], Config.AWAY_COLOR_OTHERS)
            
            if has_defenders:
                self._draw_team_players_on_ax(
                    ax, data['home_row'], top5_defender_ids, defender_rankings,
                    Config.HOME_COLOR_TOP5, Config.HOME_COLOR_OTHERS, show_rankings=True
                )
            else:
                self._draw_team_players_plain_on_ax(ax, data['home_row'], Config.HOME_COLOR_OTHERS)
    
    def _draw_team_players_on_ax(self, ax, row, top5_ids, rankings, color_top5, color_others, show_rankings=True):
        """Draw players for a single team on a specific axis with optional rankings."""
        x_cols = [c for c in row.keys() if c[-2:].lower() == '_x' and c != 'ball_x'
                  and 'visibility' not in c.lower()]
        y_cols = [c for c in row.keys() if c[-2:].lower() == '_y' and c != 'ball_y'
                  and 'visibility' not in c.lower()]
        
        top5_x, top5_y = [], []
        other_x, other_y = [], []
        
        for x_col, y_col in zip(x_cols, y_cols):
            player_id = x_col.replace('_x', '')
            x_pos = row[x_col]
            y_pos = row[y_col]
            
            if not pd.isna(x_pos) and not pd.isna(y_pos):
                if player_id in top5_ids:
                    top5_x.append(x_pos)
                    top5_y.append(y_pos)
                else:
                    other_x.append(x_pos)
                    other_y.append(y_pos)
        
        # Draw other players
        if other_x:
            line = ax.plot(
                other_x, other_y, 'o', color=color_others,
                markersize=Config.PLAYER_MARKER_SIZE, alpha=0.6,
                markeredgecolor='black', markeredgewidth=1.5
            )[0]
            self.player_artists.append(line)
        
        # Draw top 5
        if top5_x:
            line = ax.plot(
                top5_x, top5_y, 'o', color=color_top5,
                markersize=Config.TOP5_MARKER_SIZE, alpha=0.9,
                markeredgecolor='black', markeredgewidth=1.5
            )[0]
            self.player_artists.append(line)
        
        # Add ranking numbers if enabled
        if show_rankings:
            for x_col, y_col in zip(x_cols, y_cols):
                player_id = x_col.replace('_x', '')
                if player_id in top5_ids:
                    x_pos = row[x_col]
                    y_pos = row[y_col]
                    if not pd.isna(x_pos) and not pd.isna(y_pos):
                        rank = rankings[player_id]
                        txt = ax.text(
                            x_pos, y_pos, str(rank),
                            fontsize=9, fontweight='bold', color='white',
                            ha='center', va='center', zorder=11
                        )
                        self.player_artists.append(txt)
    
    def _draw_team_players_plain_on_ax(self, ax, row, color):
        """Draw players for a team on a specific axis without rankings."""
        x_cols = [c for c in row.keys() if c[-2:].lower() == '_x' and c != 'ball_x'
                  and 'visibility' not in c.lower()]
        y_cols = [c for c in row.keys() if c[-2:].lower() == '_y' and c != 'ball_y'
                  and 'visibility' not in c.lower()]
        
        x_positions = []
        y_positions = []
        
        for x_col, y_col in zip(x_cols, y_cols):
            x_pos = row[x_col]
            y_pos = row[y_col]
            if not pd.isna(x_pos) and not pd.isna(y_pos):
                x_positions.append(x_pos)
                y_positions.append(y_pos)
        
        if x_positions:
            line = ax.plot(
                x_positions, y_positions, 'o', color=color,
                markersize=Config.PLAYER_MARKER_SIZE, alpha=0.5,
                markeredgecolor='black', markeredgewidth=1
            )[0]
            self.player_artists.append(line)
    
    def _draw_team_players(self, row, top5_ids, rankings, color_top5, color_others, show_rankings=True):
        """Draw players for a single team with optional rankings."""
        x_cols = [c for c in row.keys() if c[-2:].lower() == '_x' and c != 'ball_x'
                  and 'visibility' not in c.lower()]
        y_cols = [c for c in row.keys() if c[-2:].lower() == '_y' and c != 'ball_y'
                  and 'visibility' not in c.lower()]
        
        top5_x, top5_y = [], []
        other_x, other_y = [], []
        
        for x_col, y_col in zip(x_cols, y_cols):
            player_id = x_col.replace('_x', '')
            x_pos = row[x_col]
            y_pos = row[y_col]
            
            if not pd.isna(x_pos) and not pd.isna(y_pos):
                if player_id in top5_ids:
                    top5_x.append(x_pos)
                    top5_y.append(y_pos)
                else:
                    other_x.append(x_pos)
                    other_y.append(y_pos)
        
        # Draw other players
        if other_x:
            line = self.ax_pitch.plot(
                other_x, other_y, 'o', color=color_others,
                markersize=Config.PLAYER_MARKER_SIZE, alpha=0.6,
                markeredgecolor='black', markeredgewidth=1.5
            )[0]
            self.player_artists.append(line)
        
        # Draw top 5
        if top5_x:
            line = self.ax_pitch.plot(
                top5_x, top5_y, 'o', color=color_top5,
                markersize=Config.TOP5_MARKER_SIZE, alpha=0.9,
                markeredgecolor='black', markeredgewidth=1.5
            )[0]
            self.player_artists.append(line)
        
        # Add ranking numbers if enabled
        if show_rankings:
            for x_col, y_col in zip(x_cols, y_cols):
                player_id = x_col.replace('_x', '')
                if player_id in top5_ids:
                    x_pos = row[x_col]
                    y_pos = row[y_col]
                    if not pd.isna(x_pos) and not pd.isna(y_pos):
                        rank = rankings[player_id]
                        txt = self.ax_pitch.text(
                            x_pos, y_pos, str(rank),
                            fontsize=9, fontweight='bold', color='white',
                            ha='center', va='center', zorder=11
                        )
                        self.player_artists.append(txt)
    
    def _draw_team_players_plain(self, row, color):
        """Draw players for a team without rankings (plain markers)."""
        x_cols = [c for c in row.keys() if c[-2:].lower() == '_x' and c != 'ball_x'
                  and 'visibility' not in c.lower()]
        y_cols = [c for c in row.keys() if c[-2:].lower() == '_y' and c != 'ball_y'
                  and 'visibility' not in c.lower()]
        
        x_positions = []
        y_positions = []
        
        for x_col, y_col in zip(x_cols, y_cols):
            x_pos = row[x_col]
            y_pos = row[y_col]
            if not pd.isna(x_pos) and not pd.isna(y_pos):
                x_positions.append(x_pos)
                y_positions.append(y_pos)
        
        if x_positions:
            line = self.ax_pitch.plot(
                x_positions, y_positions, 'o', color=color,
                markersize=Config.PLAYER_MARKER_SIZE, alpha=0.5,
                markeredgecolor='black', markeredgewidth=1
            )[0]
            self.player_artists.append(line)
    
    def _update_bar_charts(self, current_time, bar_data):
        """Update bar chart based on analysis mode."""
        attacking_team = self.influence_calc.attacking_team
        defending_team = self.influence_calc.defending_team
        analysis_mode = self.influence_calc.analysis_mode
        
        # Use universal ax_bar if single mode, otherwise use specific axes
        ax_to_use = None
        if hasattr(self, 'ax_bar') and self.ax_bar is not None:
            ax_to_use = self.ax_bar
        elif analysis_mode == 'attacking' and self.ax_bar_attackers is not None:
            ax_to_use = self.ax_bar_attackers
        elif analysis_mode == 'defending' and self.ax_bar_defenders is not None:
            ax_to_use = self.ax_bar_defenders
        
        if ax_to_use is None:
            return
        
        ax_to_use.clear()
        
        if analysis_mode == 'attacking':
            # Calculate cumulative values up to current time for attackers
            net_cumulative = {pid: 0.0 for pid in bar_data['net_ids']}
            
            for result in self.influence_calc.influence_results:
                if result['time_t1'] <= current_time:
                    for pid, inf in result['player_influences'].items():
                        if pid in net_cumulative:
                            pos = inf.get('positive_additive', inf.get('positive_influence', 0))
                            neg = inf.get('negative_additive', inf.get('negative_influence', 0))
                            net_cumulative[pid] += (pos + neg)
            
            net_vals = [net_cumulative.get(pid, 0) for pid in bar_data['net_ids']]
            player_names = bar_data['net_names']
            num_players = len(bar_data['net_ids'])
            title = f'{attacking_team} - Attackers'
            main_color = 'crimson'
            alt_color = '#E57373'
        else:
            # Defending mode - calculate cumulative values for defenders
            defender_cumulative = {pid: 0.0 for pid in bar_data['defender_ids']}
            
            for result in self.influence_calc.influence_results:
                if result['time_t1'] <= current_time:
                    for pid, inf in result['player_influences'].items():
                        if pid in defender_cumulative:
                            # Use net_necessity for defending mode
                            val = inf.get('net_necessity', inf.get('net', 0))
                            defender_cumulative[pid] += val
            
            net_vals = [defender_cumulative.get(pid, 0) for pid in bar_data['defender_ids']]
            player_names = bar_data['defender_names']
            num_players = len(bar_data['defender_ids'])
            title = f'{defending_team} - Defenders'
            main_color = 'dodgerblue'
            alt_color = '#64B5F6'
        
        # Draw bar chart
        if num_players > 0:
            x_pos = np.arange(num_players)
            min_val = min(net_vals) if net_vals else 0
            max_val = max(net_vals) if net_vals else 1
            y_margin = max(abs(max_val), abs(min_val), 0.1) * 0.15
            
            ax_to_use.set_xlim(-0.5, num_players - 0.5)
            ax_to_use.set_ylim(min_val - y_margin, max_val + y_margin)
            ax_to_use.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
            ax_to_use.set_xticks(x_pos)
            ax_to_use.set_xticklabels(player_names, rotation=45, ha='right', fontsize=8)
            ax_to_use.set_ylabel('Net Influence', fontsize=10, fontweight='bold')
            ax_to_use.set_title(title, fontsize=11, fontweight='bold', color=main_color)
            ax_to_use.grid(axis='y', alpha=0.3, zorder=0)
            
            # Color bars based on positive/negative
            colors = [main_color if v >= 0 else alt_color for v in net_vals]
            ax_to_use.bar(x_pos, net_vals, color=colors, alpha=0.8,
                         edgecolor='black', linewidth=0.5)
    
    def _try_save_gif(self, anim, output_path):
        """Try to save as GIF as fallback."""
        print("\nAttempting to save as GIF instead...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=1, dpi=100)
            print(f"GIF saved to: {gif_path}")
        except Exception as e:
            print(f"GIF save also failed: {e}")
