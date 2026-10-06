"""
CAPTCHA Experiment Result Visualization Module
Automatically handles missing data and generates publication-quality charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from adjustText import adjust_text  # type: ignore
except Exception:
    adjust_text = None

sns.set_style("whitegrid")
sns.set_palette("husl")

class CAPTCHAVisualizer:
    """CAPTCHA experiment result visualizer"""

    DEFAULT_MODEL_NAMES = {
        'openai/gpt-5': 'GPT-5 (Medium)',
        'openai/gpt-5.1_medium': 'GPT-5.1 (Medium)',
        'openai/gpt-5.1_none': 'GPT-5.1 (None)',
        'anthropic/claude-sonnet-4-5': 'Claude Sonnet 4.5',
        'anthropic/claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet',
        'gemini/gemini-2.5-flash': 'Gemini 2.5 Flash',
        'gemini/gemini-2.5-pro': 'Gemini 2.5 Pro',
        'openrouter/qwen_qwen3-vl-235b-a22b-instruct': 'Qwen3-VL-235B-A22B-Instruct (OpenRouter)',
    }

    DEFAULT_EXP_NAMES = {
        'exp1': 'Exp1 (Original Prompts)',
        'exp2': 'Exp2 (Optimized Prompts)',
        'exp3': 'Exp3 (Until-Correct, Expected)',
    }

    TASK_FAMILY = {
        'Dice_Count': 'Click/Coordinate',
        'Symbol_Count': 'Counting/Generalization',
        'Click_Order': 'Click/Coordinate',
        'Place_Dot': 'Click/Coordinate',
        'Geometry_Click': 'Click/Coordinate',
        'Pick_Area': 'Click/Coordinate',
        'Misleading_Click': 'Click/Coordinate',

        'Patch_Select': 'Grid Selection',
        'Hole_Counting': 'Grid Selection',
        'Select_Animal': 'Grid Selection',
        'Image_Recognition': 'Grid Selection',
        'Unusual_Detection': 'Grid Selection',

        'Image_Matching': 'Image Matching',
        'Relation_Match': 'Semantic Matching',
        'Object_Match': 'Image Matching',
        'Path_Finder': 'Image Matching',
        'Rotation_Match': 'Image Matching',

        'Bingo': 'Logic/Reasoning',
        'Dart_Count': 'Logic/Reasoning',
        'Coordinates': 'Logic/Reasoning',
        'Connect_Icon': 'Logic/Reasoning',
    }

    HARD_TASKS = {
        'Patch_Select', 'Rotation_Match', 'Click_Order',
        'Pick_Area', 'Place_Dot', 'Dice_Count'
    }

    def __init__(self, results_dir: str = "./results", error_dir: str = "./error_analysis",
                 model_names: Optional[Dict[str, str]] = None,
                 exp_names: Optional[Dict[str, str]] = None,
                 exclude_models: Optional[List[str]] = None):
        """
        Initialize CAPTCHA visualizer

        Args:
            results_dir: Results directory path
            error_dir: Error analysis directory path
            model_names: Custom model display names (overrides defaults)
            exp_names: Custom experiment display names (overrides defaults)
            exclude_models: Provider/model identifiers to exclude from charts
        """
        self.results_dir = Path(results_dir)
        self.error_dir = Path(error_dir)
        default_exclude = ["openai/gpt-5-chat-latest"]
        self.exclude_models = set(exclude_models if exclude_models is not None else default_exclude)

        # Merge custom names with defaults
        self.model_names = self.DEFAULT_MODEL_NAMES.copy()
        if model_names:
            self.model_names.update(model_names)

        self.exp_names = self.DEFAULT_EXP_NAMES.copy()
        if exp_names:
            self.exp_names.update(exp_names)

        self.data = self._load_all_data()
        if self.exclude_models and not self.data.empty:
            self.data = self.data[~self.data['provider_model'].isin(self.exclude_models)].copy()
        self.task_types = sorted(self.data['task_type'].unique()) if not self.data.empty else []

    def _format_exp_label(self, experiments) -> str:
        """
        Build standardized experiment label line: "(Exp #) Name" joined by " / " for multiple.
        """
        import re

        def fmt_single(exp: str) -> str:
            disp = self._get_display_name(exp, 'experiment')
            m = re.match(r"Exp\s*([0-9]+)\s*\(?([^)]*)\)?", disp, re.IGNORECASE)
            if m:
                num = m.group(1)
                name = m.group(2).strip() or disp
                return f"(Exp {num}) {name}"
            return f"(Exp {exp.replace('exp','').strip()}) {disp}"

        if isinstance(experiments, (list, tuple, set)):
            parts = [fmt_single(str(e)) for e in experiments if e]
            return " / ".join(parts)
        if experiments:
            return fmt_single(str(experiments))
        return ""

    def _get_display_name(self, identifier: str, name_type: str = 'model') -> str:
        """
        Get display name for model or experiment

        Args:
            identifier: Original identifier (e.g., 'openai/gpt-5')
            name_type: 'model' or 'experiment'

        Returns:
            Display name or original identifier if not found
        """
        if name_type == 'model':
            return self.model_names.get(identifier, identifier)
        elif name_type == 'experiment':
            return self.exp_names.get(identifier, identifier)
        return identifier

    def _load_all_data(self) -> pd.DataFrame:
        """Load all experiment results, automatically handle missing data"""
        all_results = []

        print("Scanning experiment result directories...")

        # Scan all results.csv files
        for csv_file in self.results_dir.rglob("results.csv"):
            try:
                parts = csv_file.relative_to(self.results_dir).parts
                if len(parts) >= 3:
                    experiment = parts[0]  # exp1, exp2, exp3, exp4
                    provider = parts[1]    # openai, anthropic, gemini, fireworks, openrouter
                    model = parts[2]       # gpt-5-chat-latest, claude-3-5-sonnet, etc.

                    df = pd.read_csv(csv_file)

                    # Handle Exp3 special format (until-correct experiment)
                    if experiment == 'exp3' and 'kind' in df.columns:
                        # Filter only summary rows
                        df = df[df['kind'] == 'summary'].copy()

                        # Rename columns to match standard format
                        df = df.rename(columns={
                            'type': 'task_type',
                            'pass1': 'pass'
                        })

                        # Prepare per-row metrics before grouping (each summary row = one task type per run)
                        import numpy as _np
                        eps = 1e-9
                        df['avg_attempts_per_row'] = df['attempt_idx']
                        df['cum_e2e_ms_per_row'] = df['cumulative_ms']
                        df['avg_e2e_ms_per_row'] = df['cumulative_ms'] / _np.clip(df['attempt_idx'], eps, None)

                        # Aggregate statistics by task type (mean across runs if multiple exist)
                        agg_df = df.groupby('task_type').agg({
                            'pass': 'mean',
                            'avg_attempts_per_row': 'mean',
                            'cum_e2e_ms_per_row': 'mean',
                            'avg_e2e_ms_per_row': 'mean'
                        }).reset_index()

                        agg_df = agg_df.rename(columns={
                            'avg_attempts_per_row': 'avg_attempts',
                            'cum_e2e_ms_per_row': 'cum_e2e_ms',
                            'avg_e2e_ms_per_row': 'avg_e2e_ms'
                        })

                        # Add metadata
                        agg_df['experiment'] = experiment
                        agg_df['provider'] = provider
                        agg_df['model'] = model
                        agg_df['provider_model'] = f"{provider}/{model}"
                        agg_df['n'] = df.groupby('task_type').size().values

                        all_results.append(agg_df)
                        print(f"[LOADED] {experiment}/{provider}/{model} ({len(agg_df)} task types, converted from Exp3 format)")
                    else:
                        # Standard format for Exp1/2/4
                        # Rename 'type' to 'task_type' if needed
                        if 'type' in df.columns and 'task_type' not in df.columns:
                            df = df.rename(columns={'type': 'task_type'})

                        # Rename 'pass_at_1' to 'pass' if needed for consistency
                        if 'pass_at_1' in df.columns and 'pass' not in df.columns:
                            df = df.rename(columns={'pass_at_1': 'pass'})

                        df['experiment'] = experiment
                        df['provider'] = provider
                        df['model'] = model
                        df['provider_model'] = f"{provider}/{model}"

                        all_results.append(df)
                        print(f"[LOADED] {experiment}/{provider}/{model} ({len(df)} records)")
            except Exception as e:
                print(f"[SKIP] File {csv_file}: {e}")

        if not all_results:
            print("[ERROR] No result data found")
            return pd.DataFrame()

        combined = pd.concat(all_results, ignore_index=True)
        print(f"\n[SUMMARY] Loaded {len(combined)} records, {len(combined['task_type'].unique())} task types")
        print(f"  Experiments: {sorted(combined['experiment'].unique())}")
        print(f"  Models: {sorted(combined['provider_model'].unique())}")

        return combined

    def _get_accuracy_pivot(self, experiment: str = 'exp1',
                            metric: str = 'pass') -> pd.DataFrame:
        """
        Get accuracy pivot table

        Args:
            experiment: Experiment name (exp1, exp2, exp3, exp4, 'all' for all)
            metric: Metric name (pass, attempts, time, etc.)

        Returns:
            Pivot table with task_type as rows, provider_model as columns
        """
        if self.data.empty:
            return pd.DataFrame()

        # Filter experiment
        if experiment == 'all':
            df = self.data.copy()
        else:
            df = self.data[self.data['experiment'] == experiment].copy()

        if df.empty:
            print(f"[WARNING] Experiment {experiment} has no data")
            return pd.DataFrame()

        # Create pivot table
        pivot = df.pivot_table(
            values=metric,
            index='task_type',
            columns='provider_model',
            aggfunc='mean'
        )

        # Convert to percentage
        if metric == 'pass':
            pivot = pivot * 100

        # Rename columns to display names
        pivot.columns = [self._get_display_name(col, 'model') for col in pivot.columns]

        return pivot

    def plot_heatmap(self, experiment: str = 'exp1',
                     figsize: Tuple[int, int] = (14, 10),
                     save_path: Optional[str] = None):
        """
        Plot heatmap: Task difficulty overview

        Args:
            experiment: Experiment name
            figsize: Figure size
            save_path: Save path (None=don't save)
        """
        # Base pivot: rows = task_type, columns = models, values = Pass@1 (%)
        pivot_base = self._get_accuracy_pivot(experiment)

        if pivot_base.empty:
            print(f"[WARNING] Cannot plot heatmap: {experiment} has no data")
            return None

        # 1) 按任务计算跨模型平均难度，并按平均值排序（行方向）
        pivot_tasks = pivot_base.copy()
        pivot_tasks['Average'] = pivot_tasks.mean(axis=1)
        pivot_tasks = pivot_tasks.sort_values('Average')

        # 2) 计算“Overall”行：每个模型列使用原始样本量加权的整体 Pass@1
        #    而不是简单平均各任务的百分比
        overall_vals: dict[str, float] = {}

        # 原始数据（0-1 概率 + n）子集
        df_exp = self.data[self.data['experiment'] == experiment].copy()
        if 'n' in df_exp.columns:
            df_exp['n'] = pd.to_numeric(df_exp['n'], errors='coerce').fillna(0)
        else:
            df_exp['n'] = 1

        # 映射：显示名 -> provider_model
        disp_to_pm: dict[str, str] = {}
        if not df_exp.empty:
            for pm in df_exp['provider_model'].unique():
                disp = self._get_display_name(pm, 'model')
                disp_to_pm[disp] = pm

        # 逐模型列计算加权总体 Pass@1（0–100）
        for col in pivot_tasks.columns:
            if col == 'Average':
                continue
            pm = disp_to_pm.get(col)
            if not pm:
                overall_vals[col] = float('nan')
                continue
            df_m = df_exp[df_exp['provider_model'] == pm]
            if df_m['n'].sum() <= 0:
                overall_vals[col] = float('nan')
                continue
            w_pass = (df_m['pass'] * df_m['n']).sum() / df_m['n'].sum() * 100.0
            overall_vals[col] = w_pass

        # 全局平均（所有模型 + 所有任务，加权）
        if df_exp['n'].sum() > 0:
            overall_vals['Average'] = (df_exp['pass'] * df_exp['n']).sum() / df_exp['n'].sum() * 100.0
        else:
            overall_vals['Average'] = float('nan')

        overall_row = pd.Series(overall_vals, name='Overall')
        pivot = pd.concat([pivot_tasks, overall_row.to_frame().T], axis=0)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)

        # Create custom colormap: light green (#EEF4ED) -> deep blue (#183F7F)
        from matplotlib.colors import LinearSegmentedColormap
        custom_colors = ['#EEF4ED', '#D9E6C9', '#97C8C5', '#4281B6', '#183F7F']
        custom_cmap = LinearSegmentedColormap.from_list('custom_heatmap', custom_colors)

        # Use custom color scheme (light=easy, dark blue=hard)
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap=custom_cmap,
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Pass@1 (%)'},
            linewidths=0.5,
            linecolor='gray',
            ax=ax,
            mask=pivot.isna(),  # Automatically handle missing values
            cbar=True
        )

        # Title and labels
        exp_title = self._format_exp_label(experiment)

        # ax.set_title(f'CAPTCHA Task Difficulty Heatmap\n{exp_title}',

        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('Task Type (Sorted by Difficulty)', fontsize=13, fontweight='bold')

        # Highlight recommended CAPTCHA zone (accuracy < 40%)
        for i, (task, row) in enumerate(pivot.iterrows()):
            if row['Average'] < 40:
                ax.add_patch(plt.Rectangle((len(pivot.columns)-1, i), 1, 1,
                                          fill=False, edgecolor='red', lw=3))

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            # Ensure output directory exists
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f"[SAVED] Heatmap saved: {save_path}")

        return fig

    def plot_cost_performance_frontier(self,
                                       experiments: List[str] = ('exp1', 'exp2'),
                                       model_filter: Optional[str] = None,
                                       log_x: bool = True,
                                       figsize: Tuple[int, int] = (12, 8),
                                       font_sizes: Optional[Dict[str, float]] = None,
                                       save_path: Optional[str] = None):
        """
        Cost–Performance frontier: x=cost_per_question (USD), y=Pass@1 (%).
        Overlays multiple experiments for the same model.

        Args:
            experiments: Experiments to overlay
            model_filter: Optional provider/model filter
            log_x: Whether to use log scale on the x-axis
            figsize: Figure size
            font_sizes: Optional font size overrides (keys: label, tick, legend, annotation)
            save_path: Optional PDF output path
        """
        if self.data.empty:
            print("[WARNING] Cannot plot frontier: No data")
            return None

        df = self.data.copy()
        if model_filter:
            df = df[df['provider_model'] == model_filter]
        df = df[df['experiment'].isin(experiments)]
        if df.empty:
            print("[WARNING] No data after filtering for frontier plot")
            return None

        # Require cost column
        if 'cost_per_question' not in df.columns:
            print("[WARNING] results.csv missing cost_per_question column; skip frontier plot")
            return None

        font_sizes = font_sizes or {}
        label_size = font_sizes.get('label', plt.rcParams.get('axes.labelsize', 12))
        legend_size = font_sizes.get('legend', plt.rcParams.get('legend.fontsize', 10))
        annot_size = font_sizes.get('annotation', max(6, plt.rcParams.get('font.size', 12) - 2))
        tick_size = font_sizes.get('tick')

        # Prepare figure
        fig, ax = plt.subplots(figsize=figsize)
        if tick_size is not None:
            ax.tick_params(axis='both', which='both', labelsize=tick_size)

        colors = ['#7eb3d6', '#2e75b6', '#70ad47', '#ffc000']
        exp_list = [e for e in experiments if e in set(df['experiment'])]
        for i, exp in enumerate(exp_list):
            sub = df[df['experiment'] == exp].copy()
            if sub.empty:
                continue
            sub['pass_pct'] = sub['pass'] * 100
            ax.scatter(sub['cost_per_question'], sub['pass_pct'],
                       s=80, alpha=0.8, label=self._get_display_name(exp, 'experiment'),
                       color=colors[i % len(colors)], edgecolors='black', linewidths=0.8)

            # Optional: simple Pareto frontier (maximize y, minimize x)
            pts = sub[['cost_per_question', 'pass_pct']].dropna().values
            if len(pts) >= 2:
                # Sort by cost asc, then sweep to keep increasing pass
                pts_sorted = pts[pts[:, 0].argsort()]
                frontier = []
                best_y = -1
                for x, y in pts_sorted:
                    if y > best_y:
                        frontier.append((x, y))
                        best_y = y
                if len(frontier) >= 2:
                    xs, ys = zip(*frontier)
                    ax.plot(xs, ys, color=colors[i % len(colors)], linestyle='-', alpha=0.6)

        # Add task type labels for all experiments
        # Use drop_duplicates to label each task_type only at its last (usually optimized) position
        if not df.empty:
            label_df = df.copy()
            label_df['pass_pct'] = label_df['pass'] * 100
            label_df = label_df.dropna(subset=['cost_per_question', 'pass_pct'])
            for _, row in label_df.iterrows():
                color = '#c0392b' if row['task_type'] in self.HARD_TASKS else '#666666'
                ax.annotate(
                    row['task_type'],
                    xy=(row['cost_per_question'], row['pass_pct']),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=annot_size, alpha=0.75, color=color
                )

        ax.axhline(y=40, color='red', linestyle='--', linewidth=2, alpha=0.5, label='CAPTCHA Threshold (40%)')
        ax.set_ylabel('Pass@1 (%)', fontsize=label_size, fontweight='bold')
        ax.set_xlabel('Cost per Question (USD)', fontsize=label_size, fontweight='bold')
        ax.set_ylim(0, 105)
        if log_x:
            ax.set_xscale('log')
        ax.grid(alpha=0.3, which='both', axis='both')


        #     title += f"\nModel: {self._get_display_name(model_filter, 'model')}"

        # ax.set_title(title, fontsize=title_size, fontweight='bold', pad=16)
        ax.legend(loc='lower left', fontsize=legend_size, framealpha=0.9)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f"[SAVED] Frontier saved: {save_path}")
        return fig

    def plot_time_performance_scatter(self,
                                       experiments: List[str] = ('exp1', 'exp2'),
                                       model_filter: Optional[str] = None,
                                       metric: str = 'avg_e2e_ms',
                                       figsize: Tuple[int, int] = (12, 8),
                                       font_sizes: Optional[Dict[str, float]] = None,
                                       save_path: Optional[str] = None):
        """
        Time–Performance scatter: x=avg_e2e_ms, y=Pass@1 (%). Overlays experiments.

        Args:
            experiments: Experiments to overlay
            model_filter: Optional provider/model filter
            metric: Column name for x-axis timing metric
            figsize: Figure size
            font_sizes: Optional font size overrides (keys: label, tick, legend, annotation)
            save_path: Optional PDF output path
        """
        if self.data.empty:
            print("[WARNING] Cannot plot time-performance: No data")
            return None

        if metric not in self.data.columns:
            print(f"[WARNING] Column '{metric}' not found; skip time-performance plot")
            return None

        font_sizes = font_sizes or {}
        label_size = font_sizes.get('label', plt.rcParams.get('axes.labelsize', 12))
        legend_size = font_sizes.get('legend', plt.rcParams.get('legend.fontsize', 10))
        annot_size = font_sizes.get('annotation', max(6, plt.rcParams.get('font.size', 12) - 2))
        tick_size = font_sizes.get('tick')

        df = self.data.copy()
        if model_filter:
            df = df[df['provider_model'] == model_filter]
        df = df[df['experiment'].isin(experiments)]
        if df.empty:
            print("[WARNING] No data after filtering for time-performance plot")
            return None

        fig, ax = plt.subplots(figsize=figsize)
        if tick_size is not None:
            ax.tick_params(axis='both', which='both', labelsize=tick_size)
        colors = ['#7eb3d6', '#2e75b6', '#70ad47', '#ffc000']
        exp_list = [e for e in experiments if e in set(df['experiment'])]

        texts = []
        for i, exp in enumerate(exp_list):
            sub = df[df['experiment'] == exp].copy()
            if sub.empty:
                continue
            sub = sub.dropna(subset=[metric])
            sub['pass_pct'] = sub['pass'] * 100
            # Convert milliseconds to seconds for unified units (s)
            ax.scatter(sub[metric] / 1000.0, sub['pass_pct'], s=80, alpha=0.85,
                       label=self._get_display_name(exp, 'experiment'),
                       color=colors[i % len(colors)], edgecolors='black', linewidths=0.8)
            # Annotate every point (keep both Exp1/Exp2 labels)
            for _, row in sub.iterrows():
                color = '#c0392b' if row['task_type'] in self.HARD_TASKS else '#666666'
                texts.append(ax.annotate(
                    row['task_type'],
                    xy=(row[metric] / 1000.0, row['pass_pct']),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=annot_size, alpha=0.75, color=color
                ))

        ax.axhline(y=40, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_ylabel('Pass@1 (%)', fontsize=label_size, fontweight='regular')
        ax.set_xlabel('Average E2E Time (s)', fontsize=label_size, fontweight='regular')
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)


        #     title += f"\nModel: {self._get_display_name(model_filter, 'model')}"

        # ax.set_title(title, fontsize=title_size, fontweight='bold', pad=16)
        ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1), fontsize=legend_size, framealpha=0.9)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f"[SAVED] Time–Performance saved: {save_path}")
        return fig


    def plot_comparison_bars(self, experiments: List[str] = ['exp1', 'exp2', 'exp4'],
                            model_filter: Optional[str] = None,
                            figsize: Tuple[int, int] = (18, 7),
                            save_path: Optional[str] = None):
        """
        Plot grouped bar chart: Optimization effect comparison

        Args:
            experiments: List of experiments to compare
            model_filter: Only show specific model (e.g., "openai/gpt-5-chat-latest")
            figsize: Figure size
            save_path: Save path
        """
        if self.data.empty:
            print("[WARNING] Cannot plot bar chart: No data")
            return None

        # Filter data
        df = self.data[self.data['experiment'].isin(experiments)].copy()
        if model_filter:
            df = df[df['provider_model'] == model_filter]

        if df.empty:
            print("[WARNING] No data after filtering")
            return None

        # Calculate average accuracy for each task type in each experiment
        grouped = df.groupby(['task_type', 'experiment'])['pass'].mean() * 100
        pivot = grouped.unstack(fill_value=np.nan)

        # Sort by difficulty of first experiment
        if experiments[0] in pivot.columns:
            pivot = pivot.sort_values(experiments[0])

        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(pivot))
        width = 0.8 / len(experiments)
        colors = ['#7eb3d6', '#2e75b6', '#70ad47', '#ffc000']

        for i, exp in enumerate(experiments):
            if exp in pivot.columns:
                offset = (i - len(experiments)/2) * width + width/2
                exp_label = self._get_display_name(exp, 'experiment')
                bars = ax.bar(x + offset, pivot[exp], width,
                             label=exp_label, color=colors[i % len(colors)],
                             alpha=0.9)

                label_size = plt.rcParams.get('legend.fontsize', 10)
                # Add value labels on bars (only for valid data)
                for j, (idx, val) in enumerate(pivot[exp].items()):
                    if not np.isnan(val):
                        ax.text(x[j] + offset, val + 2, f'{val:.0f}',
                               ha='center', va='bottom', fontsize=label_size)

        # Add CAPTCHA recommendation threshold line
        ax.axhline(y=40, color='red', linestyle='--', linewidth=2,
                  label='CAPTCHA Threshold (40%)', alpha=0.7)

        label_size = plt.rcParams.get('axes.labelsize', 13)
        title_size = plt.rcParams.get('axes.titlesize', 16)
        legend_size = plt.rcParams.get('legend.fontsize', 11)
        ax.set_ylabel('Pass@1 (%)', fontsize=label_size, fontweight='bold')
        ax.set_xlabel('Task Type (Sorted by Difficulty)', fontsize=label_size, fontweight='bold')


        #     title += f'\nModel: {model_display}'
        # ax.set_title(title, fontsize=title_size, fontweight='bold', pad=20)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=legend_size, framealpha=0.9)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            # Ensure output directory exists
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f"[SAVED] Bar chart saved: {save_path}")

        return fig

    def plot_optimization_resistance(self, base_exp: str = 'exp1',
                                    opt_exp: str = 'exp2',
                                    model_filter: Optional[str] = None,
                                    figsize: Tuple[int, int] = (11, 11),
                                    save_path: Optional[str] = None,
                                    use_adjust_text: bool = False):
        """
        Plot scatter plot: Optimization resistance analysis

        Args:
            base_exp: Baseline experiment
            opt_exp: Optimized experiment
            model_filter: Only show specific model
            figsize: Figure size
            save_path: Save path
        """
        if self.data.empty:
            print("[WARNING] Cannot plot scatter plot: No data")
            return None

        # Get data from both experiments
        df = self.data.copy()
        if model_filter:
            df = df[df['provider_model'] == model_filter]

        base_data = df[df['experiment'] == base_exp].groupby('task_type')['pass'].mean() * 100
        opt_data = df[df['experiment'] == opt_exp].groupby('task_type')['pass'].mean() * 100

        # Merge data
        comparison = pd.DataFrame({
            'baseline': base_data,
            'optimized': opt_data
        }).dropna()

        if comparison.empty:
            print(f"[WARNING] {base_exp} and {opt_exp} have no overlapping task types")
            return None

        # Plot scatter plot
        fig, ax = plt.subplots(figsize=figsize)

        # Assign colors based on position (threshold = 40%)
        colors = []
        for idx, row in comparison.iterrows():
            if row['baseline'] < 40 and row['optimized'] < 40:
                colors.append('#d32f2f')  # Red: Recommended CAPTCHA zone
            elif row['baseline'] < 40 and row['optimized'] >= 40:
                colors.append('#ffa726')  # Orange: Significant improvement
            elif row['baseline'] >= 40 and row['optimized'] < 40:
                colors.append('#7e57c2')  # Purple: Abnormal degradation
            else:
                colors.append('#78909c')  # Gray: Easy task type

        # Plot scatter points
        scatter = ax.scatter(comparison['baseline'], comparison['optimized'],
                           s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)

        # Add task labels; prefer using adjustText to reduce overlaps if available
        texts = []
        for idx, row in comparison.iterrows():
            color = '#c0392b' if idx in self.HARD_TASKS else '#666666'
            t = ax.text(row['baseline'], row['optimized'], idx,
                        fontsize=9, alpha=0.9, color=color)
            texts.append(t)

        if use_adjust_text and adjust_text is not None and len(texts) > 0:
            try:
                adjust_text(
                    texts,
                    ax=ax,
                    only_move={'points': 'y', 'texts': 'xy'},
                    autoalign='y',
                    force_points=0.2,
                    force_text=0.2,
                    expand_text=(1.05, 1.2),
                    expand_points=(1.1, 1.3),
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6)
                )
            except Exception as e:
                print(f"[WARNING] adjustText failed: {e}. Falling back to static labels.")
        elif use_adjust_text and adjust_text is None:
            print("[INFO] 'adjustText' not installed. Run 'pip install adjustText' to enable automatic label adjustment.")

        # Diagonal line (y=x)
        lims = [0, 105]
        ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1.5, label='No Change (y=x)')

        # CAPTCHA recommendation zone (red box, threshold = 40%)
        ax.axhline(y=40, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(x=40, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.fill_between([0, 40], 0, 40, alpha=0.1, color='red',
                       label='Recommended CAPTCHA Zone')

        # Labels and title
        base_exp_display = self._get_display_name(base_exp, 'experiment')
        opt_exp_display = self._get_display_name(opt_exp, 'experiment')

        ax.set_xlabel(f'{base_exp_display} Pass@1 (%)',
                     fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{opt_exp_display} Pass@1 (%)',
                     fontsize=13, fontweight='bold')


        #     title += f'\nModel: {model_display}'
        # ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Anchor origin at (0,0) so axes start from zero
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        # Legend for reference lines/zones (single legend, placed at lower right)
        guide_handles = [
            Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, alpha=0.4,
                   label='No Change (y=x)'),
            Patch(facecolor='red', alpha=0.1, edgecolor='red', linestyle='--',
                  label='Recommended CAPTCHA Zone')
        ]
        ax.legend(handles=guide_handles, loc='lower right', fontsize=11, framealpha=0.9)

        plt.tight_layout()

        if save_path:
            # Ensure output directory exists
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f"[SAVED] Scatter plot saved: {save_path}")

        return fig

    def plot_cross_model_stability(self, experiment: str = 'exp1',
                                   figsize: Tuple[int, int] = (16, 8),
                                   save_path: Optional[str] = None):
        """
        Plot box plot: Cross-model stability analysis

        Args:
            experiment: Experiment name
            figsize: Figure size
            save_path: Save path
        """
        if self.data.empty:
            print("[WARNING] Cannot plot box plot: No data")
            return None

        # Filter experiment data
        df = self.data[self.data['experiment'] == experiment].copy()
        if df.empty:
            print(f"[WARNING] Experiment {experiment} has no data")
            return None

        # Calculate statistics for each task type
        df['accuracy'] = df['pass'] * 100

        # Sort task types by median
        median_acc = df.groupby('task_type')['accuracy'].median().sort_values()
        sorted_tasks = median_acc.index.tolist()

        # Plot box plot
        fig, ax = plt.subplots(figsize=figsize)

        # Prepare data
        data_to_plot = [df[df['task_type'] == task]['accuracy'].values
                       for task in sorted_tasks]

        # Set colors based on median
        colors = ['#ef5350' if median_acc[task] < 40 else '#90a4ae'
                 for task in sorted_tasks]

        bp = ax.boxplot(data_to_plot,
                       labels=sorted_tasks,
                       patch_artist=True,
                       widths=0.6,
                       showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='yellow',
                                    markersize=6, markeredgecolor='black'))

        # Set box colors
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # CAPTCHA threshold line
        ax.axhline(y=40, color='red', linestyle='--', linewidth=2,
                  label='CAPTCHA Threshold (40%)', alpha=0.7)

        ax.set_ylabel('Pass@1 (%) across Models', fontsize=13, fontweight='bold')
        ax.set_xlabel('Task Type (Sorted by Median Difficulty)',
                     fontsize=13, fontweight='bold')


        # ax.set_title(f'Cross-Model Stability Analysis\n{exp_line}',


        plt.xticks(rotation=45, ha='right')
        ax.set_ylim(-5, 105)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=11)

        plt.tight_layout()

        if save_path:
            # Ensure output directory exists
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f"[SAVED] Box plot saved: {save_path}")

        return fig


    def generate_captcha_recommendation(self, experiment: str = 'exp2',
                                       threshold: float = 40.0,
                                       top_n: int = 8) -> pd.DataFrame:
        """
        Generate CAPTCHA recommended task type list

        Args:
            experiment: Experiment for evaluation
            threshold: CAPTCHA recommendation threshold
            top_n: Return top N hardest task types

        Returns:
            Recommended task types DataFrame
        """
        if self.data.empty:
            print("[WARNING] No data")
            return pd.DataFrame()

        # Get experiment data
        df = self.data[self.data['experiment'] == experiment].copy()

        # Calculate statistics for each task type
        stats = df.groupby('task_type').agg({
            'pass': ['mean', 'std', 'count']
        }).round(4)

        stats.columns = ['avg_pass@1', 'std_pass@1', 'n_models']
        stats['avg_pass@1'] *= 100
        stats['std_pass@1'] *= 100

        # Filter difficult task types
        difficult_tasks = stats[stats['avg_pass@1'] < threshold].copy()
        difficult_tasks = difficult_tasks.sort_values('avg_pass@1')

        # Add recommendation score
        difficult_tasks['captcha_score'] = (
            (100 - difficult_tasks['avg_pass@1']) * 0.6 +  # Difficulty weight
            (100 - difficult_tasks['std_pass@1']) * 0.4     # Stability weight
        )

        difficult_tasks = difficult_tasks.sort_values('captcha_score', ascending=False)

        result = difficult_tasks.head(top_n).copy()
        result['rank'] = range(1, len(result) + 1)

        return result[['rank', 'avg_pass@1', 'std_pass@1', 'n_models', 'captcha_score']]

    def plot_all(self, output_dir: str = "./figures"):
        """
        Generate all charts and save

        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        print(f"\nGenerating charts, saving to: {output_dir}/\n")

        # Detect available experiments
        available_exps = sorted(self.data['experiment'].unique()) if not self.data.empty else []
        available_models = sorted(self.data['provider_model'].unique()) if not self.data.empty else []

        print(f"[DETECTED] Experiments: {available_exps}")
        print(f"[DETECTED] Models: {available_models}\n")

        figures = []

        # Filter out Exp4 from global plotting (focus on Exp1/Exp2 baselines and optional Exp3)
        baseline_exps = [e for e in available_exps if e in ('exp1', 'exp2', 'exp3')]

        # 1. Heatmap (for each remaining experiment)
        for exp in baseline_exps:
            try:
                fig = self.plot_heatmap(
                    experiment=exp,
                    save_path=str(output_path / f"heatmap_{exp}.pdf")
                )
                if fig:
                    figures.append(fig)
                    plt.close(fig)
            except Exception as e:
                print(f"[WARNING] Heatmap {exp} generation failed: {e}")

        # 2. Grouped bar chart (compare experiments, baseline only uses Exp1/Exp2/Exp4)
        compare_exps = [e for e in baseline_exps if e in ('exp1', 'exp2')]
        if len(compare_exps) >= 2:
            for model in available_models:
                try:
                    fig = self.plot_comparison_bars(
                        experiments=compare_exps,
                        model_filter=model,
                        save_path=str(output_path / f"comparison_{model.replace('/', '_')}.pdf")
                    )
                    if fig:
                        figures.append(fig)
                        plt.close(fig)
                except Exception as e:
                    print(f"[WARNING] Bar chart {model} generation failed: {e}")

        # 3. Scatter plot (exp1 vs exp2)
        if 'exp1' in baseline_exps and 'exp2' in baseline_exps:
            for model in available_models:
                try:
                    fig = self.plot_optimization_resistance(
                        base_exp='exp1',
                        opt_exp='exp2',
                        model_filter=model,
                        save_path=str(output_path / f"resistance_{model.replace('/', '_')}.pdf")
                    )
                    if fig:
                        figures.append(fig)
                        plt.close(fig)
                except Exception as e:
                    print(f"[WARNING] Scatter plot {model} generation failed: {e}")

        # 4. Box plot (cross-model stability)
        for exp in available_exps:
            if len(self.data[self.data['experiment'] == exp]['provider_model'].unique()) >= 2:
                try:
                    fig = self.plot_cross_model_stability(
                        experiment=exp,
                        save_path=str(output_path / f"stability_{exp}.pdf")
                    )
                    if fig:
                        figures.append(fig)
                        plt.close(fig)
                except Exception as e:
                    print(f"[WARNING] Box plot {exp} generation failed: {e}")

        # 5. Cost–Performance frontier per model (if cost columns present)
        for model in available_models:
            try:
                fig = self.plot_cost_performance_frontier(
                    experiments=[e for e in available_exps if e in ('exp1', 'exp2', 'exp4')],
                    model_filter=model,
                    save_path=str(output_path / f"frontier_{model.replace('/', '_')}.pdf")
                )
                if fig:
                    figures.append(fig)
                    plt.close(fig)
            except Exception as e:
                print(f"[WARNING] Frontier {model} generation failed: {e}")

        # 6. Time–Performance scatter per model (baseline experiments only)
        for model in available_models:
            try:
                fig = self.plot_time_performance_scatter(
                    experiments=[e for e in available_exps if e in ('exp1', 'exp2', 'exp4')],
                    model_filter=model,
                    save_path=str(output_path / f"time_perf_{model.replace('/', '_')}.pdf")
                )
                if fig:
                    figures.append(fig)
                    plt.close(fig)
            except Exception as e:
                print(f"[WARNING] Time–Performance {model} generation failed: {e}")

        print(f"\n[COMPLETED] Chart generation finished! Total {len(list(output_path.glob('*.pdf')))} PDF files")
        print(f"[OUTPUT] Save location: {output_path.absolute()}")

        return figures


# Convenience Functions


def quick_visualize(results_dir: str = "./results",
                   output_dir: str = "./figures",
                   show: bool = False,
                   exclude_models: Optional[List[str]] = None):
    """
    Quickly generate all visualization charts

    Args:
        results_dir: Results directory
        output_dir: Output directory
        show: Whether to display charts (recommend False in Jupyter)
        exclude_models: Optional list of provider/model identifiers to drop (e.g., ["openai/gpt-5-chat-latest"])
    """
    viz = CAPTCHAVisualizer(
        results_dir=results_dir,
        exclude_models=exclude_models or ["openai/gpt-5-chat-latest"]
    )

    if viz.data.empty:
        print("[ERROR] No experiment data found")
        return None

    # Generate all charts
    viz.plot_all(output_dir=output_dir)

    # Generate recommendation report
    print("\n" + "="*60)
    print("CAPTCHA Recommended Task Types (Based on Exp2 Optimized Results)")
    print("="*60)

    if 'exp2' in viz.data['experiment'].values:
        recommendations = viz.generate_captcha_recommendation(
            experiment='exp2',
            threshold=40.0,
            top_n=8
        )
        if not recommendations.empty:
            print(recommendations.to_string())

            # Save recommendation list
            rec_path = Path(output_dir) / "captcha_recommendations.csv"
            recommendations.to_csv(rec_path)
            print(f"\n[SAVED] Recommendation list saved: {rec_path}")
    else:
        print("[WARNING] Exp2 not run yet, cannot generate recommendations")

    print("\n[COMPLETED] Visualization finished!")

    if not show:
        plt.close('all')

    return viz


# Example Usage


if __name__ == "__main__":
    # Quickly generate all charts
    viz = quick_visualize(
        results_dir="./results",
        output_dir="./figures",
        show=False
    )
