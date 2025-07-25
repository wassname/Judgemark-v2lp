import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict
from scipy.stats import linregress
from scipy.stats import spearmanr, theilslopes
from judgemark_v2lp.config.constants import NEGATIVE_MARKERS, MODEL_NAME_REPLACEMENTS


def create_side_by_side_score_charts(run_data: Dict, judge_model: str, samples_data: Dict, method: str = "raw"):
    """
    Produces two figures:
      • Figure #1 with three subplots side-by-side:
        (1) Raw Scores bar chart (+ 95% CI)
        (2) Calibrated Scores bar chart (+ 95% CI)
        (3) Heatmap of all per-criterion scores across each model (10 - score for negative markers).
      • Figure #2: A 4×4 grid of mini scatter plots, one per model (up to 16),
        showing item length (chars) vs. aggregated_score_raw. A linear regression
        line and correlation stats are included for each model if enough points exist.
    """
    # -------------------------------------------------------------------
    # 1) The main (raw / calibrated / heatmap) figure
    # -------------------------------------------------------------------
    raw_stats = run_data["raw_model_stats"]
    cal_stats = run_data["calibrated_model_stats"]

    if judge_model in MODEL_NAME_REPLACEMENTS:
        judge_model = MODEL_NAME_REPLACEMENTS[judge_model]
    
    # All model names in raw_stats
    model_names = list(raw_stats.keys())

    # Convert to arrays for sorting
    raw_means = [raw_stats[m]["mean"] for m in model_names]
    cal_means = [cal_stats[m]["mean"] for m in model_names]
    raw_cis   = [raw_stats[m]["ci95"] for m in model_names]
    cal_cis   = [cal_stats[m]["ci95"] for m in model_names]
    
    # Sort by calibrated score descending
    sorted_indices = np.argsort(cal_means)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    raw_means   = [raw_means[i]   for i in sorted_indices]
    cal_means   = [cal_means[i]   for i in sorted_indices]
    raw_cis     = [raw_cis[i]     for i in sorted_indices]
    cal_cis     = [cal_cis[i]     for i in sorted_indices]
    
    # 1.A) Build data for the heatmap: per-criterion scores (with negative flips)
    #     We gather them from run_data "parsed_scores"
    #     Then we convert them to 0..10 bins and store percentage distribution.
    all_scores_by_model = {m: [] for m in model_names}
    results = run_data.get("results", {})
    
    for model_name in model_names:
        iter_dict = results.get(model_name, {})
        for iteration_key, item_dict in iter_dict.items():
            if not isinstance(item_dict, dict):
                continue
            for item_id, item_info in item_dict.items():
                if not isinstance(item_info, dict):
                    continue
                parsed_scores = item_info.get("parsed_scores", {})
                if not isinstance(parsed_scores, dict):
                    continue
                for crit_name, val in parsed_scores.items():
                    if isinstance(val, (int, float)) and 0 <= val <= 10:
                        crit_lower = crit_name.strip().lower()
                        # Flip negative
                        if any(nm in crit_lower for nm in NEGATIVE_MARKERS):
                            final_val = 10 - val
                        else:
                            final_val = val
                        all_scores_by_model[model_name].append(final_val)
    
    # Convert to a 2D array for the heatmap (rows = models, columns = bins)
    bins = np.linspace(0, 10, 11)
    heatmap_rows = []
    for m in model_names:
        scores = all_scores_by_model[m]
        if scores:
            counts, _ = np.histogram(scores, bins=bins)
            pct = (counts / len(scores)) * 100.0
        else:
            pct = np.zeros(len(bins)-1, dtype=float)
        heatmap_rows.append(pct)
    heatmap_data = np.array(heatmap_rows, dtype=float)
    
    # 1.B) Plot the main figure with 3 subplots
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))  # Increased height

    # Increase the base font size for all text elements
    plt.rcParams.update({'font.size': 14})
    
    # (A) Raw bar chart    
    y_pos = np.arange(len(model_names))
    ax1.barh(y_pos, raw_means, color='skyblue', alpha=0.7)
    for i, (mean_val, ci95) in enumerate(zip(raw_means, raw_cis)):
        ax1.errorbar(mean_val, i, xerr=ci95, color='red', capsize=5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(model_names, fontsize=12)
    ax1.invert_yaxis()
    ax1.set_xlabel("Raw Scores", fontsize=14)
    ax1.set_title("Raw Model Scores (95% CI)", fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', pad=10)

    # (B) Calibrated bar chart
    ax2.barh(y_pos, cal_means, color='lightgreen', alpha=0.7)
    for i, (mean_val, ci95) in enumerate(zip(cal_means, cal_cis)):
        ax2.errorbar(mean_val, i, xerr=ci95, color='red', capsize=5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_names, fontsize=12)
    ax2.invert_yaxis()
    ax2.set_xlabel("Calibrated Scores", fontsize=14)
    ax2.set_title("Calibrated Model Scores (95% CI)", fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', pad=10)

    # (C) Heatmap
    bin_edges = bins[:-1]
    ax3.set_xticks(np.arange(len(bin_edges)))
    ax3.set_xticklabels([str(int(be)) for be in bin_edges])
    im = ax3.imshow(heatmap_data, aspect='auto', origin='upper', cmap='plasma')
    #ax3.set_xticks(np.arange(len(bin_centers)))
    #ax3.set_xticklabels([f"{bc:.0f}" for bc in bin_centers], fontsize=12)
    ax3.set_yticks(np.arange(len(model_names)))
    ax3.set_yticklabels(model_names, fontsize=12)
    ax3.set_xlabel("Score Bin (0–10)", fontsize=14)
    ax3.set_title("Per-Criterion Score Distribution (Heatmap)", fontsize=16)
    ax3.tick_params(axis='y', pad=10)
    
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label("% of Criteria in Bin", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter = ticker.PercentFormatter(decimals=1)
    cbar.update_ticks()

    # Overall title
    
    sanitized_judge = re.sub(r"[^\w\-]", "-", judge_model.replace("/", "__"))
    fig1.suptitle(f"Judgemark: Raw/Calibrated/Heatmap - Judge: {judge_model}", fontsize=20)
    
    # Adjust layout with more space
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if plt.get_backend() == "inline":
        # In Jupyter, we show the figure inline
        plt.show()
    else:
        plt.savefig(f"results/charts/judgemark_3chart_{method}_{sanitized_judge}.png", 
                    bbox_inches='tight', 
                    dpi=150,
                    pad_inches=0.5)
        plt.close(fig1)

    # -------------------------------------------------------------------
    # 2) Second Figure: A 4×4 grid of scatter plots (per-model), 
    #    where X = item length, Y = aggregated_score_raw.
    #    We'll gather text straight from samples_data, compute len(...).
    # -------------------------------------------------------------------
    # If you want exactly 16 models, skip any extra or exclude some.
    # Example: exclude "gemini-1.5-pro-001" 
    excluded_models = {"gemini-1.5-pro-001"}
    model_list_for_scatter = [m for m in model_names if m not in excluded_models]
    
    # If it's still longer than 16, slice it to 16
    if len(model_list_for_scatter) > 16:
        model_list_for_scatter = model_list_for_scatter[:16]

    # Build figure and subplots: 4x4
    fig2, axes2 = plt.subplots(4, 4, figsize=(20, 20))  # each cell is a scatter
    fig2.suptitle(f"Judgemark: Per-Model Length vs. Score - Judge: {judge_model}", fontsize=18)
    
    # We might have fewer than 16 models. We'll track them by row & col.
    for idx, mname in enumerate(model_list_for_scatter):
        row = idx // 4
        col = idx % 4
        ax = axes2[row, col]
        
        # Collect all (length, raw_score) for this model
        length_vals = []
        score_vals  = []
        
        # For each (iteration_key, item_id), find the text in samples_data,
        # find aggregated_score_raw in run_data, then store pairs
        model_res = run_data["results"].get(mname, {})
        for it_key, it_dict in model_res.items():
            if not isinstance(it_dict, dict):
                continue
            for item_id, item_info in it_dict.items():
                if not isinstance(item_info, dict):
                    continue
                raw_score = item_info.get("aggregated_score_raw", None)
                if not isinstance(raw_score, (int, float)):
                    continue

                # Look up the text in samples_data:
                text = (samples_data
                        .get(mname, {})
                        .get("samples", {})
                        .get(it_key, {})
                        .get(item_id, "")) 
                text_len = len(text)

                # If it's non-empty text
                if text_len > 0:
                    length_vals.append(text_len)
                    score_vals.append(raw_score)
        
        ax.set_title(mname, fontsize=12)
        ax.set_xlabel("Length")
        ax.set_ylabel("Raw Score")
        
        if len(length_vals) > 1:
            ax.scatter(length_vals, score_vals, alpha=0.4, color='blue')
            
            # -- Rank-based correlation (Spearman) --
            rho, p_value = spearmanr(length_vals, score_vals)
            
            # -- Robust linear fit (Theil-Sen) --
            # returns slope, intercept, lower_slope, upper_slope
            slope, intercept, lo_slope, hi_slope = theilslopes(score_vals, length_vals, alpha=0.95)
            
            # Build the line
            xline = np.linspace(min(length_vals), max(length_vals), 200)
            yline = slope * xline + intercept
            ax.plot(xline, yline, color='red', linewidth=2,
                    label=f"Spearman ρ={rho:.2f}, p={p_value:.2g}")
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "Not enough data",
                    ha='center', va='center', color='red', transform=ax.transAxes)
    
    # If we have fewer than 16 models, blank out remaining subplots
    total_subplots = 16
    for i in range(len(model_list_for_scatter), total_subplots):
        row = i // 4
        col = i % 4
        axes2[row, col].axis("off")
    
    plt.tight_layout()
    if plt.get_backend() == "inline":
        # In Jupyter, we show the figure inline
        plt.show()
    else:
        plt.savefig(f"results/charts/judgemark_scattergrid_{method}_{sanitized_judge}.png", bbox_inches='tight', dpi=200)
        plt.close(fig2)
