import math
import statistics
import numpy as np
import scipy.stats
from scipy.stats import kendalltau
from loguru import logger
from typing import Dict, List
import re
from judgemark_v2lp.config.constants import REFERENCE_MODEL_SCORES
from judgemark_v2lp.utils.stats import normalize

def detokenize(tokens: List[str]) -> str:
    """
    Very rough undo for common Hugging-Face / SentencePiece / BPE markers:
      - GPT-2: 'Ġ' prefix → leading space
      - SentencePiece: '▁' prefix → leading space
      - newline marker: 'Ċ' prefix → space
      - BERT WordPieces: '##' prefix → no space, just glue on
    Falls back to stripping any leading run of those markers, then
    inserting a single space before each token that isn't a '##' piece.
    """
    out = ""
    for t in tokens:
        if "Ċ" in t:
            t = t.replace("Ċ", "\n")
        
        if t.startswith("##"):
            # BERT-style subword: glue to previous
            out += t[2:]
        elif t.startswith("Ġ") or t.startswith("▁"):
            # GPT-2 or SentencePiece: prefix with space
            if out and not out.endswith(" "):
                out += " "
            out += t[1:]
        else:
            # Normal token: just append it
            out += t
    return out.lstrip()

def parse_scores(judge_model_response: str, logprobs: list) -> Dict[str,float]:
    """
    Extracts zero or more named numeric scores from a text using a simple Regex pattern:

      <metric name>: <score>

    The metric name can be any string without newlines or colons.
    The score can be a positive or negative float or integer.
    Example lines in the judge output might be:
      "Realism Score: 7.5"
      "Melodramatic: 2"
    """
    scores: Dict[str, float] = {}
    logps ={}
    pattern = r"(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)"
    choices = [str(i) for i in range(11)]
    window_size = 20
    # Look for lines or statements like "Something: 3.5" or "Something Score 3.5"
    for ti, row in enumerate(logprobs):
        if row['token'] in choices:
            # get previous window (incl) to check for regexp
            prev_window = [t['token'] for t in logprobs[max(0, ti-window_size):ti+1]]
            prev_text = detokenize(prev_window)
            matches = re.findall(pattern, prev_text)
            if matches:
                match = matches[-1]  # take the last match in the window
                metric_name = match[0].strip()
                numeric_val = float(match[1])

                scores[metric_name] = numeric_val

                logp_dict = {t['token']:t['logprob'] for t in row['top_logprobs']}
                logp_arr = [logp_dict.get(c, -100) for c in choices]

                # extra logprob of choices
                logps[metric_name] = logp_arr

    return scores, logps

def compute_raw_score(scores: Dict[str,float]) -> float:
    """
    Given a dict of {criteria: numeric score}, compute a single raw score by adjusting 
    negative-themed criteria by inverting them, then normalizing to 0-10 scale.
    """
    valid_scores = {k: v for k, v in scores.items() if 0 <= v <= 10}
    
    if len(valid_scores) < 10:
        return None
    
    negative_markers = [
        "melodramatic", "shallow resolution", "unearned resolution",
        "simplistic moralizing", "shallow optimism", "forced optimism", 
        "trite", "overwrought", "amateurish", "contrived", "uninspiring",
        "characters are too good", "incongruent ending positivity",
        "unearned transformations", "profundity over-reach",
        "amateurish descriptives", "clunky asides", "stilted dialogue",
        "tit-for-tat dialogue", "purple prose", "uncreative", "tell-don't-show",
        "weak dialogue", "meandering"
    ]
    
    sum_val = 0.0
    for criteria, val in valid_scores.items():
        crit_lower = criteria.lower().strip()
        if any(neg in crit_lower for neg in negative_markers):
            sum_val += (10 - val)
        else:
            sum_val += val
    
    avg_val = sum_val / len(valid_scores)
    return round(avg_val, 2)

def confidence_interval_95(data: List[float]) -> float:
    """
    Computes the 95% confidence interval for the mean using normal approximations:
    CI95 = 1.96 * (std / sqrt(n)), for n>30 or so.
    """
    n = len(data)
    if n < 2:
        return 0.0
    mean_ = statistics.mean(data)
    stdev_ = statistics.pstdev(data) if n == 1 else statistics.stdev(data)
    ci95 = 1.96 * (stdev_ / math.sqrt(n))
    return ci95

def compute_detailed_distribution(scores):
    if not scores:
        return {}
    return {
        "count": len(scores),
        "min": round(min(scores), 3),
        "max": round(max(scores), 3),
        "mean": round(statistics.mean(scores), 3),
        "median": round(statistics.median(scores), 3),
        "stdev": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 3),
        "p10": round(float(np.percentile(scores, 10)), 3),
        "p25": round(float(np.percentile(scores, 25)), 3),
        "p75": round(float(np.percentile(scores, 75)), 3),
        "p90": round(float(np.percentile(scores, 90)), 3)
    }

def compute_model_level_stats(scores_by_model, lengths_by_model):
    model_stats = {}
    for model_name, scores in scores_by_model.items():
        lengths = lengths_by_model[model_name]
        stats = {
            "count": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "ci95": confidence_interval_95(scores),
            "min": min(scores),
            "max": max(scores)
        }
        
        # Length correlation
        if len(lengths) == len(scores):
            corr, _ = scipy.stats.pearsonr(lengths, scores)
            stats["length_correlation"] = corr
        
        model_stats[model_name] = stats
    return model_stats

def compute_cross_model_stats(scores_by_model_all, scores_by_model_by_iter):
    """
    Computes cross-model stats. ANOVA, Kruskal, and std_dev_across_models
    are calculated over all scores. Pearson/Kendall correlations are now
    computed on a per-iteration basis and then averaged.
    """
    # ANOVA/Kruskal portion remains over ALL model scores
    arrays = list(scores_by_model_all.values())
    f_stat, f_p = scipy.stats.f_oneway(*arrays)
    kw_stat, kw_p = scipy.stats.kruskal(*arrays)

    # Standard deviation across model means (over entire distribution)
    model_means = [statistics.mean(scores) for scores in arrays]
    std_across_models = statistics.pstdev(model_means)

    # --------------------
    # Compute correlation stats per iteration, then average
    # --------------------
    iteration_keys = set()
    for model, by_iter_dict in scores_by_model_by_iter.items():
        iteration_keys |= set(by_iter_dict.keys())  # union of all iteration keys

    iteration_pearsons = []
    iteration_kendalls = []
    
    for it_key in iteration_keys:
        # Gather means for each model in this iteration
        model_means_dict = {}
        for model, by_iter_dict in scores_by_model_by_iter.items():
            if it_key in by_iter_dict and len(by_iter_dict[it_key]) > 0:
                model_means_dict[model] = statistics.mean(by_iter_dict[it_key])
        
        # Pair each model's iteration-mean with reference score
        ref_pairs = []
        for m, mean_val in model_means_dict.items():
            if m in REFERENCE_MODEL_SCORES:
                ref_pairs.append((mean_val, REFERENCE_MODEL_SCORES[m]))
        
        # If enough models exist for correlation, compute it
        if len(ref_pairs) >= 2:
            means, refs = zip(*ref_pairs)
            p_r, _ = scipy.stats.pearsonr(means, refs)
            k_tau, _ = scipy.stats.kendalltau(means, refs)
        else:
            p_r, k_tau = (0.0, 0.0)
        
        iteration_pearsons.append(p_r)
        iteration_kendalls.append(k_tau)
    
    # Final correlation = average across iteration-level correlations
    if iteration_pearsons:
        pearson_r = statistics.mean(iteration_pearsons)
    else:
        pearson_r = 0.0

    if iteration_kendalls:
        kendall_tau = statistics.mean(iteration_kendalls)
    else:
        kendall_tau = 0.0

    return {
        "anova_f": f_stat,
        "anova_p": f_p,
        "kw_stat": kw_stat,
        "kw_p": kw_p,
        "std_dev_across_models": std_across_models,
        "pearson_r": pearson_r,
        "kendall_tau": kendall_tau,
        "normalized_components": {
            "pearson_r": normalize(pearson_r, 0.7, 1.0),
            "kendall_tau": normalize(kendall_tau, 0.1, 1.0),
            "anova_f": normalize(f_stat, 0.0, 350.0),
            "kw_stat": normalize(kw_stat, 0.0, 1800.0),
            "std_dev": normalize(std_across_models, 0.0, 2.6)
        }
    }

def build_landmark_calibration_config(scores, desired_points=None):
    """
    Creates a piecewise-linear calibration from these raw distribution 
    landmarks: [min, Q1, median, Q3, max]
    to the given desired_points, e.g. [0, 3, 5, 7, 10].
    Returns a dict describing how to transform future scores.
    """
    if not scores or len(scores) < 2:
        # Degenerate case: no meaningful distribution
        return {
            "method": "piecewise_landmark",
            "in_landmarks": [],
            "out_landmarks": []
        }

    if desired_points is None:
        desired_points = [0, 3, 5, 7, 10]

    in_min = min(scores)
    in_q1 = float(np.percentile(scores, 25))
    in_med = float(statistics.median(scores))
    in_q3 = float(np.percentile(scores, 75))
    in_max = max(scores)

    return {
        "method": "piecewise_landmark",
        "in_landmarks": [in_min, in_q1, in_med, in_q3, in_max],
        "out_landmarks": desired_points
    }

def apply_landmark_calibration(x, config):
    """
    Apply the piecewise-linear transform defined by config:
      "in_landmarks" = [minVal, q1Val, medVal, q3Val, maxVal]
      "out_landmarks" = [outMin, outQ1, outMed, outQ3, outMax].
    If x is < min or > max, we extrapolate linearly beyond that segment.
    """
    inL = config.get("in_landmarks", [])
    outL = config.get("out_landmarks", [])
    if len(inL) != 5 or len(outL) != 5:
        # Invalid or degenerate config => just return x unchanged
        return x

    in_min, in_q1, in_med, in_q3, in_max = inL
    out_min, out_q1, out_med, out_q3, out_max = outL

    def linear_map(val, old_lo, old_hi, new_lo, new_hi):
        if abs(old_hi - old_lo) < 1e-12:
            return new_lo
        frac = (val - old_lo) / (old_hi - old_lo)
        return new_lo + frac * (new_hi - new_lo)

    # Determine which segment x belongs to:
    if x <= in_q1:
        # (in_min -> in_q1) -> (out_min -> out_q1), but possibly x < in_min => extrapolate
        return linear_map(x, in_min, in_q1, out_min, out_q1)
    elif x <= in_med:
        return linear_map(x, in_q1, in_med, out_q1, out_med)
    elif x <= in_q3:
        return linear_map(x, in_med, in_q3, out_med, out_q3)
    else:
        # (in_q3 -> in_max) -> (out_q3 -> out_max), possibly x > in_max => extrapolate
        return linear_map(x, in_q3, in_max, out_q3, out_max)

def log_score_summary(score_type: str, cross_stats: Dict, model_stats: Dict):
    """Log a readable summary of score statistics."""
    s = ""
    s += f"\n------- {score_type} Summary -------"
    s += f"ANOVA F-value: {cross_stats['anova_f']:.4f}, p={cross_stats['anova_p']:.4f}"
    s += f"Kruskal-Wallis: {cross_stats['kw_stat']:.4f}, p={cross_stats['kw_p']:.4f}"
    s += f"Pearson r={cross_stats['pearson_r']:.4f}"
    s += f"Kendall τ={cross_stats['kendall_tau']:.4f}"
    s += f"Std.Dev across models: {cross_stats['std_dev_across_models']:.4f}"

    s += "\nModel Scores:"
    sorted_models = sorted(
        model_stats.items(),
        key=lambda kv: kv[1]["mean"],
        reverse=True
    )
    for model, stats in sorted_models:
        line = f"{model:.<40} {stats['mean']:.3f} ±{stats['ci95']:.3f}"
        s += line
    s += "\n------------------------------------"
    logger.info(s)
    return s


def compute_weighted_score(logp):
    outs = {}
    choices = np.arange(11)  # Choices are 0-10
    for metric, logp_arr in logp.items():
        probs = np.exp(logp_arr)
        weights = probs / (probs.sum() + 1e-12)
        outs[metric] = (weights * choices).sum().item()

    return outs

def compute_ranked_score(logp):
    outs = {}
    choices = np.arange(11)  # Choices are 0-10
    for metric, logp_arr in logp.items():
        # res = kendalltau(choices, logp_arr, variant='b')

        # lets just use the common numbers 1,3,5,7,9, as some models like to skip some
        res = kendalltau(choices, logp_arr, variant='b')
        # print(res.correlation, res.pvalue, res2.correlation, res2.pvalue)
        # correlation weighted by pvalue


        decision = (res.correlation+1)*5 # scale to 0-10
        # decision = (2*decision*res.pvalue).clip(0, 10)
        outs[metric] = decision.item()

    return outs
