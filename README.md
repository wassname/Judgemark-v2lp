# Judgemark V2 - Log Probability Evaluation

A fork of Judgemark that evaluates whether log probability-based scoring methods outperform traditional text-based judgment approaches fo## Interpreting th## Contributing

## Overview

This project compares different methods of extracting scores from language models:

- **Raw**: Traditional text-based scoring (baseline Judgemark method)
- **Weighted**: Log probability weighting using normalized choice probabilities (similar to [G-Eval](https://arxiv.org/abs/2303.16634))
- **Ranked**: Ranking-based evaluation using Kendall's tau correlation between log probability rankings and score rankings
****
## Results

| name          | judgemark_score_calib | stability_calib | separability_calib | human_correlation_calib |
| ------------- | --------------------- | --------------- | ------------------ | ----------------------- |
| ranked_scaled | **0.788**             | **1.0**         | **0.785**          | 0.592                   |
| published     | 0.761                 | 0.894           | 0.691              | **0.908**               |
| ranked        | 0.74                  | 0.895           | 0.665              | 0.882                   |
| raw           | 0.731                 | 0.895           | 0.653              | 0.882                   |
| weighted      | 0.716                 | 0.886           | 0.633              | 0.876                   |
| ranked_norm   | 0.575                 | 0.644           | 0.506              | 0.781                   |
| weighted_norm | 0.545                 | 0.547           | 0.49               | 0.761                   |

*results for DeepSeek R1**


| Method         | Score | Score (Normalized) |
| -------------- | ----- | ------------------ |
| ranked_scaled  | 0.62  | 0.80               |
| ranked_norm    | 0.65  | 0.74               |
| weighted       | 0.63  | 0.65               |
| raw (baseline) | 0.63  | 0.65               |
| weighted_norm  | 0.62  | 0.64               |

*Results for DeepSeek Chat V3 0324*

The ranking approach performs best, particularly when scaled. This approach treats LLM log probabilities as rankings rather than true probabilities, which aligns better with how sampling methods like greedy and top-k actually work.

- ranked_scaled: this method is kendall tau (scaled to [0, 10] after normalising by the mean log probs over all samples 
  `kendallstau(logprobs-logprobs_all_mean(), range(10).collection`
- ranked: kendall tau (scaled to [0, 10]
- weighted: this method is similar to G-Eval, where the log probabilities are used to weight the choices based on their normalized probabilities.\
  - `weighted_choice = choice * logprob / sum(logprobs) * 10`

## Methodology: Ranking Approach

Instead of treating log probabilities as probabilities, the ranking method:

1. Extracts the full distribution of log probabilities for all possible choices (0-10):
   ```json
   {
     "0": -1.2,
     "1": -0.5,
     "2": -0.3,
     "3": -0.1,
     "4": -0.05,
     "5": -0.02,
     "6": -0.01,
     "7": -0.005,
     "8": -0.002,
     "9": -0.001
   }
   ```

2. Ranks choices by their log probabilities
3. Uses Kendall's tau to measure correlation between log probability rankings and expected score rankings
4. Achieves high efficiency by extracting complete score distributions from a single token


## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/wassname/Judgemark-v2lp.git
   cd Judgemark-v2
   ```

2. **Install Python dependencies** (requires Python 3.9+):
   ```bash
   uv sync
   source ./venv/bin/activate  # Activate the virtual environment
   ```

3. **Set up environment variables** for API credentials:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export OPENAI_API_URL="https://openrouter.ai/api/v1/chat/completions"
   ```

## Usage

Run the benchmark with the main script:

```bash
# test
uv run python judgemark_v2.py \
  --judge-model "meta-llama/llama-3.2-3b-instruct" \
  --samples-file data/judgemark_v2.1_samples.json \
  --prompts-file data/judge_prompts.json \
  --runs-file outputs/my_judgemark_runs.json \
  --threads 1 \
  --num-runs 1 \
  --save-raw-judge-output

uv run python judgemark_v2.py \
  --judge-model "qwen/qwen3-235b-a22b" \
  --samples-file data/judgemark_v2.1_samples.json \
  --prompts-file data/judge_prompts.json \
  --runs-file outputs/my_judgemark_runs2.json \
  --num-runs 1 \
  --save-raw-judge-output \
  --threads 8

uv run python judgemark_v2.py \
  --judge-model "deepseek/deepseek-r1" \
  --samples-file data/judgemark_v2.1_samples.json \
  --prompts-file data/judge_prompts.json \
  --runs-file outputs/my_judgemark_runs2.json \
  --num-runs 1 \
  --save-raw-judge-output
```

See original repo https://github.com/EQ-bench/Judgemark-v2 for citation, license, and more details.
