Fork of judgemark to see if using weighted logprob, or ranklogprob work better than the current method

Changes
- openrouters only
- get logprobs
- added options
  - `--score-weighted`
  - `--score-ranklog`

models
- meta-llama/llama-3.2-3b-instruct	
- qwen/qwen-2.5-72b-instruct
- deepseek/deepseek-chat-v3-0324	
- nousresearch/hermes-3-llama-3.1-405b

```bash
python judgemark_v2.py \
  --judge-model "meta-llama/llama-3.2-3b-instruct" \
  --samples-file data/judgemark_v2.1_samples.json \
  --prompts-file data/judge_prompts.json \
  --runs-file my_judgemark_runs.json \
  --threads 1 \
  --num-runs 1 \
  --save-raw-judge-output
```

## Results

TODO

----

# Judgemark V2

**Judgemark V2** is a benchmark that evaluates how well a language model can judge creative writing. Instead of relying on simple pairwise preferences, Judgemark V2 prompts the judge model to assign numeric scores for multiple literary criteria (e.g., “Nuanced Characters,” “Overwrought,” “Emotionally Engaging”). It then aggregates those scores, measures how consistent and discriminative they are, and derives a final numeric rating of the judge model’s performance.

The Judgemark leaderboard can be found here: [https://eqbench.com/judgemark-v2.html](https://eqbench.com/judgemark-v2.html)

## Key Features

- **Complex Numeric Scoring**: Requires the judge model to provide 0–10 scores for dozens of criteria, highlighting any shortcomings in following complex instructions.
- **Raw & Calibrated Scores**: The system calculates a “raw” Judgemark score from the judge’s out-of-the-box distribution, and a “calibrated” score after normalizing the distribution for fairer cross-model comparisons.
- **Stability & Separability Metrics**: Goes beyond correlation to measure *how stable* the judge’s rankings are across repeated runs, and *how well* it separates strong from weak creative outputs.
- **Threaded Execution**: Supports multi-threaded item processing, drastically reducing the time required to score multiple creative samples.


## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/wassname/Judgemark-v2lp.git
   cd Judgemark-v2
   ```

2. **Install Python dependencies** (make sure you’re on Python 3.9+):

   ```bash
   uv sync
   . ./venv/bin/activate  # Activate the virtual environment
   ```

3. **Set up environment variables** to include your judge model’s API credentials. For example, if you’re using OpenAI-compatible endpoints:

   ```bash
   # (in .env or system env)
   export OPENAI_API_KEY="sk-..."
   export OPENAI_API_URL="https://openrouter.ai/api/v1/chat/completions"
   ```

## Usage

Run the benchmark via the main script `judgemark_v2.py`. For instance:

```bash
python judgemark_v2.py \
  --judge-model "openai/gpt-4o-mini" \
  --samples-file data/judgemark_v2.1_samples.json \
  --prompts-file data/judge_prompts.json \
  --runs-file my_judgemark_runs.json \
  --threads 20 \
  --num-runs 1 \
  --save-raw-judge-output
```

### Command-Line Options

- **`--judge-model`** (required): The model identifier (e.g. `openai/gpt-4`, `anthropic/claude-v1`).
- **`--samples-file`**: Path to the JSON with creative-writing samples to be judged. Default: `data/judgemark_v2.1_samples.json`.
- **`--prompts-file`**: Path to the JSON with partial prompts for the judge. Default: `data/judge_prompts.json`.
- **`--runs-file`**: The output JSON to store final run results. Default: `judgemark_v2_runs.json`.
- **`--run-id`**: A custom run ID for continuing or naming a run (optional).
- **`--threads`**: Number of threads for parallel scoring. Default: `6`.
- **`--verbosity`**: Log verbosity: one of `[DEBUG, INFO, WARNING, ERROR, CRITICAL]`.
- **`--num-runs`**: Number of times to repeat the entire benchmark. Default: `1`.
- **`--save-raw-judge-output`**: Store the raw text responses from the judge into the results JSON.

## How It Works

1. **Reading In Samples**  
   The script loads `samples_file`, which contains completions to creative writing prompts from multiple “writer models.”

2. **Generating Judge Prompts**  
   For each completion, we load a judge prompt from `prompts_file`. This typically includes instructions like:
   ```
   Please assign numeric scores (0-10) for these criteria:
   - Nuanced Characters
   - Overwrought
   - ...
   [TEST MODEL RESPONSE]
   ...
   ```

3. **Sending Requests to the Judge Model**  
   Each completion + prompt is sent to the `--judge-model` via the functions in `utils/api.py`. We specify a moderate temperature (often `0.5`) and top-k for variability.

4. **Parsing the Judge Output**  
   The script captures lines like `Nuanced Characters: 8` or `Weak Dialogue: 3`, extracts the numeric scores, and aggregates them into a single raw score. Negative criteria (like “Weak Dialogue”) are inverted so 10 = worst.

5. **Storing & Re-Trying**  
   Results are saved in your designated `runs-file`. If an item fails or provides incomplete scores, the script can retry (in subsequent runs) without overwriting previous data.

6. **Final Judgemark Scores**  
   Once all samples are scored:
   - A *raw* Judgemark score is computed from the distribution of assigned scores.  
   - A *calibrated* score is computed after normalizing each judge’s “score spread” to a standard distribution anchored to the mean, 25th & 75th percentile, upper & lower range. Calibration linearly transforms the distribution from these anchor points to match an ideal distribution of 0-10 range, 5 mean, and 25th & 75th percentile 
   - Additional metrics quantify how consistent (stable) and discriminative the judge is.

## Interpreting the Results

The output JSON in your `--runs-file` will contain many details, including per-model breakdowns, iteration-level stats, and final composite scores:

- **`final_judgemark_score`**: The primary benchmark result (based on calibrated distribution). A higher value suggests better correlation with reference preferences, stronger separation between good and weak writing, and higher consistency.
- **`final_judgemark_score_raw`**: A non-calibrated version that shows how well the judge performs “out of the box.”
- **Per-model details**: Found under `results[MODEL_NAME]`, including each snippet’s aggregated raw score and partial criterion scores.

You can also enable **visualization**: the code in `utils/visualization.py` produces bar charts, heatmaps, and scatter plots illustrating how the judge assigned scores across models.

## Contributing

Contributions and bug reports are welcome! If you’d like to add new features—such as custom scoring criteria, improved calibration, or alternative reference sets—feel free to open a PR or file an issue.

## License

This project is licensed under an [MIT License](LICENSE). See the `LICENSE` file for more details.

## Acknowledgments

- **LMSys Chatbot Arena** -- the source for the rankings used in the benchmark for human preference correlation.

---

**Happy Judging!** If you have any questions, reach out via [GitHub Issues](https://github.com/EQ-bench/judgemark-v2/issues) or contact the maintainers.
