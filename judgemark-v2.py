import sys
import signal
import argparse
from loguru import logger
import time


from judgemark_v2lp.utils.logging_setup import setup_logging, get_verbosity
from judgemark_v2lp.utils.file_io import load_json_file
from judgemark_v2lp.benchmark import run_judgemark_v2
from judgemark_v2lp.utils.api import API_KEY
from judgemark_v2lp.utils.state import should_exit, executor


def signal_handler(signum, frame):
    """Handle interrupt signals (SIGINT, SIGTERM)."""
    global executor, should_exit
    print(f"\n[DEBUG] Signal {signum} caught!")
    logger.warning("Signal handler called")
    should_exit = True
    time.sleep(0.1)  # Give workers a moment to see the flag
    if executor:
        logger.info("Shutting down executor from signal handler")
        executor.shutdown(wait=False)
        logger.info("Executor shutdown complete")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Judgemark-v2 Benchmark')
    parser.add_argument(
        '--judge-model',
        required=True,
        help='Judge model identifier (e.g., openai/gpt-4)'
    )
    parser.add_argument(
        '--samples-file',
        default="data/judgemark_v2.1_samples.json",
        help='JSON file containing pre-generated samples from various writer models'
    )
    parser.add_argument(
        '--prompts-file',
        default="data/judge_prompts.json",
        help='JSON file containing the partial judge prompts to be filled with test responses'
    )
    parser.add_argument(
        '--runs-file',
        default="judgemark_v2_runs.json",
        help='Path to store the Judgemark run results'
    )
    parser.add_argument(
        '--run-id',
        help='Resume (or create) a run using this base ID, to be combined with the judge model name'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=0,
        help='Number of threads to use'
    )
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging verbosity level'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1,
        help='Number of benchmark runs to execute'
    )
    parser.add_argument(
        '--save-raw-judge-output',
        action='store_true',
        default=True,
        help='If set, store the raw judge model output in the results JSON (default: true)'
    )
    parser.add_argument(
        '--score-weighted',
        action='store_true',
        default=False,
        help='If set, use weighted scoring for the judge model (default: false)'
    )
    parser.add_argument(
        '--score-ranked',
        action='store_true',
        default=False,
        help='If set, use ranked logprob scoring for the judge model (default: false)'
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Reset sentinel
    should_exit = False
    
    # Parse args
    args = parse_args()
    
    # Setup logging
    verbosity = get_verbosity(args.verbosity)
    setup_logging(verbosity)
    logger.debug("Logging initialized")
    
    # Check that we have an API key for the judge model
    if not API_KEY:
        logger.critical("No OPENAI_API_KEY found in environment variables.")
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    run_ids = []
    for i in range(1, args.num_runs + 1):
        if should_exit:
            break
        logger.info(f"Starting Judgemark-v2 run {i} of {args.num_runs}")
        rid = run_judgemark_v2(
            judge_model=args.judge_model,
            samples_file=args.samples_file,
            prompts_file=args.prompts_file,
            runs_file=args.runs_file,
            num_threads=args.threads,
            run_id=args.run_id,
            save_raw_judge_output=args.save_raw_judge_output
        )
        run_ids.append(rid)
    
    # Finally, print summary
    runs = load_json_file(args.runs_file)
    logger.info("\nAll Judgemark-v2 runs completed:")
    print("\nAll Judgemark-v2 runs completed:")
    for rid in run_ids:
        rd = runs.get(rid, {})
        final_score = rd.get("final_judgemark_score", "N/A")
        logger.info(f"Run ID: {rid}, Final Judgemark Score: {final_score}")
        print(f"Run ID: {rid}")
        print(f"Final Judgemark-v2 Score: {final_score}")
