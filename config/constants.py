"""Global constants and reference scores for the Judgemark-v2 benchmark."""

# Stability test configuration
STABILITY_ITEMS = [
    ("gemma-2b-it", "2", "28"),
    ("Llama-3-70b-chat-hf", "1", "42"),
    ("DeepSeek-R1", "1", "2"),
]
STABILITY_REPS = 100

# Reference scores for correlation
REFERENCE_MODEL_SCORES = {
    "DeepSeek-R1": 1430, # !! need to update this
    "gpt-4o-2024-11-20": 1402,
    "gemini-1.5-pro-002": 1333,
    "gemini-1.5-pro-001": 1276,
    "claude-3-5-sonnet-20240620": 1243,
    "Mistral-Large-Instruct-2411": 1246,
    "claude-3-opus-20240229": 1240,
    #"c4ai-command-r-plus-08-2024": 1236,
    "Llama-3-70b-chat-hf": 1214,
    "claude-3-haiku-20240307": 1163,
    "c4ai-command-r-08-2024": 1159,
    "Mixtral-8x22B-Instruct-v0.1": 1147,
    "Mixtral-8x7B-Instruct-v0.1": 1114,
    "databricks/dbrx-instruct": 1102,
    #"openchat-3.5-1210": 1127,
    "gpt-3.5-turbo-0125": 1099,
    "Llama-2-13b-chat-hf": 1050,
    "gemma-7b-it": 1029,
    "gemma-2b-it": 989,
}

# Negative criteria markers for score computation
NEGATIVE_MARKERS = [
    "melodramatic", "shallow resolution", "unearned resolution",
    "simplistic moralizing", "shallow optimism", "forced optimism", 
    "trite", "overwrought", "amateurish", "contrived", "uninspiring",
    "characters are too good", "incongruent ending positivity",
    "unearned transformations", "profundity over-reach",
    "amateurish descriptives", "clunky asides", "stilted dialogue",
    "tit-for-tat dialogue", "purple prose", "uncreative", "tell-don't-show",
    "weak dialogue", "meandering"
]

MODEL_NAME_REPLACEMENTS = {
    "mistralai/ministral-3b": "ministral/Ministral-3b-instruct",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "deepseek/deepseek-r1": "deepseek-ai/deepseek-r1",
    "anthropic/claude-3.5-haiku-20241022": "claude-3.5-haiku-20241022",
    "anthropic/claude-3.5-sonnet-20240620": "claude-3.5-sonnet-20240620",
    "openai/gpt-4o-2024-11-20": "gpt-4o-2024-11-20",
    "deepseek/deepseek-r1-distill-llama-70b": "deepseek-ai/deepseek-r1-distill-llama-70b",
    "mistralai/mistral-large-2411": "mistralai/mistral-large-instruct-2411",
}