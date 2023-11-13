# import: in-house
from instruct_mining.quality_metrics import evaluator
from instruct_mining.data_pipeline import data_loader
from instruct_mining.data_pipeline import data_filtering
from instruct_mining.data_pipeline import data_standardize


DATA_CACHE_PATH = "./cache"
DATA_SCORE_PATH = "./datasets/scored"
DATA_OUTPUT_PATH = "./datasets/processed"
TOP_N = 200

CONFIG = {
    "dataset_name": "platypus",
    "dataset_path": "garage-bAInd/Open-Platypus",
    "prompt_key": "instruction",
    "completion_key": "output",
    "context_key": "input",
    "input_key": "_input",
}


if __name__ == "__main__":
    # load datasets.
    dataset = data_loader(**CONFIG)
    dataset = evaluator(
        dataset=dataset,
        prompt_key=CONFIG["prompt_key"],
        completion_key=CONFIG["completion_key"],
        input_key=CONFIG["input_key"],
        cache=True,
        cache_path=DATA_CACHE_PATH,
    )

    # save the result.
    dataset.cleanup_cache_files()
    dataset.save_to_disk(f"{DATA_SCORE_PATH}/{CONFIG['dataset_name']}")

    # filter top n.
    dataset = data_filtering(
        dataset=dataset,
        top_n=TOP_N,
    )

    # standardize the dataset.
    dataset = data_standardize(dataset)

    # save the final dataset as `jsonl`.
    dataset.to_json(f"{DATA_OUTPUT_PATH}/{CONFIG['dataset_name']}.jsonl")
