# import: huggingface
from datasets import load_dataset
from datasets import Dataset

# import: in-house
from instruct_mining.utils import authenticate_hf


def data_loader(
    dataset_path: str,
    prompt_key: str,
    completion_key: str,
    context_key: str = "_context",
    input_key: str = "_input",
    prompt_template: str = "###Instruction:\n{}###Response:\n{}",
    **kwargs,
) -> Dataset:

    """
    Load huggingface's dataset and create a column combining prompt, context, and completion.
    
    Args:
        dataset_name (str): short name for a specific dataset.
        dataset_path (str): path to load the dataset.
        prompt_key (str): column name representing prompt in the dataset.
        completion_key (str): column name representing completion in the dataset.
        context_key (str): column name representing additional context in the dataset.
        input_key (str): column name representing combined prompt, context, and completion.
        prompt_template (str): template for input format to feed in a model.

    Returns:
        Huggingface dataset.
    """
    
    # load dataset.
    authenticate_hf()
    dataset = load_dataset(dataset_path, split="train", use_auth_token=True)
        
    # fill the context column with a placeholder.
    if context_key == "_context":
        dataset = dataset.map(
            lambda x: {
                context_key: "",
            }
        )

    # prepare input of a model.
    dataset = dataset.map(
        lambda x: {
            input_key: prompt_template.format(
                x[prompt_key],
                x[completion_key],
            )
        } 
    )
    dataset.cleanup_cache_files()

    return dataset
