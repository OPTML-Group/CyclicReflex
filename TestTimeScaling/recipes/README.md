# Recipes

| Model | Method |
| :--- | :--- |
| DeepSeek-R1-Distill-Qwen-1.5B | [Best-of-N w/ orginal decoding](DeepSeek-R1-Distill-Qwen-1.5B/best_of_n.yaml) |
| | [Best-of-N w/ CyclicReflex](DeepSeek-R1-Distill-Qwen-1.5B/best_of_n_cyclical.yaml) |
| | [Beam search w/ orginal decoding](DeepSeek-R1-Distill-Qwen-1.5B/beam_search.yaml) |
| | [Beam search w/ CyclicReflex](DeepSeek-R1-Distill-Qwen-1.5B/beam_search_cyclical.yaml) |


## Testing
Each approach can be launched by specifying the associated YAML file, for example:
```shell
export CONFIG=recipes/DeepSeek-R1-Distill-Qwen-1.5B/best_of_n_cyclical.yaml

python scripts/test_time_compute.py $CONFIG --dataset_name=HuggingFaceH4/MATH-500 --dataset_split=train
```



## Extracting the MATH-500 accuracy numbers

To get the final numbers for the evalations, we use a [fork](https://github.com/huggingface/Qwen2.5-Math) of the [Qwen2.5-Math evaluation repo](https://github.com/QwenLM/Qwen2.5-Math). Please follow the installation and usage instructions in our fork to obtain accuracies on MATH-500.