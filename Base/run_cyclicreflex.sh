#!/bin/bash

# DeepSeek-R1-Distill-Qwen-7B on MATH500 w/ CyclicReflex
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --model_choice qwen7b --datasets math500 --max_new_tokens 8192 --logits_processor_type cyclical --wait_cyclical_amplitude 5.0 --wait_cyclical_period 1200 --mode CoT_withThinking --wandb_project cylicreflex

# DeepSeek-R1-Distill-Qwen-7B on AIME2024 w/ CyclicReflex
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --model_choice qwen7b --datasets aime2024 --max_new_tokens 8192 --logits_processor_type cyclical --wait_cyclical_amplitude 5.0 --wait_cyclical_period 1200 --mode CoT_withThinking --wandb_project cylicreflex

# DeepSeek-R1-Distill-Qwen-7B on AIME2025 w/ CyclicReflex
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --model_choice qwen7b --datasets aime2025 --max_new_tokens 8192 --logits_processor_type cyclical --wait_cyclical_amplitude 5.0 --wait_cyclical_period 1200 --mode CoT_withThinking --wandb_project cylicreflex

# DeepSeek-R1-Distill-Qwen-7B on AMC2023 w/ CyclicReflex
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --model_choice qwen7b --datasets amc2023 --max_new_tokens 8192 --logits_processor_type cyclical --wait_cyclical_amplitude 5.0 --wait_cyclical_period 900 --mode CoT_withThinking --wandb_project cylicreflex