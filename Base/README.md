# Overall performance of CyclicReflex

1. Set up the Python environment

   First, create and activate a Python virtual environment using Conda (Python 3.11 is recommended), then install the dependencies:

   ```bash
   conda create -n cr python=3.11
   conda activate cr
   pip install -r requirements.txt
   ```

2. Run the evaluation script

   Execute the following command to start evaluation:

   ```bash
   ./run_cyclicreflex.sh
   ```

   - `--model_choice` supports:
     - `qwen1.5b`: DeepSeek-R1-Distill-Qwen-1.5B  
     - `qwen7b`: DeepSeek-R1-Distill-Qwen-7B  
     - `llama8b`: DeepSeek-R1-Distill-Llama-8B  

   - `--datasets` supports:
     - `math500`
     - `aime2024`
     - `aime2025`
     - `amc2023`

3. Evaluation results will be stored under: `results/evaluation/cyclicreflex`. You can also monitor detailed logging via wandb.