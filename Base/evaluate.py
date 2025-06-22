import argparse
import os

import numpy as np
import torch
import wandb
from datasets import load_dataset, concatenate_datasets
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ------------------------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------------------------
from utils import model_data, check_answer_overall
from transformers import LogitsProcessor, LogitsProcessorList, AutoTokenizer


def cyclical_processor(
    tokenizer,
    wait_token_strs=["wait", "Wait", "but", "But", "Alternatively"],
    amplitude=3.0,
    period=100.0,
    shift=0.0,
    phi=None
):
    wait_token_ids = [tokenizer.convert_tokens_to_ids(s) for s in wait_token_strs]
    end_think_token_id = tokenizer.convert_tokens_to_ids("</think>")

    def processor(token_ids, logits):
        current_pos = len(token_ids)

        if end_think_token_id in token_ids:
            return logits

        if phi is not None and not any(start <= current_pos < end for start, end in phi):
            return logits

        shifted_pos = (current_pos + shift * period) % period
        cycle_pos = shifted_pos / period

        if cycle_pos <= 0.25:
            penalty = (cycle_pos / 0.25) * amplitude  # 0 → +A
        elif cycle_pos <= 0.75:
            penalty = amplitude - ((cycle_pos - 0.25) / 0.5) * 2 * amplitude  # +A → -A
        else:
            penalty = -amplitude + ((cycle_pos - 0.75) / 0.25) * amplitude  # -A → 0

        for wait_token_id in wait_token_ids:
            logits[wait_token_id] += penalty

        return logits

    return processor


def silver_processor(
    tokenizer,
    wait_token_strs=["wait", "Wait", "but", "But", "Alternatively"],
    rho=2.0,
    phi=None
):
    wait_token_ids = [tokenizer.convert_tokens_to_ids(s) for s in wait_token_strs]
    end_think_token_id = tokenizer.convert_tokens_to_ids("</think>")

    def processor(token_ids, logits):
        current_pos = len(token_ids)
        if end_think_token_id in token_ids:
            return logits
        if phi is not None and not any(start <= current_pos < end for start, end in phi):
            return logits

        # Silver schedule: α_t = 1 + rho^(ν(t+1) - 1)
        t = current_pos % 64
        nu = ((t + 1) & -(t + 1)).bit_length() - 2
        penalty = rho ** (nu - 1)
        # print(penalty)

        for wait_token_id in wait_token_ids:
            logits[wait_token_id] += penalty
        return logits

    return processor



def add_processor(tokenizer, wait_token_strs=["wait", "Wait", "but", "But", "Alternatively"], delta=-3, phi=None):
    wait_token_ids = [tokenizer.convert_tokens_to_ids(s) for s in wait_token_strs]
    end_think_token_id = tokenizer.convert_tokens_to_ids("</think>")

    def processor(token_ids, logits):
        current_pos = len(token_ids)
        if end_think_token_id in token_ids:
            return logits

        if phi is not None and not any(start <= current_pos < end for start, end in phi):
            return logits
        for wait_token_id in wait_token_ids:
            logits[wait_token_id] += delta
        return logits
    return processor


def load_model_and_tokenizer(model_choice, device="cuda"):
    num_gpus = torch.cuda.device_count()
    llm = LLM(model=model_data[model_choice]["model_name"], download_dir="./cache", tensor_parallel_size=num_gpus, dtype="float16")  # 也可以改成 "bf16"
    tokenizer = AutoTokenizer.from_pretrained(model_data[model_choice]["tokenizer_name"], cache_dir="./cache")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, llm


# ------------------------------------------------------------------------------------
# AIME2024 Evaluation
# ------------------------------------------------------------------------------------
def evaluate_aime2024(
        model,                # vLLM.LLM
        tokenizer,            
        device="cuda",
        split="train",
        max_samples=None,
        nothink=False,
        cot=False,
        manual_prompt=False,
        wandb_logging=False,
        file_name=None,
        sampling_params=None,
        qualified_idx_list=None
):
    # Load dataset
    ds = load_dataset("HuggingFaceH4/aime_2024")
    data = ds[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
        
    if qualified_idx_list is not None:
        data = data.select(qualified_idx_list)

    correct = 0
    total = 0
    total_generated_length = 0

    results_list = []

    # Reuse the generated outputs if available
    if file_name and os.path.exists(file_name):
        results_list = torch.load(file_name)
        if type(results_list) is not list:
            results_list = results_list["outputs"]

    with tqdm(data, desc=f"Evaluating AIME2024 (nothink={nothink}, cot={cot})") as pbar:
        for step, example in enumerate(pbar):
            question = example["problem"]
            gold_solution = example["solution"]
            gold_answer = example["answer"]

            prompt = "You are a helpful AI bot that answers questions for a user. Keep your response short and direct. <｜User｜>"
            if manual_prompt:
                prompt += f"Question: {question}\nLet's reason this step by step.\nAnswer:"
            else:
                if cot:
                    if nothink:
                        prompt += f"Question: {question}\n<think> </think> Let's reason this step by step.\nAnswer:"
                    else:
                        prompt += f"Question: {question}\nLet's reason this step by step.\nAnswer: <think>"
                else:
                    if nothink:
                        prompt += f"Question: {question}\n<think> </think> Answer:"
                    else:
                        prompt += f"Question: {question}\nAnswer: <think>"

            if step < len(results_list) and results_list[step]["question"] == question:
                output_text = results_list[step]["full_generation"]
                generated_length = results_list[step]["generation_length"]
            else:
                outputs = model.generate(prompt, sampling_params)
                output_text = outputs[0].outputs[0].text
                generated_length = len(output_text.split())

            is_correct, predicted_answer = check_answer_overall(output_text, gold_answer)
            correct += 1 if is_correct else 0
            total += 1

            total_generated_length += generated_length

            # Calculate current step metrics
            current_accuracy = correct / total if total > 0 else 0
            avg_generation_length = total_generated_length / total

            # Update TQDM postfix
            pbar.set_postfix({
                "accuracy": f"{current_accuracy:.3f}",
                "avg_len": f"{avg_generation_length:.1f}"
            })

            # Optionally log to wandb
            if wandb_logging:
                wandb.log({
                    "accuracy": current_accuracy,
                    "avg_generation_length": avg_generation_length
                }, step=step)

            if step == len(results_list):
                results_list.append({
                    "question": question,
                    "prompt": prompt,
                    "gold_solution": gold_solution,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "full_generation": output_text,
                    "generation_length": generated_length,
                    "correct": is_correct
                })
            else:
                results_list[step] = {
                    "question": question,
                    "prompt": prompt,
                    "gold_solution": gold_solution,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "full_generation": output_text,
                    "generation_length": generated_length,
                    "correct": is_correct
                }

            if file_name:
                torch.save(results_list, file_name)

    accuracy = correct / total if total > 0 else 0
    return accuracy, results_list


# ------------------------------------------------------------------------------------
# AMC2023 Evaluation
# ------------------------------------------------------------------------------------
def evaluate_amc2023(
        model, 
        tokenizer,            
        device="cuda",
        split="test",
        max_samples=None,
        nothink=False,
        cot=False,
        manual_prompt=False,
        wandb_logging=False,
        file_name=None,
        sampling_params=None,
        qualified_idx_list=None
):
    # Load dataset
    ds = load_dataset("zwhe99/amc23")
    data = ds[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
        
    if qualified_idx_list is not None:
        data = data.select(qualified_idx_list)

    correct = 0
    total = 0
    total_generated_length = 0

    results_list = []

    # Reuse the generated outputs if available
    if file_name and os.path.exists(file_name):
        results_list = torch.load(file_name)
        if type(results_list) is not list:
            results_list = results_list["outputs"]

    with tqdm(data, desc=f"Evaluating AMC2023 (nothink={nothink}, cot={cot})") as pbar:
        for step, example in enumerate(pbar):
            question = example["question"]
            gold_answer = str(example["answer"])

            prompt = "You are a helpful AI bot that answers questions for a user. Keep your response short and direct. <｜User｜>"
            if manual_prompt:
                prompt += f"Question: {question}\nLet's reason this step by step.\nAnswer:"
            else:
                if cot:
                    if nothink:
                        prompt += f"Question: {question}\n<think> </think> Let's reason this step by step.\nAnswer:"
                    else:
                        prompt += f"Question: {question}\nLet's reason this step by step.\nAnswer: <think>"
                else:
                    if nothink:
                        prompt += f"Question: {question}\n<think> </think> Answer:"
                    else:
                        prompt += f"Question: {question}\nAnswer: <think>"

            if step < len(results_list) and results_list[step]["question"] == question:
                output_text = results_list[step]["full_generation"]
                generated_length = results_list[step]["generation_length"]
            else:
                outputs = model.generate(prompt, sampling_params)
                output_text = outputs[0].outputs[0].text
                generated_length = len(output_text.split())

            is_correct, predicted_answer = check_answer_overall(output_text, gold_answer)
            correct += 1 if is_correct else 0
            total += 1

            total_generated_length += generated_length

            # Calculate current step metrics
            current_accuracy = correct / total if total > 0 else 0
            avg_generation_length = total_generated_length / total

            # Update TQDM postfix
            pbar.set_postfix({
                "accuracy": f"{current_accuracy:.3f}",
                "avg_len": f"{avg_generation_length:.1f}"
            })

            # Optionally log to wandb
            if wandb_logging:
                wandb.log({
                    "accuracy": current_accuracy,
                    "avg_generation_length": avg_generation_length
                }, step=step)

            # 缓存结果
            if step == len(results_list):
                results_list.append({
                    "question": question,
                    "prompt": prompt,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "full_generation": output_text,
                    "generation_length": generated_length,
                    "correct": is_correct
                })
            else:
                results_list[step] = {
                    "question": question,
                    "prompt": prompt,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "full_generation": output_text,
                    "generation_length": generated_length,
                    "correct": is_correct
                }

            if file_name:
                torch.save(results_list, file_name)

    accuracy = correct / total if total > 0 else 0
    return accuracy, results_list


# ------------------------------------------------------------------------------------
# AIME2025 Evaluation 
# ------------------------------------------------------------------------------------
def evaluate_aime2025(
        model,                # vLLM.LLM
        tokenizer, 
        device="cuda",
        split="train",
        max_samples=None,
        nothink=False,
        cot=False,
        manual_prompt=False,
        wandb_logging=False,
        file_name=None,
        sampling_params=None,
        qualified_idx_list=None
):

    aime2025_i = load_dataset("opencompass/AIME2025", 'AIME2025-I')[split]
    aime2025_ii = load_dataset("opencompass/AIME2025", 'AIME2025-II')[split]
    ds = concatenate_datasets([aime2025_i, aime2025_ii])

    data = ds
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    if qualified_idx_list is not None:
        data = data.select(qualified_idx_list)

    correct = 0
    total = 0
    total_generated_length = 0

    results_list = []

    # Reuse the generated outputs if available
    if file_name and os.path.exists(file_name):
        results_list = torch.load(file_name)
        if type(results_list) is not list:
            results_list = results_list["outputs"]

    with tqdm(data, desc=f"Evaluating AIME2025 (nothink={nothink}, cot={cot})") as pbar:
        for step, example in enumerate(pbar):
            question = example["question"]
            gold_answer = example["answer"]

            prompt = "You are a helpful AI bot that answers questions for a user. Keep your response short and direct. <｜User｜>"
            if manual_prompt:
                prompt += f"Question: {question}\nLet's reason this step by step.\nAnswer:"
            else:
                if cot:
                    if nothink:
                        prompt += f"Question: {question}\n<think> </think> Let's reason this step by step.\nAnswer:"
                    else:
                        prompt += f"Question: {question}\nLet's reason this step by step.\nAnswer: <think>"
                else:
                    if nothink:
                        prompt += f"Question: {question}\n<think> </think> Answer:"
                    else:
                        prompt += f"Question: {question}\nAnswer: <think>"

            if step < len(results_list) and results_list[step]["question"] == question:
                output_text = results_list[step]["full_generation"]
                generated_length = results_list[step]["generation_length"]
            else:
                outputs = model.generate(prompt, sampling_params)
                output_text = outputs[0].outputs[0].text
                generated_length = len(output_text.split())

            is_correct, predicted_answer = check_answer_overall(output_text, gold_answer)
            correct += 1 if is_correct else 0
            total += 1

            total_generated_length += generated_length

            # Calculate current step metrics
            current_accuracy = correct / total if total > 0 else 0
            avg_generation_length = total_generated_length / total

            # Update TQDM postfix
            pbar.set_postfix({
                "accuracy": f"{current_accuracy:.3f}",
                "avg_len": f"{avg_generation_length:.1f}"
            })

            # Optionally log to wandb
            if wandb_logging:
                wandb.log({
                    "accuracy": current_accuracy,
                    "avg_generation_length": avg_generation_length
                }, step=step)

            # 缓存结果
            if step == len(results_list):
                results_list.append({
                    "question": question,
                    "prompt": prompt,
                    # "gold_solution": gold_solution,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "full_generation": output_text,
                    "generation_length": generated_length,
                    "correct": is_correct
                })
            else:
                results_list[step] = {
                    "question": question,
                    "prompt": prompt,
                    # "gold_solution": gold_solution,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "full_generation": output_text,
                    "generation_length": generated_length,
                    "correct": is_correct
                }

            if file_name:
                torch.save(results_list, file_name)

    accuracy = correct / total if total > 0 else 0
    return accuracy, results_list


# ------------------------------------------------------------------------------------
# Math500 evaluator
# ------------------------------------------------------------------------------------
def evaluate_math500(
        model,                 # vLLM.LLM
        tokenizer,
        device="cuda",
        split="test",
        max_samples=None,
        nothink=False,
        cot=False,
        manual_prompt=False,
        wandb_logging=False,
        file_name=None,
        sampling_params=None,
        qualified_idx_list=None
):
    data = load_dataset("HuggingFaceH4/MATH-500")[split]

    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    if qualified_idx_list is not None:
        data = data.select(qualified_idx_list)

    correct = 0
    total = 0
    total_generated_length = 0
    results_list = []
    correctness_per_level = {1: [], 2: [], 3: [], 4: [], 5: []}

    if file_name and os.path.exists(file_name):
        results_list = torch.load(file_name)
        if type(results_list) is not list:
            results_list = results_list["outputs"]

    with tqdm(data, desc=f"Evaluating Math500 (nothink={nothink}, cot={cot})") as pbar:
        for step, example in enumerate(pbar):
            question = example["problem"]
            gold_solution = example["solution"]
            gold_answer = example["answer"]
            difficulty_level = example["level"]

            prompt = "You are a helpful AI bot that answers questions for a user. Keep your response short and direct. <｜User｜>"
            if manual_prompt:
                prompt += f"Question: {question}\nLet's reason this step by step.\nAnswer:"
            else:
                if cot:
                    if nothink:
                        prompt += f"Question: {question}\n<think> </think> Let's reason this step by step.\nAnswer:"
                    else:
                        prompt += f"Question: {question}\nLet's reason this step by step.\nAnswer: <think>"
                else:
                    if nothink:
                        prompt += f"Question: {question}\n<think> </think> Answer:"
                    else:
                        prompt += f"Question: {question}\nAnswer: <think>"

            if step < len(results_list) and results_list[step]["question"] == question and results_list[step]["prompt"] == prompt:
                output_text = results_list[step]["full_generation"]
                generated_length = results_list[step]["generation_length"]
            else:
                outputs = model.generate(prompt, sampling_params)
                output_text = outputs[0].outputs[0].text
                generated_length = len(output_text.split())

            total_generated_length += generated_length
            is_correct, predicted_answer = check_answer_overall(output_text, gold_answer)

            if is_correct:
                correct += 1
                correctness_per_level[difficulty_level].append(1)
            else:
                correctness_per_level[difficulty_level].append(0)
            total += 1

            current_accuracy = correct / total
            avg_generation_length = total_generated_length / total

            pbar.set_postfix({
                "accuracy": f"{current_accuracy:.3f}",
                "avg_len": f"{avg_generation_length:.1f}"
            })

            if wandb_logging:
                wandb.log({
                    "accuracy": current_accuracy,
                    "avg_generation_length": avg_generation_length,
                    "math500_acc_level_1": np.mean(correctness_per_level[1]) if len(correctness_per_level[1]) > 0 else 0,
                    "math500_acc_level_2": np.mean(correctness_per_level[2]) if len(correctness_per_level[2]) > 0 else 0,
                    "math500_acc_level_3": np.mean(correctness_per_level[3]) if len(correctness_per_level[3]) > 0 else 0,
                    "math500_acc_level_4": np.mean(correctness_per_level[4]) if len(correctness_per_level[4]) > 0 else 0,
                    "math500_acc_level_5": np.mean(correctness_per_level[5]) if len(correctness_per_level[5]) > 0 else 0,
                }, step=step)

            # 存结果
            result_entry = {
                "question": question,
                "prompt": prompt,
                "gold_solution": gold_solution,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "full_generation": output_text,
                "generation_length": generated_length,
                "correct": is_correct,
                "difficulty_level": difficulty_level,
            }
            if step == len(results_list):
                results_list.append(result_entry)
            else:
                results_list[step] = result_entry

            if file_name:
                torch.save(results_list, file_name)

    accuracy = correct / total
    return accuracy, results_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_choice", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="math500")
    parser.add_argument("--mode", choices=["NoCoT_withThinking", "NoCoT_withoutThinking", "CoT_withoutThinking", "CoT_withThinking", "Plain"], required=True)
    parser.add_argument("--max_new_tokens", type=int, default=4000)

    parser.add_argument("--logits_processor_type", type=str, default=None,
                        choices=["add", "cyclical", "silver", None])
    parser.add_argument("--wait_decay_alpha", type=float, default=0.0)
    parser.add_argument("--wait_add_delta", type=float, default=0.0)

    parser.add_argument("--wait_cyclical_amplitude", type=float, default=1.0)
    parser.add_argument("--wait_cyclical_period", type=float, default=600)
    parser.add_argument("--wait_cyclical_shift", type=float, default=0)

    parser.add_argument("--wait_silver_rho", type=float, default=1.5)

    parser.add_argument("--wait_range_phi", type=int, nargs='+', default=[0, 8192])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    phi_list = [(args.wait_range_phi[i], args.wait_range_phi[i+1]) for i in range(0, len(args.wait_range_phi), 2)]
    phi_str = "_".join([f"{s}-{e}" for s, e in phi_list])

    name_parts = [str(args.datasets), args.model_choice, args.mode, str(args.max_new_tokens), str(args.seed)]
    if args.logits_processor_type:
        name_parts.append(f"{args.logits_processor_type}")
        if args.logits_processor_type == "add":
            name_parts.append(f"delta{args.wait_add_delta}phi{phi_str}")
        if args.logits_processor_type == "cyclical":
            name_parts.append(f"amplitude{args.wait_cyclical_amplitude}period{args.wait_cyclical_period}shift{args.wait_cyclical_shift}phi{phi_str}")
        if args.logits_processor_type == "silver":
            name_parts.append(f"rho{args.wait_silver_rho}phi{phi_str}")

    run_name = "_".join(map(str, name_parts))
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=run_name + "-eval",
            config=vars(args)
        )

    os.makedirs(f"results/evaluation/{args.wandb_project}/", exist_ok=True)
    filename = f"results/evaluation/{args.wandb_project}/{run_name}_vllmoutputs.pt"

    tokenizer, model = load_model_and_tokenizer(args.model_choice, device=args.device)

    logits_processor = None
    if args.logits_processor_type is not None:
        processors = []
        if args.logits_processor_type == "add":
            processors.append(add_processor(tokenizer=tokenizer, delta=args.wait_add_delta, phi=phi_list))
        if args.logits_processor_type == "cyclical":
            processors.append(cyclical_processor(tokenizer=tokenizer, amplitude=args.wait_cyclical_amplitude, period=args.wait_cyclical_period, shift=args.wait_cyclical_shift, phi=phi_list))
        if args.logits_processor_type == "silver":
            processors.append(silver_processor(tokenizer=tokenizer, rho=args.wait_silver_rho, phi=phi_list))
        logits_processor = LogitsProcessorList(processors)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        logits_processors=logits_processor,
        seed=args.seed
    )

    combos = {
        "NoCoT_withThinking": [False, False],
        "NoCoT_withoutThinking": [True, False],
        "CoT_withThinking": [False, True],
        "CoT_withoutThinking": [True, True],
        "Plain": [False, False]
    }
    manual_prompt = (args.mode == "Plain")
    qualified_idx_list = None


    if "math500" in args.datasets:
        accuracy, math500_results_list = evaluate_math500(
            model,
            tokenizer,
            device=args.device,
            wandb_logging=bool(args.wandb_project),
            max_samples=args.max_samples,
            nothink=combos[args.mode][0],
            cot=combos[args.mode][1],
            manual_prompt=manual_prompt,
            file_name=filename,
            sampling_params=sampling_params,
            qualified_idx_list=qualified_idx_list
        )

        math500_results = {
            "model": args.model_choice,
            "mode": args.mode,
            "accuracy": accuracy,
            "outputs": math500_results_list
        }
        print(f"[{args.mode}] Math500 accuracy over {args.max_samples} samples: {accuracy:.3f}")
        if args.wandb_project:
            wandb.run.summary[f"{args.model_choice}_{args.mode}_math500_final_accuracy"] = accuracy
        torch.save(math500_results, filename)


    if 'aime2024' in args.datasets:
        accuracy, results_list = evaluate_aime2024(
            model,
            tokenizer,
            device=args.device,
            split="train",
            max_samples=args.max_samples,
            nothink=combos[args.mode][0],
            cot=combos[args.mode][1],
            manual_prompt=manual_prompt,
            wandb_logging=bool(args.wandb_project),
            file_name=filename,
            sampling_params=sampling_params,
            qualified_idx_list=qualified_idx_list
        )
        aime2024_results = {
            "model": args.model_choice,
            "mode": args.mode,
            "accuracy": accuracy,
            "outputs": results_list
        }
        print(f"[{args.mode}] AIME2024 accuracy over {args.max_samples} samples: {accuracy:.3f}")

        if args.wandb_project:
            wandb.run.summary[f"{args.model_choice}_{args.mode}_aime2024_final_accuracy"] = accuracy
        torch.save(aime2024_results, filename)


    if 'aime2025' in args.datasets:
        accuracy, results_list = evaluate_aime2025(
            model,
            tokenizer,
            device=args.device,
            split="test",
            max_samples=args.max_samples,
            nothink=combos[args.mode][0],
            cot=combos[args.mode][1],
            manual_prompt=manual_prompt,
            wandb_logging=bool(args.wandb_project),
            file_name=filename,
            sampling_params=sampling_params,
            qualified_idx_list=qualified_idx_list
        )
        aime2025_results = {
            "model": args.model_choice,
            "mode": args.mode,
            "accuracy": accuracy,
            "outputs": results_list
        }
        print(f"[{args.mode}] AIME2025 accuracy over {args.max_samples} samples: {accuracy:.3f}")

        if args.wandb_project:
            wandb.run.summary[f"{args.model_choice}_{args.mode}_aime2025_final_accuracy"] = accuracy
        torch.save(aime2025_results, filename)

    if 'amc2023' in args.datasets:
        accuracy, results_list = evaluate_amc2023(
            model,
            tokenizer,
            device=args.device,
            split="test",
            max_samples=args.max_samples,
            nothink=combos[args.mode][0],
            cot=combos[args.mode][1],
            manual_prompt=manual_prompt,
            wandb_logging=bool(args.wandb_project),
            file_name=filename,
            sampling_params=sampling_params,
            qualified_idx_list=qualified_idx_list
        )
        amc2023_results = {
            "model": args.model_choice,
            "mode": args.mode,
            "accuracy": accuracy,
            "outputs": results_list
        }
        print(f"[{args.mode}] AMC2023 accuracy over {args.max_samples} samples: {accuracy:.3f}")

        if args.wandb_project:
            wandb.run.summary[f"{args.model_choice}_{args.mode}_aime2024_final_accuracy"] = accuracy
        torch.save(amc2023_results, filename)


if __name__ == "__main__":
    main()
