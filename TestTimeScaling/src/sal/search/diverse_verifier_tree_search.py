#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv, generate_k_steps

logger = logging.getLogger()


from transformers import LogitsProcessorList
def cyclical_processor(
    tokenizer,
    wait_token_strs=["wait", "Wait", "but", "But", "Alternatively"],
    amplitude=3.0,   # 最大振幅
    period=100.0,    # 完整周期步数
    shift=0.0,       # 水平偏移（占周期的比例）
    phi=None         # 限制 penalty 应用的 token 区间
):
    wait_token_ids = [tokenizer.convert_tokens_to_ids(s) for s in wait_token_strs]
    end_think_token_id = tokenizer.convert_tokens_to_ids("</think>")

    def processor(token_ids, logits):
        current_pos = len(token_ids)

        # ✅ 如果已经生成了 </think>，不再加 penalty
        if end_think_token_id in token_ids:
            return logits

        # ✅ 如果设置了 phi，只在指定区间加 penalty
        if phi is not None and not any(start <= current_pos < end for start, end in phi):
            return logits

        # ✅ 计算周期性 penalty：0 → +A → -A → 0
        shifted_pos = (current_pos + shift * period) % period
        cycle_pos = shifted_pos / period  # 范围 [0, 1)

        if cycle_pos <= 0.25:
            penalty = (cycle_pos / 0.25) * amplitude  # 0 → +A
        elif cycle_pos <= 0.75:
            penalty = amplitude - ((cycle_pos - 0.25) / 0.5) * 2 * amplitude  # +A → -A
        else:
            penalty = -amplitude + ((cycle_pos - 0.75) / 0.25) * amplitude  # -A → 0

        # ✅ 应用于所有 wait token
        for wait_token_id in wait_token_ids:
            logits[wait_token_id] += penalty

        return logits

    return processor


def add_processor(tokenizer, wait_token_strs=["wait", "Wait", "but", "But", "Alternatively"], delta=-3, phi=[(0, 600)]):
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


def _dvts(batch_of_prompts: list[str], config: Config, llm: LLM, prm: PRM):

    tokenizer = llm.get_tokenizer()

    # 构造 logits processor
    processors = []
    if config.processor == "cyclical":
        processors.append(cyclical_processor(tokenizer=tokenizer, **config.processor_kwargs))
    if config.processor == "add":
        processors.append(add_processor(tokenizer=tokenizer, **config.processor_kwargs))
    logits_processor = LogitsProcessorList(processors)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=2048,
        top_p=config.top_p,
        stop=[
            "\n\n"
        ],  # we consider that a step in the problem is indicated by a double newline
        include_stop_str_in_output=True,
        n=1,
        # Resource Allocation
        logits_processors=logits_processor,
    )

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n_beams):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    best_scores=[0.0],
                    all_scores=[],
                    previous_text=None,
                    pruned=False,
                    stop_reasons=None,
                    history=[],
                )
            )

    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        # generation
        gen_beams = [b for b in beams if not b.pruned]
        if len(gen_beams) == 0:
            break

        if i == config.num_iterations - 1:
            # last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=2048,
                top_p=config.top_p,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in gen_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        # TODO: set the augmented template from a file
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps(
            templated_convs, lookahead, llm, sampling_params, config.beam_width
        )

        prompts, completions = [], []
        for beam, gen_result in zip(gen_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            if len(beam.next_texts) != config.beam_width:
                beam.pruned = True
                # rarely ~1/1000 the model will generate few beams than expected. #TODO: investigate why
                logger.warning(
                    f"beam {beam.index} has {len(beam.next_texts)} completions"
                )
            prompts.append(beam.prompt)
            completions.append([beam.current_text + t for t in beam.lookahead_texts])

        # scoring and chose best generation per beam TODO: add option for selection across beams within the same prompt

        all_scores = prm.score(prompts, completions)

        for beam, scores in zip(gen_beams, all_scores, strict=True):
            agg_scores = [aggregate_scores(s, config.agg_strategy) for s in scores]
            best_score_ind = np.argmax(agg_scores)
            beam.all_scores = scores
            beam.previous_text = beam.current_text
            beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
            beam.history.append(beam.next_texts[best_score_ind])
            beam.best_scores = scores[best_score_ind]
            if (
                beam.next_texts[best_score_ind] == ""
                or beam.stop_reasons[best_score_ind] == "EOS"
            ):
                # stopped on EOS, prune
                beam.pruned = True

        # filter / prune
        for beam in gen_beams:
            if "boxed{" in beam.current_text:
                beam.pruned = True

    # we need to copy the results from the last iteration in to beam_width beams as otherwise we would only have n/m results
    output: list[Beam] = []
    for beam in beams:
        for i in range(config.beam_width):
            output.append(
                Beam(
                    prompt=beam.prompt,
                    index=beam.index,
                    current_text=beam.previous_text + beam.next_texts[i],
                    next_texts=None,
                    lookahead_texts=None,
                    stop_reasons=None,
                    best_scores=beam.all_scores[i],
                    all_scores=beam.all_scores,
                    previous_text=beam.current_text,
                    pruned=beam.pruned,
                    history=beam.history,
                )
            )

    return output


def dvts(examples, config: Config, llm: LLM, prm: PRM):
    problems = examples["problem"]
    beam_results = _dvts(problems, config, llm, prm)

    # group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        results["completions"].append([b.current_text for b in beams])
        results["pred"].append(
            beams[
                np.argmax(
                    [
                        aggregate_scores(b.best_scores, config.agg_strategy)
                        for b in beams
                    ]
                )
            ].current_text
        )
        results["scores"].append([b.best_scores for b in beams])
        results["completion_tokens"].append(-1)

    # TODO: construct and store the tree

    return results
