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

import numpy as np
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores

# Resource Allocation
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


def best_of_n(x, config: Config, llm: LLM, prm: PRM):
    tokenizer = llm.get_tokenizer()

    # 构造 logits processor
    processors = []
    if config.processor == "cyclical":
        processors.append(cyclical_processor(tokenizer=tokenizer, **config.processor_kwargs))
    if config.processor == "add":
        processors.append(add_processor(tokenizer=tokenizer, **config.processor_kwargs))
    logits_processor = LogitsProcessorList(processors)

    # ✅ 自动获取 prompt 字段（支持 "problem" 或 "question"）
    if "problem" in x:
        prompts = x["problem"]
    elif "question" in x:
        prompts = x["question"]
    else:
        raise KeyError(f"Expected 'problem' or 'question' in input, but got keys: {x.keys()}")

    convs = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]

    if config.custom_chat_template is not None:
        tokenizer.chat_template = config.custom_chat_template

    templated_convs = tokenizer.apply_chat_template(
        convs, tokenize=False, add_generation_prompt=True
    )

    # Duplicate convs
    templated_convs = [c for conv in templated_convs for c in [conv] * config.n]

    completions = [[] for _ in range(len(prompts))]
    completion_tokens = [[] for _ in range(len(prompts))]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,
        logits_processors=logits_processor,
    )

    responses = llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    if len(responses) != len(prompts) * config.n:
        raise ValueError(f"Generated {len(responses)} responses instead of {len(prompts) * config.n}")

    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]
        completion_tokens[i] = [
            len(output.token_ids)
            for r in responses[i * config.n : (i + 1) * config.n]
            for output in r.outputs
        ]

    for c in completions:
        if len(c) != config.n:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n}")

    scores = prm.score(prompts, completions)
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
    ]

    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    x["completions"] = completions
    x["scores"] = scores
    x["pred"] = pred
    x["completion_tokens"] = completion_tokens

    return x
