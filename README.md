<div align='center'>
 
# CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling

[![Venue: ICLR 2026](https://img.shields.io/badge/Venue-ICLR%202026-green)](https://iclr.cc/virtual/2026/poster/10011528)
[![issues](https://img.shields.io/badge/Issues-Welcome!-yellow)](https://github.com/OPTML-Group/CyclicReflex/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/OPTML-Group/CyclicReflex?tab=MIT-1-ov-file)
[![GitHub top language](https://img.shields.io/github/languages/top/OPTML-Group/CyclicReflex)](https://github.com/OPTML-Group/CyclicReflex)
[![GitHub repo size](https://img.shields.io/github/repo-size/OPTML-Group/CyclicReflex)](https://github.com/OPTML-Group/CyclicReflex)
[![GitHub stars](https://img.shields.io/github/stars/OPTML-Group/CyclicReflex)](https://github.com/OPTML-Group/CyclicReflex)
</div>

<table align="center">
  <tr>
    <td align="center"> 
      <img src="Images/teaser.png" alt="Teaser" style="width: 700px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> Schematic overview of CyclicReflex. The
rightmost subfigure presents a comparison of final answer accuracy between
CyclicReflex, the original LRM, and decoding
variants using TIP and S1.</em>
    </td>
  </tr>
</table>

This is the official code repository for the ICLR 2026 paper [CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling](https://arxiv.org/abs/2506.11077).

## Abstract
Large reasoning models (LRMs), such as OpenAI’s o1 and DeepSeek-R1, harness test-time scaling to perform multi-step reasoning for complex problem-solving. This reasoning process, executed before producing final answers, is often guided by special juncture tokens that prompt self-evaluative reflection. These transition markers and reflective cues are referred to as “reflection tokens” (e.g., “wait”, “but”, “alternatively”). In this work, we treat reflection tokens as a “resource” and introduce the problem of resource allocation, aimed at improving the test-time compute performance of LRMs by adaptively regulating the frequency and placement of reflection tokens. Through empirical analysis, we show that both excessive and insufficient use of reflection tokens, referred to as over-reflection and under-reflection, can degrade model performance. To better understand this trade-off, we draw an analogy between reflection token usage and learning rate scheduling in optimization. Building on this insight, We propose cyclical reflection token scheduling (termed CyclicReflex), a training-free decoding strategy that dynamically modulates reflection token logits with a bidirectional, position-dependent triangular waveform, incurring no additional computation cost. Experiments on MATH500, AIME2024/2025, AMC2023, GPQA Diamond and LiveCodeBench demonstrate that CyclicReflex consistently improves performance across model sizes (1.5B–14B), outperforming standard decoding and recent approaches such as TIP (thought switching penalty) and S1.

## Getting Started

* [Overall performance of CyclicReflex.](Base)
* [Integration with other test-time scaling methods.](TestTimeScaling)


## Contributors
* [Chongyu Fan](https://chongyu-fan.netlify.app/)

## Cite This Work
```
@inproceedings{fancyclicreflex,
  title={CyclicReflex: Improving Reasoning Models via Cyclical Reflection Token Scheduling},
  author={Fan, Chongyu and Zhang, Yihua and Jia, Jinghan and Hero, Alfred O and Liu, Sijia},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```