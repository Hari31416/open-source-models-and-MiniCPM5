# Open Source Model Performance and Efficiency Findings

## Overview of the Evaluation

This report synthesizes findings from an analysis of 362 open source language models cataloged by Artificial Analysis. The evaluation focuses on how parameter scaling interacts with test-time reasoning, Mixture of Experts (MoE) architectures, latency, pricing, and domain benchmarks.

The dataset partitions models across four canonical parameter tiers:
- Tiny: Fewer than 5 billion parameters (40 models evaluated).
- Small: 5 billion to 27 billion parameters (115 models evaluated).
- Medium: 28 billion to 124 billion parameters (53 models evaluated).
- Large: More than 124 billion parameters (92 models evaluated).

Interactive visualizations documenting these findings are available in [index.html](index.html).

## Core Findings from the Visualizations

### Weight Class Defiance and Capability Inversion

Parameter volume has ceased to be a dependable indicator of model intelligence. Across the 362 models, upper-quartile models in lower weight tiers systematically outperform median and upper-quartile models in tiers above them.

- Tiny models (< 5B) exhibit an intelligence index span from 4.83 to 14.28. The top Tiny model, `MiniCPM5-2B` (Score 14.28), surpasses the median of the Small tier (7.43) and the median of the Medium tier (7.66).
- Small models (5B - 27B) span from 3.75 to 33.90. The top Small model, `Qwen3.8 27B (xhigh)` (Score 33.90), exceeds the 90th percentile of the Large tier (26.34) and beats 90.2% of all evaluated models exceeding 124B parameters.
- Medium models (28B - 124B) span from 5.03 to 24.94. The top Medium model, `Ling 3.0 Flash` (124B total, Score 24.94), beats 75.0% of models in the Large tier.
- Distributional overlaps show that a top 2B to 8B parameter model deployed today delivers higher empirical capability than standard 70B dense models released in late 2024 and early 2025.

Visual evidence is detailed in [chart1_tier_distributions.html](chart1_tier_distributions.html).

### The Reasoning Premium Across Tiers

The bifurcation between reasoning (chain-of-thought / test-time compute) and non-reasoning architectures is the primary driver of tier defiance.

- In the Tiny tier (< 5B), reasoning models achieve a median score of 7.14 versus 5.19 for non-reasoning models (a +1.94 point advantage).
- In the Small tier (5B - 27B), reasoning models achieve a median score of 9.47 versus 6.20 for non-reasoning models (a +3.27 point advantage).
- In the Medium tier (28B - 124B), reasoning models achieve a median score of 10.49 versus 6.70 for non-reasoning models (a +3.79 point advantage).
- In the Large tier (> 124B), reasoning models achieve a median score of 22.44 versus 11.55 for non-reasoning models (a +10.89 point advantage).

Non-reasoning models hit an asymptotic ceiling near 11.5 points regardless of parameter size. To exceed an Intelligence Index of 15, test-time reasoning is mandatory.

Visual evidence is detailed in [chart2_parameter_efficiency_scatter.html](chart2_parameter_efficiency_scatter.html).

### Scorecard of Outperforming Models

Evaluating models against discrete weight classes reveals the following performance benchmarks:

- `MiniCPM5-2B` (2.6B, Score 14.3): Beats 100% of Tiny models (< 5B), 90.4% of Small models (5B - 27B), 94.3% of Medium models (28B - 124B), and 40.2% of Large models (> 124B).
- `Ling 3.0 Tiny` (7.9B total / 1.3B active, Score 11.9): Beats 97.5% of Tiny models, 80.7% of Small models, 90.6% of Medium models, and 30.4% of Large models.
- `Granite 4.2 8B` (8.0B, Score 12.4): Beats 97.5% of Tiny models, 83.3% of Small models, 92.5% of Medium models, and 32.6% of Large models.
- `Qwen3.5 4B (Reasoning)` (4.66B, Score 13.1): Beats 97.5% of Tiny models, 85.1% of Small models, 92.5% of Medium models, and 37.0% of Large models.
- `G9v3-39A5B` (39.0B total / 5.0B active, Score 21.8): Beats 100% of Tiny models, 96.5% of Small models, 98.1% of Medium models, and 60.9% of Large models.
- `Qwen3.6 35B A3B (Reasoning)` (36.0B total / 3.0B active, Score 22.3): Beats 100% of Tiny models, 98.2% of Small models, 98.1% of Medium models, and 63.0% of Large models.
- `Qwen3.5 27B (Reasoning)` (27.8B, Score 22.9): Beats 100% of Tiny models, 99.1% of Small models, 98.1% of Medium models, and 68.5% of Large models.
- `Ling 3.0 Flash` (124.0B total / 6.0B active, Score 24.9): Beats 100% of Tiny models, 99.1% of Small models, 100% of Medium models, and 75.0% of Large models.
- `Qwen3.8 27B (xhigh)` (27.0B, Score 33.9): Beats 100% of Tiny models, 100% of Small models, 100% of Medium models, and 90.2% of Large models.

Visual evidence is detailed in [chart3_higher_tiers_beaten.html](chart3_higher_tiers_beaten.html).

### Active Parameter Efficiency and Sparse Architectures

Evaluating intelligence index points per active parameter billion separates architectural density from conditional routing:

- Top Dense Efficiency Champion: `MiniCPM5-2B` achieves 5.49 points per active billion (Score 14.28 / 2.6B). It is the most parameter-efficient dense model in the dataset.
- Ultra-Sparse MoE Routing: `Ling 3.0 Tiny` activates only 1.3B parameters per forward token pass, generating 9.13 points per active billion. `Qwen3.6 35B A3B` activates 3.0B parameters, achieving 7.42 points per active billion.
- Frontier MoE Diminishing Returns: While large MoEs like `Qwen3.8-Flash-Next` (180B total / 6B active, Score 42.24) generate 7.04 points per active billion, massive dense models like `Llama 3.1 405B` (Score 7.50 / 405B) drop to 0.02 points per active billion. Dense scaling past 70B without reasoning yields minimal active-parameter return.

Visual evidence is detailed in [chart4_parameter_efficiency_index.html](chart4_parameter_efficiency_index.html).

### MiniCPM5-2B Benchmark Domain Scorecard

Evaluating `MiniCPM5-2B` (2.6B dense) across individual capability benchmarks reveals both its primary strengths and structural limits when benchmarked against discrete size classes (identical to Chart 3):

- Overall Intelligence Index: Scores 14.3 pts, beating 100% of Tiny (< 5B), 90.4% of Small (5B - 27B), 94.3% of Medium (28B - 124B), and 40.2% of Large (> 124B).
- Multi-Turn Tool Calling (`tauBanking`): MiniCPM5-2B scores 20.8%, beating 100% of Tiny, 84.0% of Small, 91.7% of Medium, and 57.8% of Large.
- Long-Context Reasoning (`LCR`): Scores 59.3%, beating 100% of Tiny, 82.2% of Small, 85.7% of Medium, and 34.6% of Large.
- Humanity's Last Exam (`HLE`): Scores 8.9%, beating 97.4% of Tiny, 73.0% of Small, 68.0% of Medium, and 28.2% of Large.
- Graduate Scientific Reasoning (`GPQA`): Scores 70.2%, beating 97.4% of Tiny, 76.8% of Small, 66.7% of Medium, and 17.4% of Large.
- Economic & Business Value (`GDPval`): Scores 16.4%, beating 100% of Tiny, 84.4% of Small, 78.6% of Medium, and 23.4% of Large.
- Agentic Terminal Execution (`TerminalBench v2.1`): Scores 8.6%, beating 91.7% of Tiny, 40.0% of Small, 21.4% of Medium, and 2.1% of Large.
- Scientific Coding (`SciCode`): Scores 26.3%, beating 100% of Tiny, 25.0% of Small, 12.5% of Medium, and 0.0% of Large.

The heatmap demonstrates that MiniCPM5-2B's tier-defying capability is concentrated in multi-turn tool interaction, deductive logic, and long-context synthesis, while agentic terminal navigation and advanced scientific code simulation remain constrained by its 2.6B parameter ceiling.

Visual evidence is detailed in [chart5_minicpm_benchmark_heatmap.html](chart5_minicpm_benchmark_heatmap.html).

## New Empirical Findings Beyond the Charts

### Historical Trajectory and Capability Compression

Tracking model performance against release dates reveals a rapid compression cycle where small models absorb frontier capability within 12 to 18 months:

- 2023 Baseline (N=10): Overall median score was 5.3, with a peak of 5.7 (`Llama 2 70B`). The top Tiny model recorded 0.0 on standardized reasoning.
- 2024 Expansion (N=44): Overall median score was 6.0, with a peak of 8.5 (`Llama 3.1 70B Instruct`). Tiny models peaked at 6.3 (`Phi-4 Mini`).
- 2025 Inflection (N=174): Test-time compute was applied at scale. The overall median rose to 7.5, with peak large models reaching 22.4. Tiny models reached 8.8 (`MiniCPM5-1B`).
- 2026 Frontier (N=132): The overall median jumped to 16.4. Tiny models peaked at 14.3 (`MiniCPM5-2B`), and Small models reached 33.9 (`Qwen3.8 27B`).

In 2024, achieving an Intelligence Index score of 7.5 required running `Llama 3.1 405B` on an 8-GPU cluster. By late 2026, a 2.6B parameter model running on a phone or laptop GPU (`MiniCPM5-2B`) scores 14.3, effectively doubling the capability while reducing parameter requirements by 99.4%.

### Context Window Democratization

Extended context windows are no longer an exclusive feature of massive models hosted on enterprise infrastructure:

- Tiny tier (< 5B): Minimum context window is 4,096 tokens, median is 128,000 tokens, and maximum is 262,144 tokens (`Ling 3.0 Tiny`).
- Small tier (5B - 27B): Minimum context window is 4,096 tokens, median is 128,000 tokens, and maximum reaches 1,000,000 tokens (`Granite 4.2 30B`, `GLM-4.7-Flash`).
- Medium tier (28B - 124B): Minimum context window is 2,048 tokens, median is 128,000 tokens, and maximum reaches 10,000,000 tokens (`LongCat Flash Lite`).
- Large tier (> 124B): Median context window is 256,000 tokens, with multiple models supporting 1,000,000 tokens (`GLM-5.3`, `DeepSeek V4 Pro`).

Context retention at 128k tokens has become the default standard even for sub-5B architectures, eliminating context limits as a differentiator between small and large models.

### Output Throughput and Latency Optimization

High intelligence scores no longer correspond to slow output speeds:

- `HyperNova 60B 2605` (Medium tier, Score 11.7): Generates 360.6 tokens per second.
- `Ling 3.0 Flash` (Medium tier, Score 24.9): Generates 315.5 tokens per second while outperforming 75% of the Large tier.
- `Trinity Large Thinking` (Large tier, Score 10.9): Generates 313.8 tokens per second.
- `Nemotron 3.5 Lightning` (Small tier, Score 13.6): Generates 299.8 tokens per second.

MoE gating and speculative decoding allow models like `Ling 3.0 Flash` to maintain output speeds in excess of 300 tokens per second while scoring above 24 on the Intelligence Index.

### Inference Cost Disruption

Analysis of blended API pricing (70% input, 20% cached input, 10% output) indicates that cost per token has collapsed for high-reasoning workloads:

- `Qwen3.5 4B (Reasoning)` (Score 13.1): Costs $0.042 per 1M blended tokens.
- `Ling 3.0 Flash` (Score 24.9): Costs $0.048 per 1M blended tokens.
- `Granite 4.2 8B` (Score 12.4): Costs $0.048 per 1M blended tokens.
- `Hy3-preview (Reasoning)` (Score 22.7): Costs $0.048 per 1M blended tokens.

Workloads requiring reasoning scores above 20 previously incurred API costs exceeding $3.00 to $10.00 per million tokens. Current open-weight models deliver this intelligence bracket below $0.05 per million tokens.

### Ecosystem and Creator Profiles

The dataset includes models from 48 distinct creators. Eight organizations account for the majority of top-tier releases:

- Alibaba (75 models): Average intelligence score of 11.8, with peak score of 42.2 (`Qwen3.8-Flash-Next`). Alibaba dominates representation across every weight bracket from 0.6B to 2.4T.
- DeepSeek (33 models): Average intelligence score of 14.6, with peak score of 36.3 (`DeepSeek V4 Pro`). DeepSeek maintains the highest average capability floor among organizations with more than 20 releases.
- Z AI (20 models): Average intelligence score of 20.3, with peak score of 44.9 (`GLM-5.3 max`). Z AI holds the highest absolute intelligence score in the entire open source dataset.
- NVIDIA (19 models): Average intelligence score of 9.3, with peak score of 23.4 (`Nemotron 3 Ultra 550B A55B`). Strong focus on speculative decoding and high-throughput small models.
- Google (19 models): Average intelligence score of 8.5, with peak score of 16.7 (`Gemma 4 26B A4B`).
- Meta (17 models): Average intelligence score of 6.8, with peak score of 18.1 (`Muse Glimmer`). Meta models in this dataset reflect predominantly dense non-reasoning architectures, yielding lower average intelligence index ratings.
- IBM (13 models): Average intelligence score of 7.1, with peak score of 14.8 (`Granite 4.2 30B`).

### Benchmark Specifics: Agentic Tooling and Scientific Reasoning

Examining sub-benchmarks highlights where small reasoning models excel and where parameter scale remains necessary:

- Agentic Terminal Execution (`TerminalBench v2.1`): Dominated by large frontier models. Top performers include `Qwen3.8-Flash-Next` (86.1%), `Kimi K3 (max)` (85.0%), and `GLM-5.3-Flash` (84.3%). Complex multi-step bash tool use, recovery from errors, and environment debugging still favor models with large memory capacity.
- Graduate-Level Reasoning (`GPQA`): Small reasoning models achieve near-parity with frontier giants. While `GLM-5.3` achieves 91.7%, `Qwen3.8 27B` reaches 90.5%, `Qwen3.5 27B` reaches 85.8%, and the 3.9B model `Nanbeige4.1-3B` scores 84.9%. Pure mathematical and deductive logic can be compressed into compact reasoning models via targeted RL.

### Hallucination and Calibration Dynamics

Evaluating the dataset's `omniscienceBreakdown` reveals an inverse relationship between model size and hallucination rates in non-reasoning models:

- Extreme Hallucination in Standard Small Models: Compact non-reasoning models exhibit hallucination rates exceeding 95% under stress (`Qwen3.5 0.8B Non-reasoning` at 98.5%, `Gemma 3 4B Instruct` at 98.2%). They generate fluent text without factual calibration.
- Hallucination Suppression via Reasoning Distillation: In contrast, models trained with rigorous reasoning verifiers achieve very low hallucination rates despite small parameter footprints (`MiniCPM5-1B` at 0.9%, `G9v3-3B` at 11.7%, `G9v3-39A5B` at 13.0%).

Reasoning training not only increases benchmark scores but also acts as an internal factual verification mechanism.

## Strategic Recommendations

- Prefer active-parameter footprint over total parameter count when provisioning memory and compute. Sparse MoEs like `Ling 3.0 Tiny` (1.3B active) and `Qwen3.6 35B A3B` (3.0B active) offer higher throughput and lower VRAM requirements than equivalent dense small models.
- Replace legacy 70B dense models with modern 2B to 8B reasoning models (`MiniCPM5-2B`, `Granite 4.2 8B`, `Qwen3.5 4B Reasoning`) for local or edge deployment.
- Reserve large models (> 124B) primarily for agentic coding and environment interaction (`TerminalBench`), where large state tracking remains advantageous over compact architectures.

## Data Access and Setup

The benchmark dataset is sourced from [Artificial Analysis](https://artificialanalysis.ai/).

To set up the data locally:

1. Download the open source models JSON dataset from [Artificial Analysis](https://artificialanalysis.ai).
2. Create the local data directory:

```bash
mkdir -p data
```

3. Place the JSON payload at:

```text
data/open-source-models.json
```

## Reproducing the Analysis and Charts

The chart generation pipeline uses Python and `uv`:

```bash
uv run python3 generate_charts.py
```

The script outputs the following visual assets:

- `index.html`: Unified interactive dashboard with a dropdown view selector.
- `chart1_tier_distributions.html` and `chart1_tier_distributions.png`: Weight class distributions and outlier leaps.
- `chart2_parameter_efficiency_scatter.html` and `chart2_parameter_efficiency_scatter.png`: Parameter count versus Intelligence Index scatter plot.
- `chart3_higher_tiers_beaten.html` and `chart3_higher_tiers_beaten.png`: Scorecard of lower-tier models beating higher-tier medians.
- `chart4_parameter_efficiency_index.html` and `chart4_parameter_efficiency_index.png`: Intelligence points per active parameter billion.
- `chart5_minicpm_benchmark_heatmap.html` and `chart5_minicpm_benchmark_heatmap.png`: MiniCPM-4-2B domain capability heatmap across discrete weight classes.
