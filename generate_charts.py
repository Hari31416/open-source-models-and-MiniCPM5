import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Tuple

import plotly.graph_objects as go

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Light theme color palette
COLOR_MAP: Dict[str, str] = {
    "tiny": "#059669",  # Emerald 600
    "small": "#2563EB",  # Blue 600
    "medium": "#D97706",  # Amber 600
    "large": "#7C3AED",  # Purple 600
}

ORDERED_TIERS: List[str] = ["tiny", "small", "medium", "large"]
TIER_SHORT_LABELS: Dict[str, str] = {
    "tiny": "Tiny (< 5B)",
    "small": "Small (5B - 27B)",
    "medium": "Medium (28B - 124B)",
    "large": "Large (> 124B)",
}


def clean_display_name(name: str) -> str:
    """Shorten verbose model names for clean chart presentation."""
    name = re.sub(r"\(Reasoning,\s*Max\s*Effort\)", "(max)", name, flags=re.IGNORECASE)
    name = re.sub(
        r"\(Reasoning,\s*High\s*Effort\)", "(high)", name, flags=re.IGNORECASE
    )
    name = re.sub(r"\(Reasoning\)", "(reas)", name, flags=re.IGNORECASE)
    name = re.sub(r"\(Non-reasoning\)", "(base)", name, flags=re.IGNORECASE)
    return name.strip()


def extract_canonical_name(name: str) -> str:
    """Extract canonical base model name by removing reasoning effort tags."""
    pattern = (
        r"\s*\((xhigh|high|medium|low|max|min|Reasoning|Non-reasoning|effort:[^)]+)\)"
    )
    return re.sub(pattern, "", name, flags=re.IGNORECASE).strip()


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load and validate dataset from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "models" not in data:
        raise ValueError("Key 'models' missing from dataset")

    return data["models"]


def filter_best_effort_models(
    raw_models: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deduplicate models to the best-performing reasoning effort per model family."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for model in raw_models:
        score = model.get("intelligenceIndex")
        size_class = model.get("sizeClass")
        if score is not None and size_class in ORDERED_TIERS:
            base_name = extract_canonical_name(model.get("name", ""))
            grouped.setdefault(base_name, []).append(model)

    best_effort_models: List[Dict[str, Any]] = []
    for base_name, variants in grouped.items():
        best_variant = max(
            variants, key=lambda item: float(item.get("intelligenceIndex", 0.0))
        )
        best_effort_models.append(best_variant)

    logger.info("Retained %d best-effort model families", len(best_effort_models))
    return best_effort_models


def calculate_tier_percentiles(
    models: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    """Group model scores by size class."""
    scores_by_tier: Dict[str, List[float]] = {tier: [] for tier in ORDERED_TIERS}
    for model in models:
        tier = model.get("sizeClass")
        score = model.get("intelligenceIndex")
        if tier in scores_by_tier and score is not None:
            scores_by_tier[tier].append(float(score))

    for tier, scores in scores_by_tier.items():
        scores.sort()
        logger.info(
            "Tier %s has %d models, median = %.2f",
            tier,
            len(scores),
            scores[len(scores) // 2] if scores else 0.0,
        )

    return scores_by_tier


def calculate_percent_beaten(
    score: float, tier_scores: List[float], is_self_in_pool: bool = False
) -> float:
    """Calculate percentage of models beaten in a given tier or pool."""
    if not tier_scores:
        return 0.0
    total = len(tier_scores) - (1 if is_self_in_pool else 0)
    if total <= 0:
        return 100.0
    strictly_beaten = sum(1 for s in tier_scores if score > s)
    return round((strictly_beaten / total) * 100.0, 1)


def build_chart1_tier_distributions(
    models: List[Dict[str, Any]], output_dir: str
) -> Tuple[go.Figure, str]:
    """Chart 1: Box and strip plot showing tier distributions with outlier callout lines."""
    fig = go.Figure()

    for tier in ORDERED_TIERS:
        tier_models = [m for m in models if m.get("sizeClass") == tier]
        scores = [float(m["intelligenceIndex"]) for m in tier_models]
        names = [m.get("name", "") for m in tier_models]
        params = [m.get("parameters", "N/A") for m in tier_models]
        reasoning = [
            "Reasoning" if m.get("isReasoning") else "Non-reasoning"
            for m in tier_models
        ]

        fig.add_trace(
            go.Box(
                y=scores,
                name=TIER_SHORT_LABELS[tier],
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker=dict(
                    color=COLOR_MAP[tier],
                    size=7,
                    opacity=0.7,
                    line=dict(width=1, color="#FFFFFF"),
                ),
                line=dict(color=COLOR_MAP[tier], width=2),
                fillcolor="rgba(241, 245, 249, 0.4)",
                customdata=list(zip(names, params, reasoning)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Tier: " + tier.title() + "<br>"
                    "Intelligence Index: %{y:.2f}<br>"
                    "Parameters: %{customdata[1]}B<br>"
                    "Type: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    # MiniCPM5-2B reference line across Small and Medium
    minicpm = next((m for m in models if "MiniCPM5-2B" in m.get("name", "")), None)
    if minicpm:
        score_val = float(minicpm["intelligenceIndex"])
        fig.add_shape(
            type="line",
            x0=-0.3,
            x1=2.3,
            y0=score_val,
            y1=score_val,
            line=dict(color="#059669", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=2.2,
            y=score_val + 1.0,
            text=f"MiniCPM5-2B ({score_val:.1f}): Beats 90.4% Small, 94.3% Medium",
            showarrow=False,
            font=dict(color="#065F46", size=12, family="Inter, sans-serif"),
            bgcolor="#ECFDF5",
            bordercolor="#059669",
            borderwidth=1.5,
            borderpad=5,
        )

    # Qwen3.8 27B reference line across Medium and Large
    qwen27b = next((m for m in models if "Qwen3.8 27B" in m.get("name", "")), None)
    if qwen27b:
        score_val = float(qwen27b["intelligenceIndex"])
        fig.add_shape(
            type="line",
            x0=0.7,
            x1=3.3,
            y0=score_val,
            y1=score_val,
            line=dict(color="#2563EB", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=2.8,
            y=score_val + 1.0,
            text=f"Qwen3.8 27B ({score_val:.1f}): Beats 100% Medium, 90.2% Large",
            showarrow=False,
            font=dict(color="#1E40AF", size=12, family="Inter, sans-serif"),
            bgcolor="#EFF6FF",
            bordercolor="#2563EB",
            borderwidth=1.5,
            borderpad=5,
        )

    fig.update_layout(
        title=dict(
            text="Weight Class Distributions and Outlier Leaps (Intelligence Index)",
            font=dict(size=18, color="#0F172A", family="Inter, sans-serif"),
            x=0.03,
            y=0.96,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#0F172A", family="Inter, sans-serif"),
        yaxis=dict(
            title="Artificial Analysis Intelligence Index",
            gridcolor="#E2E8F0",
            zerolinecolor="#CBD5E1",
            range=[0, 48],
        ),
        xaxis=dict(
            title="Model Size Tier",
            gridcolor="#E2E8F0",
        ),
        showlegend=False,
        margin=dict(l=70, r=40, t=80, b=60),
    )

    out_file = os.path.join(output_dir, "chart1_tier_distributions.html")
    fig.write_html(out_file, include_plotlyjs="cdn")
    png_file = os.path.join(output_dir, "chart1_tier_distributions.png")
    fig.write_image(png_file, width=1200, height=700, scale=2)
    logger.info("Saved Chart 1 to %s and %s", out_file, png_file)
    return fig, out_file


def build_chart2_parameter_efficiency(
    models: List[Dict[str, Any]], output_dir: str
) -> Tuple[go.Figure, str]:
    """Chart 2: Parameter size (log scale) vs Intelligence Index scatter plot."""
    fig = go.Figure()

    # Tier median horizontal reference lines
    tier_scores = calculate_tier_percentiles(models)
    for tier in ORDERED_TIERS:
        scores = tier_scores[tier]
        if scores:
            median_val = scores[len(scores) // 2]
            fig.add_hline(
                y=median_val,
                line_dash="dot",
                line_color=COLOR_MAP[tier],
                line_width=1.5,
                annotation_text=f"{tier.title()} Median: {median_val:.1f}",
                annotation_position="bottom right",
                annotation_font=dict(color=COLOR_MAP[tier], size=11),
            )

    # Plot points by size tier
    for tier in ORDERED_TIERS:
        tier_models = [
            m
            for m in models
            if m.get("sizeClass") == tier
            and m.get("parameters") is not None
            and float(m.get("parameters", 0)) > 0
        ]
        params = [float(m["parameters"]) for m in tier_models]
        scores = [float(m["intelligenceIndex"]) for m in tier_models]
        names = [m.get("name", "") for m in tier_models]
        symbols = ["diamond" if m.get("isReasoning") else "circle" for m in tier_models]
        reasoning_text = [
            "Reasoning" if m.get("isReasoning") else "Non-reasoning"
            for m in tier_models
        ]

        fig.add_trace(
            go.Scatter(
                x=params,
                y=scores,
                mode="markers",
                name=TIER_SHORT_LABELS[tier],
                marker=dict(
                    color=COLOR_MAP[tier],
                    size=9,
                    symbol=symbols,
                    opacity=0.8,
                    line=dict(width=1, color="#FFFFFF"),
                ),
                customdata=list(zip(names, reasoning_text)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Tier: " + tier.title() + "<br>"
                    "Parameters: %{x}B<br>"
                    "Intelligence Index: %{y:.2f}<br>"
                    "Type: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    # Explicit clean log scale tick values and labels
    tick_vals = [0.5, 1, 2, 4, 8, 16, 32, 70, 140, 300, 700, 1500, 3000]
    tick_texts = [
        "0.5B",
        "1B",
        "2B",
        "4B",
        "8B",
        "16B",
        "32B",
        "70B",
        "140B",
        "300B",
        "700B",
        "1.5T",
        "3T",
    ]

    # Prominent callouts with clean arrows anchored via pixel offsets
    annotations = [
        (
            "MiniCPM5-2B",
            2.6,
            14.28,
            "<b>MiniCPM5-2B (2.6B)</b><br>Score 14.3 | Beats 94.3% Medium",
            75,
            -45,
            "#059669",
            "#ECFDF5",
        ),
        (
            "G9v3-3B",
            3.0,
            10.8,
            "<b>G9v3-3B (3B)</b><br>Score 10.8 | Beats 77.4% Medium",
            75,
            35,
            "#059669",
            "#ECFDF5",
        ),
        (
            "Qwen3.8 27B (xhigh)",
            27.0,
            33.9,
            "<b>Qwen3.8 27B (27B)</b><br>Score 33.9 | Beats 90.2% Large",
            -85,
            -35,
            "#2563EB",
            "#EFF6FF",
        ),
        (
            "Ling 3.0 Flash",
            124.0,
            24.9,
            "<b>Ling 3.0 Flash (124B)</b><br>Score 24.9 | Beats 75% Large",
            -85,
            -35,
            "#D97706",
            "#FFFBEB",
        ),
    ]

    for (
        model_name,
        p_val,
        s_val,
        note,
        ax_val,
        ay_val,
        border_col,
        bg_col,
    ) in annotations:
        fig.add_annotation(
            x=math.log10(p_val),
            y=s_val,
            text=note,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor="#475569",
            axref="pixel",
            ayref="pixel",
            ax=ax_val,
            ay=ay_val,
            font=dict(color="#0F172A", size=11, family="Inter, sans-serif"),
            bgcolor=bg_col,
            bordercolor=border_col,
            borderwidth=1.5,
            borderpad=5,
        )

    fig.update_layout(
        title=dict(
            text="Parameter Efficiency: Active Billions vs Intelligence Index",
            font=dict(size=18, color="#0F172A", family="Inter, sans-serif"),
            x=0.03,
            y=0.96,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#0F172A", family="Inter, sans-serif"),
        xaxis=dict(
            title="Parameters in Billions (Log Scale)",
            type="log",
            range=[math.log10(0.4), math.log10(3500)],
            tickvals=tick_vals,
            ticktext=tick_texts,
            gridcolor="#E2E8F0",
            zeroline=False,
        ),
        yaxis=dict(
            title="Artificial Analysis Intelligence Index",
            gridcolor="#E2E8F0",
            zerolinecolor="#CBD5E1",
            range=[0, 48],
        ),
        legend=dict(
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            borderwidth=1,
            entrywidth=140,
            entrywidthmode="pixels",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
        ),
        margin=dict(l=70, r=40, t=80, b=60),
    )

    out_file = os.path.join(output_dir, "chart2_parameter_efficiency_scatter.html")
    fig.write_html(out_file, include_plotlyjs="cdn")
    png_file = os.path.join(output_dir, "chart2_parameter_efficiency_scatter.png")
    fig.write_image(png_file, width=1200, height=700, scale=2)
    logger.info("Saved Chart 2 to %s and %s", out_file, png_file)
    return fig, out_file


def build_chart3_higher_tiers_beaten(
    models: List[Dict[str, Any]], output_dir: str
) -> Tuple[go.Figure, str]:
    """Chart 3: Heatmap Scorecard showing cumulative percentage of models beaten across weight classes."""
    scores_by_tier = calculate_tier_percentiles(models)

    # Cumulative pools so higher tiers strictly encompass larger model brackets
    tiny_scores = scores_by_tier["tiny"]
    small_scores = scores_by_tier["small"]
    medium_scores = scores_by_tier["medium"]
    large_scores = scores_by_tier["large"]

    # Standout overachiever candidates
    targets = [
        ("Qwen3.8 27B (xhigh)", "small", 27.0),
        ("Ling 3.0 Flash", "medium", 124.0),
        ("Qwen3.5 27B (Reasoning)", "small", 27.8),
        ("Qwen3.6 35B A3B (Reasoning)", "small", 36.0),
        ("G9v3-39A5B", "small", 39.0),
        ("Gemma 4 26B A4B (Reasoning)", "small", 25.2),
        ("Mistral Medium 3.5", "medium", 128.0),
        ("MiniCPM5-2B", "tiny", 2.6),
        ("Gemma 4 12B (Reasoning)", "small", 12.0),
        ("Apriel-v1.5-15B-Thinker", "small", 15.0),
        ("Qwen3.5 9B (Reasoning)", "small", 9.65),
        ("Qwen3.5 4B (Reasoning)", "small", 4.66),
        ("Granite 4.2 8B", "small", 8.0),
        ("Ling 3.0 Tiny", "small", 7.9),
        ("G9v3-3B", "tiny", 3.0),
        ("Granite 4.2 3B", "tiny", 3.0),
        ("MiniCPM5-1B (Reasoning)", "tiny", 1.0),
    ]

    items: List[Dict[str, Any]] = []
    for name, own_tier, param_val in targets:
        m = next((x for x in models if name in x.get("name", "")), None)
        if not m:
            continue
        score = float(m["intelligenceIndex"])
        pct_tiny = calculate_percent_beaten(
            score, tiny_scores, is_self_in_pool=(own_tier == "tiny")
        )
        pct_small = calculate_percent_beaten(
            score, small_scores, is_self_in_pool=(own_tier == "small")
        )
        pct_medium = calculate_percent_beaten(
            score, medium_scores, is_self_in_pool=(own_tier == "medium")
        )
        pct_large = calculate_percent_beaten(
            score, large_scores, is_self_in_pool=(own_tier == "large")
        )
        items.append(
            {
                "label": f"{clean_display_name(m['name'])} ({param_val}B, Score {score:.1f})",
                "own_tier": own_tier,
                "score": score,
                "pct_tiny": pct_tiny,
                "pct_small": pct_small,
                "pct_medium": pct_medium,
                "pct_large": pct_large,
            }
        )

    # Sort descending by score
    items.sort(key=lambda x: x["score"], reverse=True)

    y_labels = [it["label"] for it in reversed(items)]
    x_labels = [
        "Tiny (< 5B)",
        "Small (5B - 27B)",
        "Medium (28B - 124B)",
        "Large (> 124B)",
    ]

    z_data: List[List[float]] = []
    text_data: List[List[str]] = []
    for it in reversed(items):
        row = [
            it["pct_tiny"],
            it["pct_small"],
            it["pct_medium"],
            it["pct_large"],
        ]
        z_data.append(row)
        formatted_row = []
        for val in row:
            text_color = "#FFFFFF" if val >= 68.0 else "#064E3B"
            formatted_row.append(
                f'<span style="color:{text_color}; font-weight:600;">{val:.0f}%</span>'
            )
        text_data.append(formatted_row)

    # Clean emerald green gradient
    green_colorscale = [
        [0.0, "#F0FDF4"],  # emerald 50
        [0.25, "#BBF7D0"],  # emerald 200
        [0.50, "#4ADE80"],  # emerald 400
        [0.75, "#16A34A"],  # emerald 600
        [1.0, "#065F46"],  # emerald 800
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            text=text_data,
            texttemplate="%{text}",
            textfont=dict(size=12, family="Inter, sans-serif"),
            colorscale=green_colorscale,
            zmin=0,
            zmax=100,
            colorbar=dict(
                title=dict(text="% Beaten", font=dict(color="#0F172A", size=11)),
                tickfont=dict(color="#0F172A", size=10),
                outlinecolor="#CBD5E1",
                outlinewidth=1,
            ),
            xgap=4,
            ygap=4,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Target Category: %{x}<br>"
                "Beats: %{z:.1f}% of models in category<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="The 'Punching Above' Scorecard: Percentage of Models Beaten Across Size Classes",
            font=dict(size=18, color="#0F172A", family="Inter, sans-serif"),
            x=0.03,
            y=0.97,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A", family="Inter, sans-serif"),
        xaxis=dict(
            side="top",
            tickfont=dict(size=12, color="#0F172A"),
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#0F172A"),
            gridcolor="rgba(0,0,0,0)",
        ),
        height=860,
        margin=dict(l=310, r=60, t=90, b=40),
    )

    out_file = os.path.join(output_dir, "chart3_higher_tiers_beaten.html")
    fig.write_html(out_file, include_plotlyjs="cdn")
    png_file = os.path.join(output_dir, "chart3_higher_tiers_beaten.png")
    fig.write_image(png_file, width=1200, height=860, scale=2)
    logger.info("Saved Chart 3 to %s and %s", out_file, png_file)
    return fig, out_file


def build_chart4_parameter_efficiency_index(
    models: List[Dict[str, Any]], output_dir: str
) -> Tuple[go.Figure, str]:
    """Chart 4: Parameter Efficiency Index (Score per Active Billion Parameters), highlighting MiniCPM5-2B."""
    selected_config = [
        # Efficiency leaders (MoE and Dense)
        ("Ling 3.0 Tiny", "moe"),
        ("MiniCPM5-1B (Reasoning)", "dense"),
        ("Qwen3.6 35B A3B (Reasoning)", "moe"),
        ("Qwen3.8-Flash-Next", "moe"),
        ("Qwen3.5 35B A3B (Reasoning)", "moe"),
        ("MiniCPM5-2B", "highlight"),
        ("GLM-4.7-Flash (Reasoning)", "moe"),
        ("Ling 3.0 Flash", "moe"),
        ("Gemma 4 26B A4B (Reasoning)", "moe"),
        ("G9v3-39A5B", "moe"),
        ("North Mini Code", "moe"),
        ("G9v3-3B", "dense"),
        ("Qwen3.5 4B (Reasoning)", "dense"),
        ("DeepSeek V4 Flash 0731 (Reasoning, Max Effort)", "moe"),
        ("Granite 4.2 8B", "dense"),
        ("Qwen3.8 27B (xhigh)", "dense"),
        ("Gemma 4 12B (Reasoning)", "dense"),
        # Reference giants (for contrast)
        ("Kimi K2.6", "giant"),
        ("DeepSeek V3.1 Terminus (Reasoning)", "giant"),
        ("DeepSeek R1 (Jan '25)", "giant"),
        ("Mistral Large 3", "giant"),
        ("Llama 3.3 Instruct 70B", "giant"),
        ("Llama 3.1 Instruct 405B", "giant"),
    ]

    rows: List[Dict[str, Any]] = []
    for key, cat in selected_config:
        m = next((x for x in models if key in x.get("name", "")), None)
        if not m:
            continue
        act = m.get("inferenceParametersActiveBillions")
        tot = m.get("parameters")
        active = (
            float(act) if act is not None else (float(tot) if tot is not None else None)
        )
        score = float(m.get("intelligenceIndex", 0.0))
        if active and active > 0:
            eff = score / active
            is_moe = act is not None and tot is not None and act < tot
            arch_label = (
                f"{active:.1f}B act / {tot:.0f}B tot"
                if is_moe
                else f"{active:.1f}B dense"
            )
            lbl = f"{clean_display_name(m['name'])} ({arch_label})"
            rows.append(
                {
                    "label": lbl,
                    "name": m["name"],
                    "eff": eff,
                    "score": score,
                    "active": active,
                    "total": tot,
                    "is_moe": is_moe,
                    "cat": cat,
                }
            )

    # Sort ascending so highest efficiency sits at the top of the horizontal bar chart
    rows.sort(key=lambda x: x["eff"])

    category_colors = {
        "highlight": "#059669",  # Emerald 600
        "moe": "#D97706",  # Amber 600
        "dense": "#2563EB",  # Blue 600
        "giant": "#94A3B8",  # Slate 400
    }

    bar_colors = [category_colors[r["cat"]] for r in rows]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[r["eff"] for r in rows],
            y=[r["label"] for r in rows],
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(
                    color=[
                        "#064E3B" if r["cat"] == "highlight" else "rgba(0,0,0,0)"
                        for r in rows
                    ],
                    width=[2 if r["cat"] == "highlight" else 0 for r in rows],
                ),
            ),
            text=[f"{r['eff']:.2f} pts/B" for r in rows],
            textposition="outside",
            textfont=dict(size=10, color="#0F172A"),
            customdata=[
                [
                    r["score"],
                    r["active"],
                    r["total"],
                    "MoE" if r["is_moe"] else "Dense",
                ]
                for r in rows
            ],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Parameter Efficiency: <b>%{x:.2f} pts / Active B</b><br>"
                "Intelligence Score: %{customdata[0]:.2f}<br>"
                "Active Parameters: %{customdata[1]}B<br>"
                "Total Parameters: %{customdata[2]}B (%{customdata[3]})<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # Legend traces with explicit entrywidth
    legend_items = [
        ("MiniCPM5-2B (Dense Leader)", "#059669"),
        ("Sparse MoE Models", "#D97706"),
        ("Standard Dense Models", "#2563EB"),
        ("Frontier Giants (Reference)", "#94A3B8"),
    ]
    for name, col in legend_items:
        fig.add_trace(go.Bar(x=[None], y=[None], name=name, marker=dict(color=col)))

    # Spotlight annotation on MiniCPM5-2B
    cpm_idx = next(i for i, r in enumerate(rows) if r["cat"] == "highlight")
    fig.add_annotation(
        x=rows[cpm_idx]["eff"],
        y=rows[cpm_idx]["label"],
        text=(
            "<b>MiniCPM5-2B: 5.49 pts/Active-B</b><br>"
            "13x more efficient than DeepSeek V3.1<br>"
            "270x more efficient than Llama 3.1 405B"
        ),
        showarrow=True,
        arrowhead=2,
        ax=130,
        ay=35,
        bgcolor="#ECFDF5",
        bordercolor="#059669",
        borderwidth=1.5,
        borderpad=6,
        font=dict(size=11, color="#065F46"),
    )

    fig.update_layout(
        title=dict(
            text="Parameter Efficiency Index: Intelligence Score per Active Billion Parameters",
            font=dict(size=18, color="#0F172A", family="Inter, sans-serif"),
            x=0.03,
            y=0.98,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#0F172A", family="Inter, sans-serif"),
        xaxis=dict(
            title="Parameter Efficiency Index (Intelligence Index ÷ Active Parameters in Billions)",
            gridcolor="#E2E8F0",
            range=[0, 11.5],
            dtick=1.0,
        ),
        yaxis=dict(
            gridcolor="#E2E8F0",
            tickfont=dict(size=11),
        ),
        height=860,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5,
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            borderwidth=1,
            entrywidth=160,
            entrywidthmode="pixels",
        ),
        margin=dict(l=320, r=40, t=100, b=60),
    )

    out_file = os.path.join(output_dir, "chart4_parameter_efficiency_index.html")
    fig.write_html(out_file, include_plotlyjs="cdn")
    png_file = os.path.join(output_dir, "chart4_parameter_efficiency_index.png")
    fig.write_image(png_file, width=1200, height=860, scale=2)
    logger.info("Saved Chart 4 to %s and %s", out_file, png_file)
    return fig, out_file


def build_chart5_minicpm_benchmark_heatmap(
    models: List[Dict[str, Any]], output_dir: str
) -> Tuple[go.Figure, str]:
    """Generate Chart 5: MiniCPM5-2B Benchmark Dominance Scorecard across size categories."""
    minicpm = next((x for x in models if x.get("name") == "MiniCPM5-2B"), None)
    if not minicpm:
        logger.error("MiniCPM5-2B not found in models dataset.")
        return go.Figure(), ""

    benchmarks_config = [
        (
            "Multi-Turn Tool Calling (tauBanking)",
            "tauBanking",
            minicpm.get("tauBanking"),
            "{:.1f}%",
            100.0,
        ),
        (
            "Overall Intelligence Index",
            "intelligenceIndex",
            minicpm.get("intelligenceIndex"),
            "{:.1f} pts",
            1.0,
        ),
        ("Long-Context Reasoning (LCR)", "lcr", minicpm.get("lcr"), "{:.1f}%", 100.0),
        (
            "Humanity's Last Exam (HLE)",
            "hle",
            minicpm.get("hle"),
            "{:.1f}%",
            100.0,
        ),
        (
            "Graduate Scientific Reasoning (GPQA)",
            "gpqa",
            minicpm.get("gpqa"),
            "{:.1f}%",
            100.0,
        ),
        (
            "Economic & Business Value (GDPval)",
            "gdpvalNormalized",
            minicpm.get("gdpvalNormalized"),
            "{:.1f}%",
            100.0,
        ),
        (
            "Agentic Terminal Tasks (TerminalBench)",
            "terminalbenchV21",
            minicpm.get("terminalbenchV21"),
            "{:.1f}%",
            100.0,
        ),
        (
            "Scientific Coding (SciCode)",
            "scicode",
            minicpm.get("scicode"),
            "{:.1f}%",
            100.0,
        ),
    ]

    pools = [
        ("Tiny (< 5B)", lambda m: m.get("sizeClass") == "tiny", True),
        ("Small (5B - 27B)", lambda m: m.get("sizeClass") == "small", False),
        ("Medium (28B - 124B)", lambda m: m.get("sizeClass") == "medium", False),
        ("Large (> 124B)", lambda m: m.get("sizeClass") == "large", False),
    ]

    x_labels = [p[0] for p in pools]

    # Process rows
    row_records: List[Dict[str, Any]] = []
    for label, key, raw_val, fmt, mult in benchmarks_config:
        if raw_val is None:
            continue
        disp_val = fmt.format(raw_val * mult if mult != 1.0 else raw_val)
        row_pcts: List[float] = []
        for p_name, p_filter, is_self in pools:
            pool_models = [m for m in models if p_filter(m) and m.get(key) is not None]
            if is_self:
                others = [m for m in pool_models if m.get("name") != "MiniCPM5-2B"]
                denom = len(others)
                beaten = sum(1 for m in others if float(m[key]) < float(raw_val))
            else:
                denom = len(pool_models)
                beaten = sum(1 for m in pool_models if float(m[key]) < float(raw_val))
            pct = (beaten / denom * 100.0) if denom > 0 else 0.0
            row_pcts.append(round(pct, 1))

        row_records.append(
            {
                "y_label": f"<b>{label}</b> (Score: {disp_val})",
                "pcts": row_pcts,
                "sort_val": row_pcts[1],  # Sort by performance in >= Small pool
            }
        )

    # In plotly heatmap, y-axis draws from bottom up, so sort ascending by sort_val
    # This places the strongest capability at the top of the heatmap
    row_records.sort(key=lambda r: r["sort_val"])

    y_labels = [r["y_label"] for r in row_records]
    z_data = [r["pcts"] for r in row_records]

    text_data: List[List[str]] = []
    for row in z_data:
        formatted_row = []
        for val in row:
            text_color = "#FFFFFF" if val >= 68.0 else "#064E3B"
            formatted_row.append(
                f'<span style="color:{text_color}; font-weight:600;">{val:.0f}%</span>'
            )
        text_data.append(formatted_row)

    # Emerald green colorscale
    green_colorscale = [
        [0.0, "#F0FDF4"],
        [0.25, "#BBF7D0"],
        [0.50, "#4ADE80"],
        [0.75, "#16A34A"],
        [1.0, "#065F46"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=x_labels,
            y=y_labels,
            text=text_data,
            texttemplate="%{text}",
            textfont=dict(family="Inter, sans-serif", size=13),
            colorscale=green_colorscale,
            zmin=0,
            zmax=100,
            colorbar=dict(
                title=dict(
                    text="% Beaten",
                    font=dict(color="#0F172A", size=11),
                ),
                tickfont=dict(color="#0F172A", size=10),
                outlinecolor="#CBD5E1",
                outlinewidth=1,
            ),
            hovertemplate="<b>%{y}</b><br>Target Category: %{x}<br>Beats: %{z:.1f}% of models in category<extra></extra>",
            xgap=5,
            ygap=5,
        )
    )

    fig.update_layout(
        title=dict(
            text="MiniCPM5-2B (2.6B) Capability Scorecard: % of Models Beaten Across Size Classes",
            font=dict(size=18, color="#0F172A", family="Inter, sans-serif"),
            x=0.03,
            y=0.97,
        ),
        xaxis=dict(
            side="top",
            tickfont=dict(size=12, color="#0F172A"),
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            tickfont=dict(size=11.5, color="#0F172A"),
            gridcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
        margin=dict(l=350, r=60, t=90, b=40),
        height=620,
    )

    out_file = os.path.join(output_dir, "chart5_minicpm_benchmark_heatmap.html")
    fig.write_html(out_file, include_plotlyjs="cdn")
    png_file = os.path.join(output_dir, "chart5_minicpm_benchmark_heatmap.png")
    fig.write_image(png_file, width=1200, height=620, scale=2)
    logger.info("Saved Chart 5 to %s and %s", out_file, png_file)
    return fig, out_file


def build_unified_dashboard(
    charts_data: List[Tuple[str, str, go.Figure]], output_dir: str
) -> str:
    """Generate a clean unified HTML dashboard embedding all charts with a dropdown selector."""
    dashboard_path = os.path.join(output_dir, "index.html")

    dropdown_options_html = ""
    chart_containers_html = ""

    for idx, (cid, title, fig) in enumerate(charts_data):
        selected_attr = "selected" if idx == 0 else ""
        dropdown_options_html += (
            f'            <option value="{cid}" {selected_attr}>{title}</option>\n'
        )
        display_style = "block" if idx == 0 else "none"
        # Render plot div directly into chart container
        plot_div = fig.to_html(
            include_plotlyjs=False,
            full_html=False,
            config=dict(responsive=True, displayModeBar=True),
        )
        chart_containers_html += f"""
        <div id="{cid}" class="chart-view" style="display: {display_style};">
            <div class="chart-card">
                {plot_div}
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Open-Source Models Punching Above Their Weight Class</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background-color: #F8FAFC;
            color: #0F172A;
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 24px;
        }}
        header {{
            max-width: 1400px;
            margin: 0 auto 20px auto;
        }}
        h1 {{
            font-size: 26px;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 8px;
        }}
        p.subtitle {{
            font-size: 14px;
            color: #64748B;
            line-height: 1.5;
        }}
        .controls-bar {{
            max-width: 1400px;
            margin: 0 auto 20px auto;
            display: flex;
            align-items: center;
            gap: 14px;
            background: #FFFFFF;
            padding: 14px 20px;
            border-radius: 10px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            flex-wrap: wrap;
        }}
        .dropdown-label {{
            font-size: 14px;
            font-weight: 600;
            color: #334155;
            white-space: nowrap;
        }}
        .select-wrapper {{
            position: relative;
            flex: 1;
            min-width: 300px;
            max-width: 600px;
        }}
        .chart-select {{
            width: 100%;
            padding: 10px 40px 10px 16px;
            font-size: 14px;
            font-weight: 500;
            font-family: inherit;
            color: #0F172A;
            background-color: #F8FAFC;
            border: 1.5px solid #CBD5E1;
            border-radius: 8px;
            appearance: none;
            -webkit-appearance: none;
            -moz-appearance: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .chart-select:hover {{
            border-color: #94A3B8;
            background-color: #FFFFFF;
        }}
        .chart-select:focus {{
            outline: none;
            border-color: #2563EB;
            background-color: #FFFFFF;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
        }}
        .select-icon {{
            position: absolute;
            right: 14px;
            top: 50%;
            transform: translateY(-50%);
            pointer-events: none;
            color: #64748B;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .chart-view {{
            width: 100%;
        }}
        .chart-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}
    </style>
</head>
<body>
    <header>
        <h1>Open-Source LLMs: Punching Above Their Weight Class</h1>
        <p class="subtitle">
            Interactive analysis of 362 open-source models from Artificial Analysis. Evaluating peak reasoning performance across size tiers (Tiny, Small, Medium, Large).
        </p>
    </header>
    <div class="controls-bar">
        <label for="chartSelect" class="dropdown-label">Select Chart View:</label>
        <div class="select-wrapper">
            <select id="chartSelect" class="chart-select" onchange="switchChart(this.value)">
{dropdown_options_html}            </select>
            <div class="select-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
            </div>
        </div>
    </div>
    <div class="container">
        {chart_containers_html}
    </div>
    <script>
        function switchChart(chartId) {{
            const views = document.querySelectorAll('.chart-view');
            views.forEach(el => el.style.display = 'none');

            const target = document.getElementById(chartId);
            if (target) {{
                target.style.display = 'block';
                if (window.history && window.history.replaceState) {{
                    window.history.replaceState(null, '', '#' + chartId);
                }}
                // Trigger resize so Plotly refits cleanly to the container
                window.dispatchEvent(new Event('resize'));
            }}
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            const hash = window.location.hash.replace('#', '');
            const selectEl = document.getElementById('chartSelect');
            if (hash && document.getElementById(hash)) {{
                if (selectEl) selectEl.value = hash;
                switchChart(hash);
            }}
        }});
    </script>
</body>
</html>
"""

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Saved unified dashboard to %s", dashboard_path)
    return dashboard_path


def main() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "data", "open-source-models.json")

    logger.info("Starting chart generation from %s", data_file)
    raw_models = load_dataset(data_file)
    best_models = filter_best_effort_models(raw_models)

    fig1, c1 = build_chart1_tier_distributions(best_models, current_dir)
    fig2, c2 = build_chart2_parameter_efficiency(best_models, current_dir)
    fig3, c3 = build_chart3_higher_tiers_beaten(best_models, current_dir)
    fig4, c4 = build_chart4_parameter_efficiency_index(best_models, current_dir)
    fig5, c5 = build_chart5_minicpm_benchmark_heatmap(best_models, current_dir)

    charts_meta = [
        ("c1", "Chart 1: Tier Distributions & Overlaps", fig1),
        ("c2", "Chart 2: Parameter vs Intelligence Scatter", fig2),
        ("c3", "Chart 3: Higher Tiers Beaten Scorecard", fig3),
        ("c4", "Chart 4: Parameter Efficiency Index (Active Params)", fig4),
        ("c5", "Chart 5: MiniCPM5-2B Benchmark Dominance Scorecard", fig5),
    ]
    build_unified_dashboard(charts_meta, current_dir)

    logger.info("All charts and unified dashboard generated successfully")


if __name__ == "__main__":
    main()
