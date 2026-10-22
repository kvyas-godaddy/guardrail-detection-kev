"""Gradio UI for the Kev guardrail classifier.

Requires a running kev.serve instance (see run_kev.sh), then:

    pip install gradio
    python kev_gradio_guardrail.py
"""

from __future__ import annotations

import gradio as gr

from guardrail_detection_kev import CATEGORIES, CONFIDENCE_THRESHOLD, GuardrailClassifier

# ---------------------------------------------------------------------------
# Shared classifier instance (initialised once on first classify call)
# ---------------------------------------------------------------------------

_classifier: GuardrailClassifier | None = None

def _get_classifier(url: str) -> GuardrailClassifier:
    global _classifier
    if _classifier is None or _classifier.url != url.rstrip("/"):
        _classifier = GuardrailClassifier(url=url)
    return _classifier


# ---------------------------------------------------------------------------
# Category colour map
# ---------------------------------------------------------------------------

CATEGORY_COLORS: dict[str, str] = {
    "none":          "#22c55e",   # green
    "none-weak":     "#86efac",   # light green
    "abuse":         "#f97316",   # orange
    "adult_content": "#a855f7",   # purple
    "harassment":    "#ef4444",   # red
    "hate_speech":   "#dc2626",   # dark red
    "violence":      "#b91c1c",   # deeper red
    "spam":          "#eab308",   # yellow
}


def _category_badge(category: str) -> str:
    color = CATEGORY_COLORS.get(category, "#6b7280")
    return (
        f'<span style="background:{color};color:#fff;padding:4px 12px;'
        f'border-radius:9999px;font-weight:600;font-size:1.1em;">'
        f'{category}</span>'
    )


def _confidence_bar(value: float) -> str:
    pct = int(value * 100)
    color = "#22c55e" if value >= CONFIDENCE_THRESHOLD else "#f97316"
    return (
        f'<div style="background:#e5e7eb;border-radius:4px;height:18px;width:100%;">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px;"></div>'
        f'</div>'
        f'<small>{value:.4f} &nbsp; (threshold {CONFIDENCE_THRESHOLD})</small>'
    )


def _prob_table(probabilities: dict[str, float]) -> str:
    rows = ""
    for cat, prob in sorted(probabilities.items(), key=lambda kv: -kv[1]):
        pct = prob * 100
        color = CATEGORY_COLORS.get(cat, "#6b7280")
        rows += (
            f"<tr>"
            f'<td style="padding:4px 10px;font-weight:500;">{cat}</td>'
            f'<td style="padding:4px 10px;width:200px;">'
            f'<div style="background:#e5e7eb;border-radius:3px;height:14px;">'
            f'<div style="background:{color};width:{pct:.1f}%;height:100%;border-radius:3px;"></div>'
            f'</div></td>'
            f'<td style="padding:4px 10px;">{prob:.4f}</td>'
            f"</tr>"
        )
    return f'<table style="width:100%;border-collapse:collapse;">{rows}</table>'


# ---------------------------------------------------------------------------
# Main classify function wired to Gradio
# ---------------------------------------------------------------------------

def classify(text: str, server_url: str):
    if not text.strip():
        return "<em>Enter some text above and click Classify.</em>", "", "", ""

    try:
        clf = _get_classifier(server_url)
        result = clf.classify(text)
    except SystemExit as e:
        return f'<span style="color:red;">{e}</span>', "", "", ""
    except Exception as e:
        return f'<span style="color:red;">Error: {e}</span>', "", "", ""

    category    = result["category"]
    raw         = result["raw_category"]
    confidence  = result["confidence"]
    latency_ms  = result["latency_ms"]
    probs       = result["probabilities"]

    # -- Category badge --
    badge = _category_badge(category)
    if category == "none-weak":
        badge += (
            f' <small style="color:#6b7280;">'
            f'(model predicted <strong>{raw}</strong>, '
            f'confidence below threshold)</small>'
        )

    # -- Confidence bar --
    conf_html = _confidence_bar(confidence)

    # -- Latency --
    latency_html = f'<span style="font-size:0.95em;">{latency_ms:.1f} ms</span>'

    # -- Probability table --
    prob_html = _prob_table(probs)

    return badge, conf_html, latency_html, prob_html


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="Kev Guardrail Classifier") as demo:
    gr.Markdown("## Kev Guardrail Classifier")
    gr.Markdown(
        "Classifies text into: **"
        + "**, **".join(CATEGORIES.keys())
        + "**. "
        f"Confidence below **{CONFIDENCE_THRESHOLD}** is treated as `none-weak`."
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Input text",
                placeholder="Type or paste text here…",
                lines=4,
            )
            server_url = gr.Textbox(
                label="Kev server URL",
                value="http://127.0.0.1:8008",
            )
            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=2):
            out_category   = gr.HTML(label="Category")
            out_confidence = gr.HTML(label="Confidence")
            out_latency    = gr.HTML(label="Latency")
            out_probs      = gr.HTML(label="Class probabilities")

    classify_btn.click(
        fn=classify,
        inputs=[text_input, server_url],
        outputs=[out_category, out_confidence, out_latency, out_probs],
    )

    # Also classify on Shift+Enter inside the text box
    text_input.submit(
        fn=classify,
        inputs=[text_input, server_url],
        outputs=[out_category, out_confidence, out_latency, out_probs],
    )


if __name__ == "__main__":
    import os
    # SageMaker Studio proxies local ports at a subpath. Setting root_path tells
    # Gradio the correct base URL so its CSS/JS assets resolve through the proxy.
    # Falls back to "" when running outside Studio (direct localhost access).
    studio_domain = os.environ.get("SAGEMAKER_APP_TYPE_LOWERCASE", "")
    root_path = "/codeeditor/default/ports/7860" if studio_domain else ""
    demo.launch(share=False, root_path=root_path)
