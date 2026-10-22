"""Guardrail classifier (Kev server): classify input text into content safety categories.

Talks to a running kev.serve instance over HTTP (POST /v1/systemone).

1) Start the Kev server:

    cd kev
    KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve \\
        --run /home/sagemaker-user/.cache/huggingface/hub/models--jaredpalmer--kev-4b/snapshots/6cfce5c2fa4b4bd64026336ab649c5ca78857d52 \\
        --port 8008

2) Run this script:

    python guardrail_detection_kev.py classify --text "some text to check"

    python guardrail_detection_kev.py batch \\
        --input texts.csv --text-column text \\
        --output results.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_URL = "http://127.0.0.1:8008"
CONFIDENCE_THRESHOLD = 0.4  # below this the model is considered weak; category is overridden to "none-weak"

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, str] = {
    "none": (
        "The text is normal and does not belong to any harmful category."
    ),
    "abuse": (
        "The text unmistakably contains explicit insults, slurs, or demeaning language."
    ),
    "adult_content": (
        "The text is sexually explicit, directly references pornographic material, or clearly "
        "describes sexual acts in graphic detail."
    ),
    "harassment": (
        "The text directly and personally threatens, intimidates, or attacks a specific named "
        "or clearly identifiable individual — e.g. explicit threats against a named person, "
        "stalking language, or targeted repeated personal insults. "
        "Impersonal commands and business phrases are never harassment."
    ),
    "hate_speech": (
        "The text explicitly attacks or demeans a group based on a protected attribute "
        "(race, ethnicity, religion, gender, sexual orientation, disability). "
        "The hateful intent is unmistakable with no neutral interpretation."
    ),
    "violence": (
        "The text explicitly describes, glorifies, or incites physical violence or harm "
        "against a person or group."
    ),
    "spam": (
        "The text is clearly an unsolicited bulk promotion, phishing message, fake prize "
        "notification, or other deceptive commercial solicitation."
    ),
}

# ---------------------------------------------------------------------------
# Kev question schema
# ---------------------------------------------------------------------------

def build_guardrail_question() -> dict:
    return {
        "category": {
            "type": "choice",
            "instructions": (
                "You are a content safety classifier. "
                "Given the text above, identify the single most applicable content safety category. "
                "Choose 'none' if the text is normal and does not match any harmful category. "
                "Be conservative — only flag clear violations."
            ),
            "criteria": CATEGORIES,
        }
    }


def build_state(text: str) -> dict:
    return {"interaction": text}


# ---------------------------------------------------------------------------
# Guardrail classifier (HTTP client for kev.serve)
# ---------------------------------------------------------------------------

class GuardrailClassifier:
    def __init__(
        self,
        url: str = DEFAULT_URL,
        model: str = "kev-latest",
        timeout: float = 60.0,
    ):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

        try:
            self.session.get(f"{self.url}/v1/models", timeout=10).raise_for_status()
        except requests.RequestException as e:
            raise SystemExit(f"cannot reach kev server at {self.url}: {e}")
        print(f"using kev server at {self.url} (model={self.model})")

    def classify(self, text: str) -> dict:
        state = build_state(text)
        questions = build_guardrail_question()
        t0 = time.perf_counter()
        resp = self.session.post(
            f"{self.url}/v1/systemone",
            json={"state": state, "model": self.model, "questions": questions},
            timeout=self.timeout,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        answer = resp.json()["answers"]["category"]
        raw_category = answer["choice"]
        confidence = answer["confidence"]
        # confidence = (p_max - 1/K) / (1 - 1/K): 0 at uniform (random), 1 at certainty.
        # Below the threshold the signal is too weak to trust; fall back to "none-weak".
        category = raw_category if confidence >= CONFIDENCE_THRESHOLD else "none-weak"
        return {
            "category": category,
            "raw_category": raw_category,
            "confidence": confidence,
            "probabilities": answer["probabilities"],
            "latency_ms": latency_ms,
        }

    def classify_frame(self, frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
        results = []
        t0 = time.perf_counter()
        n_err = 0

        for i, (_, row) in enumerate(frame.iterrows()):
            text = str(row[text_column])

            try:
                result = self.classify(text)
            except requests.RequestException as e:
                results.append({"category": None, "confidence": None, "probabilities": {}})
                n_err += 1
                continue

            results.append({
                "category": result["category"],
                "raw_category": result["raw_category"],
                "confidence": round(result["confidence"], 4),
                "latency_ms": round(result["latency_ms"], 1),
                "probabilities": {k: round(v, 4) for k, v in result["probabilities"].items()},
            })

            if (i + 1) % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i+1}/{len(frame)} rows ({elapsed:.1f}s, {elapsed/(i+1)*1000:.0f} ms/row)")

        elapsed = time.perf_counter() - t0
        suffix = f", {n_err} errors" if n_err else ""
        print(f"classified {len(frame):,} rows in {elapsed:.1f}s ({elapsed/max(1, len(frame))*1000:.0f} ms/row{suffix})")

        out = frame.copy()
        out["category"] = [r["category"] for r in results]
        out["raw_category"] = [r["raw_category"] for r in results]
        out["confidence"] = [r["confidence"] for r in results]
        out["latency_ms"] = [r["latency_ms"] for r in results]
        out["probabilities"] = [r["probabilities"] for r in results]
        return out


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def read_input(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_output(frame: pd.DataFrame, path: str) -> None:
    if Path(path).suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def command_classify(args) -> None:
    classifier = GuardrailClassifier(args.url, args.model, args.timeout)
    result = classifier.classify(args.text)
    weak = result["category"] == "none-weak"
    print(f"\ncategory:   {result['category']}", end="")
    if weak:
        print(f"  (model predicted '{result['raw_category']}' but confidence is below threshold)", end="")
    print()
    K = len(result["probabilities"])
    print(
        f"confidence: {result['confidence']:.4f}"
        f"  [0=random across {K} classes, 1=certain; threshold={CONFIDENCE_THRESHOLD}]"
    )
    print(f"latency:    {result['latency_ms']:.1f} ms")
    if weak:
        print(f"  -> below threshold ({result['confidence']:.4f} < {CONFIDENCE_THRESHOLD}): treating as 'none-weak'")
    print("probabilities:")
    for cat, prob in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<16} {prob:.4f}")


def command_batch(args) -> None:
    frame = read_input(args.input)
    if args.limit:
        frame = frame.head(args.limit)
    classifier = GuardrailClassifier(args.url, args.model, args.timeout)
    out = classifier.classify_frame(frame, args.text_column)
    write_output(out, args.output)
    print(f"wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("--url", default=DEFAULT_URL, help="kev.serve base URL")
        sub.add_argument("--model", default="kev-latest")
        sub.add_argument("--timeout", type=float, default=60.0)

    classify = subparsers.add_parser("classify", help="classify a single text")
    add_common(classify)
    classify.add_argument("--text", required=True, help="text to classify")
    classify.set_defaults(func=command_classify)

    batch = subparsers.add_parser("batch", help="classify a CSV or parquet file")
    add_common(batch)
    batch.add_argument("--input", required=True, help="CSV or parquet input")
    batch.add_argument("--output", required=True, help="CSV or parquet output")
    batch.add_argument("--text-column", default="text", help="column containing the text to classify")
    batch.add_argument("--limit", type=int, help="process only the first N rows")
    batch.set_defaults(func=command_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
