# Guardrail Detection

Content safety classifier built on [kev-4b](https://huggingface.co/jaredpalmer/kev-4b). Classifies text into safety categories by talking to a running `kev.serve` instance over HTTP.

## Categories

| Category | Description |
|---|---|
| `none` | Normal text, no harmful content |
| `abuse` | Explicit insults, slurs, or demeaning language |
| `adult_content` | Sexually explicit or pornographic material |
| `harassment` | Direct threats or targeted personal attacks |
| `hate_speech` | Attacks on groups based on protected attributes |
| `violence` | Text glorifying or inciting physical harm |
| `spam` | Unsolicited promotions, phishing, or deceptive solicitations |

Responses with confidence below `0.5` are returned as `none-weak`.

## Setup

> **Run `setup.sh` only once** — it clones the `kev` repo and downloads the model weights.

```bash
bash setup.sh
```

## Starting the server

After setup, start the kev server with:

```bash
bash run_kev.sh
```

The server listens on `http://127.0.0.1:8008` by default.

## Usage

### Classify a single text (CLI)

```bash
python guardrail_detection_kev.py classify --text "some text to check"
```

### Batch classify a CSV or Parquet file

```bash
python guardrail_detection_kev.py batch \
    --input texts.csv --text-column text \
    --output results.parquet
```

### Gradio UI

```bash
python gradio_demo.py
```

Opens at `http://localhost:7860`. On SageMaker Studio it is accessible via the built-in port proxy.

## Files

| File | Description |
|---|---|
| `setup.sh` | One-time setup: clones kev repo and starts initial model download |
| `run_kev.sh` | Starts the kev.serve HTTP server |
| `guardrail_detection_kev.py` | CLI tool for single and batch classification |
| `gradio_demo.py` | Gradio web UI for interactive classification |
