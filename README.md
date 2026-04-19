# Qwen2.5 Ranking Service

This project implements a ranking service using Qwen2.5-7B-Instruct. It ranks a list of texts based on their adherence to a specific style.

## Setup

1.  **Install `uv`** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

## Usage

### 1. Start the Server
The server loads the model into memory once.

```bash
uv run rank_server.py
```

*Note: This requires a GPU with enough VRAM for a 7B model (approx 14GB for fp16, less for quantized).*

### 2. Run the Test Client
In a separate terminal:

```bash
uv run test_client.py
```

## API Endpoint

**POST** `/rank`

**Request Body:**
```json
{
  "style_name": "Formal Scientific",
  "texts": [
    "Text A...",
    "Text B..."
  ]
}
```

**Response:**
```json
{
  "ranked_indices": [1, 0],
  "ranked_texts": ["Text B...", "Text A..."],
  "raw_output": "[2] > [1]"
}
```
