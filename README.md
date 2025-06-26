# Self-Healing Classification DAG with a Fine-Tuned Transformer

**ATG Technical Assignment**

This project implements a robust, self-healing text classification pipeline for sentiment analysis. It uses a `distilbert-base-uncased` model fine-tuned with LoRA on the SST-2 dataset. The entire workflow is orchestrated by LangGraph and includes a backup zero-shot model and live session statistics as bonus features.

The system intelligently decides whether to trust its primary model, consult a backup model, or ask a human for clarification based on prediction confidence, ensuring high reliability in human-in-the-loop workflows.

## System Architecture

The core of the system is a Directed Acyclic Graph (DAG) that prioritizes reliability. The workflow is as follows:

```
+----------------+      +------------------+      +-----------------------+
|   User Input   |----->|  InferenceNode   |----->|  ConfidenceCheck      |
+----------------+      | (Primary Model)  |      |    (Edge Logic)       |
                        +------------------+      +-----------+-----------+
                                                          |
             +--------------------------------------------+---------------------------------------------+
             | (Confidence >= 99%)                                                                      | (Confidence < 99%)
             v                                                                                          v
+------------------------+                                                                  +------------------------+
|    Final Output &      |                                                                  |      FallbackNode      |
| Session Statistics     |                                                                  | (+ Backup Zero-Shot)   |
+------------------------+                                                                  +-----------+------------+
                                                                                                        | (Engage Human)
                                                                                                        v
                                                                                            +------------------------+
                                                                                            |    Final Output &      |
                                                                                            | Session Statistics     |
                                                                                            +------------------------+
```

1.  **Inference Node**: Classifies input using the fine-tuned LoRA model.
2.  **Confidence Check (Conditional Edge)**: If confidence is high (`>= 99%`), the prediction is accepted. If low, it triggers the self-healing path.
3.  **Fallback Node**: This node activates two **bonus features**:
    *   **Backup Model**: It gets a "second opinion" from a `bart-large-mnli` zero-shot classifier to provide more context.
    *   **Human-in-the-Loop**: It presents both predictions to the user and asks for the definitive classification.
4.  **Session Statistics**: After every prediction, the CLI displays an updated dashboard showing total predictions, average model confidence, and the fallback frequency rate, including a CLI histogram.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AlphaPriyan08/self-healing-classifier
    cd self-healing-classifier
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The first run of `main.py` will take longer as it downloads the 1.6GB backup zero-shot model.)*

## How to Run

### Step 1: Fine-Tune the Model

This script downloads the `sst2` dataset and fine-tunes the DistilBERT model using LoRA. The resulting model adapter will be saved to the `./distilbert-sst2-lora/` directory.

```bash
python fine_tune.py
```

### Step 2: Launch the Self-Healing DAG

This script loads the fine-tuned model and starts the interactive command-line interface with all features.

```bash
python main.py
```

## Sample CLI Interaction

Here is a sample of a full interaction with the application, demonstrating both the high-confidence path and the self-healing fallback path.

```
Loading model from ./distilbert-sst2-lora/...
Model loaded successfully.
Loading backup zero-shot model...
Backup model loaded.
──────────────── Self-Healing Sentiment Classifier ─────────────────
Confidence Threshold: 99%
Enter text to classify or type 'quit' to exit.

Enter a sentence: I love this movie
╭─────────────────────── Inference Result ─────────────────────────╮
│ [InferenceNode] Predicted label: 'POSITIVE' | Confidence: 99.95% │
╰──────────────────────────────────────────────────────────────────╯
╭──────────────── Final Classification ────────────────╮
│ Input: 'I love this movie'                           │
│         Final Label: POSITIVE                        │
│         Correction Required: No                      │
╰──────────────────────────────────────────────────────╯
       Session Statistics
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric              ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Total Predictions   │ 1      │
│ Average Confidence  │ 99.95% │
│ Fallbacks Triggered │ 0      │
│ Fallback Rate       │ 0.0%   │
│ Fallback Frequency  │ [ ]    │
└─────────────────────┴────────┘

Enter a sentence: The film was okay I guess
╭────────────────────── Inference Result ──────────────────────────╮
│ [InferenceNode] Predicted label: 'POSITIVE' | Confidence: 91.47% │
╰──────────────────────────────────────────────────────────────────╯
╭─────────────────────── Fallback Activated ─────────────────────────╮
│ [FallbackNode] Confidence is low. Engaging user for clarification. │
╰────────────────────────────────────────────────────────────────────╯
Getting a second opinion from a zero-shot model...
╭─────────────── Backup Model Result (Zero-Shot) ──────────╮
│ [BackupModel] Predicted: 'POSITIVE' | Confidence: 76.52% │
╰──────────────────────────────────────────────────────────╯
The primary model predicted 'POSITIVE', and the backup predicted 'POSITIVE'.
How would you classify this? (Or type 'y' to accept 'POSITIVE') [y/positive/negative] (y): negative
[FallbackNode] User corrected prediction to: 'NEGATIVE'
╭──────────────── Final Classification ────────────────╮
│ Input: 'The film was okay I guess'                   │
│         Final Label: NEGATIVE                        │
│         Correction Required: Yes                     │
╰──────────────────────────────────────────────────────╯
       Session Statistics
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric              ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Total Predictions   │ 2      │
│ Average Confidence  │ 95.71% │
│ Fallbacks Triggered │ 1      │
│ Fallback Rate       │ 50.0%  │
│ Fallback Frequency  │ [█ ]   │
└─────────────────────┴────────┘

Enter a sentence: quit
```

## Demo Video

[View the Demo on Google Drive](https://drive.google.com/file/d/11Utos_eanTtRJABIUV6vJRbRBZL7L7jk/view?usp=sharing)
