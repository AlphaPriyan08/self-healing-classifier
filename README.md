# Self-Healing Classification DAG with a Fine-Tuned Transformer

**ATG Technical Assignment**

This project implements a robust, self-healing text classification pipeline for sentiment analysis. It uses a `distilbert-base-uncased` model fine-tuned with LoRA on the SST-2 dataset. The entire workflow is orchestrated by LangGraph and includes a backup zero-shot model and live session statistics as bonus features.

The system intelligently decides whether to trust its primary model, consult a backup model, or ask a human for clarification based on prediction confidence, ensuring high reliability in human-in-the-loop workflows.

## System Architecture

The core of the system is a Directed Acyclic Graph (DAG) that prioritizes reliability. The workflow is as follows:

```
+----------------+      +------------------+      +-----------------------+         +---------------------+
|   User Input   |----->|  InferenceNode   |----->|  ConfidenceCheck      |-------->|  Final Output &     |
+----------------+      | (Primary Model)  |      |    (Edge Logic)       |         | Session Statistics  |
                        +------------------+      +-----------+-----------+         +---------------------+
                                                          |                 (Confidence >= 99%)
                                       (Confidence < 99%) |
                                                          v
                                              +------------------------+
                                              |      FallbackNode      |
                                              | (+ Backup Zero-Shot)   |
                                              +-----------+------------+
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

### CLI Flow Explained

1.  You will be prompted to **"Enter a sentence"**.
2.  For a clear sentence (e.g., "I love this movie"), the model will be highly confident, and you will see the final result immediately.
3.  For an ambiguous sentence (e.g., "The film was okay I guess"), the model's confidence will be below the threshold, triggering the fallback.
4.  The system will ask you to confirm or correct the prediction.
5.  After your input, the final, verified classification is shown.
6.  All interactions are logged in `app.log`.