# main.py (with all bonus features and final rich fix)
import logging
from typing import TypedDict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
import numpy as np

from langgraph.graph import StateGraph, END

# --- Constants ---
MODEL_PATH = "./distilbert-sst2-lora/"
CONFIDENCE_THRESHOLD = 0.99 # A strict threshold to trigger fallbacks

# --- Setup Logging and Console ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", mode='w'),
        logging.StreamHandler()
    ]
)
console = Console()

# --- Classifier Class to encapsulate model logic ---
class SentimentClassifier:
    def __init__(self, model_path: str):
        console.print(f"[bold cyan]Loading model from {model_path}...[/bold cyan]")
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        console.print("[bold green]Model loaded successfully.[/bold green]")

    def predict(self, text: str) -> (str, float):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class_id = torch.max(probabilities, dim=1)
        
        label = self.model.config.id2label[predicted_class_id.item()]
        return label, confidence.item()

# --- LangGraph State Definition ---
class GraphState(TypedDict):
    input_text: str
    prediction: str
    confidence: float
    fallback_triggered: bool
    final_label: str
    is_corrected: bool
    log_messages: List[str]

# --- LangGraph Nodes ---
def inference_node(state: GraphState) -> GraphState:
    text = state["input_text"]
    logging.info(f"Running inference on: '{text}'")
    
    prediction, confidence = classifier.predict(text)
    
    log_msg = f"[InferenceNode] Predicted label: '{prediction}' | Confidence: {confidence:.2%}"
    console.print(Panel(log_msg, title="Inference Result", border_style="yellow"))
    
    return {**state, "prediction": prediction, "confidence": confidence, "log_messages": state.get("log_messages", []) + [log_msg]}

def fallback_node(state: GraphState) -> GraphState:
    log_msg = "[FallbackNode] Confidence is low. Engaging user for clarification."
    console.print(Panel(log_msg, title="Fallback Activated", border_style="red"))
    logging.warning(log_msg)

    console.print("[cyan]Getting a second opinion from a zero-shot model...[/cyan]")
    backup_result = backup_classifier(state["input_text"], candidate_labels=["POSITIVE", "NEGATIVE"])
    backup_label = backup_result['labels'][0]
    backup_score = backup_result['scores'][0]
    
    backup_log_msg = f"[BackupModel] Predicted: '{backup_label}' | Confidence: {backup_score:.2%}"
    console.print(Panel(backup_log_msg, title="Backup Model Result (Zero-Shot)", border_style="blue"))
    logging.info(backup_log_msg)
    
    clarification = Prompt.ask(
        f"The primary model predicted '{state['prediction']}', and the backup predicted '{backup_label}'.\nHow would you classify this? (Or type 'y' to accept '{state['prediction']}')",
        choices=["y", "positive", "negative"],
        default="y"
    ).lower()

    if clarification == 'y':
        final_label = state['prediction']
        is_corrected = False
        log_msg_2 = f"[FallbackNode] User confirmed original prediction: '{final_label}'"
    else:
        is_corrected = True
        final_label = clarification.upper()
        log_msg_2 = f"[FallbackNode] User corrected prediction to: '{final_label}'"
        
    console.print(f"[bold magenta]{log_msg_2}[/bold magenta]")
    logging.info(log_msg_2)
    
    return {**state, "final_label": final_label, "is_corrected": is_corrected, "fallback_triggered": True, "log_messages": state.get("log_messages", []) + [log_msg, backup_log_msg, log_msg_2]}

# --- LangGraph Conditional Edge ---
def confidence_check(state: GraphState) -> str:
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        logging.info(f"Confidence {state['confidence']:.2f} is below threshold {CONFIDENCE_THRESHOLD}. Routing to fallback.")
        return "trigger_fallback"
    else:
        logging.info(f"Confidence {state['confidence']:.2f} is sufficient. Routing to end.")
        return "accept_prediction"
        
def finalize_output(state: GraphState) -> GraphState:
    if not state.get("final_label"):
        return {**state, "final_label": state["prediction"], "is_corrected": False, "fallback_triggered": False}
    return state

# --- Statistics Display Function ---
def display_statistics(confidences: List[float], fallback_count: int):
    if not confidences: return
    table = Table(title="Session Statistics")
    table.add_column("Metric", style="cyan"); table.add_column("Value", style="magenta")
    total_predictions = len(confidences)
    avg_confidence = np.mean(confidences)
    fallback_rate = (fallback_count / total_predictions) * 100
    table.add_row("Total Predictions", str(total_predictions))
    table.add_row("Average Confidence", f"{avg_confidence:.2%}")
    table.add_row("Fallbacks Triggered", str(fallback_count))
    table.add_row("Fallback Rate", f"{fallback_rate:.1f}%")
    bar = "█" * fallback_count + " " * (total_predictions - fallback_count)
    table.add_row("Fallback Frequency", f"[{bar}]")
    console.print(table)

# --- Main Application ---
if __name__ == "__main__":
    classifier = SentimentClassifier(MODEL_PATH)
    console.print("[bold cyan]Loading backup zero-shot model...[/bold cyan]")
    backup_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    console.print("[bold green]Backup model loaded.[/bold green]")

    workflow = StateGraph(GraphState)
    workflow.add_node("inference", inference_node)
    workflow.add_node("fallback", fallback_node)
    workflow.set_entry_point("inference")
    workflow.add_conditional_edges("inference", confidence_check, {"trigger_fallback": "fallback", "accept_prediction": END})
    workflow.add_edge("fallback", END)
    app = workflow.compile()

    session_confidences = []; session_fallback_count = 0
    console.rule("[bold green]Self-Healing Sentiment Classifier[/bold green]")
    console.print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}")
    console.print("Enter text to classify or type 'quit' to exit.")
    
    while True:
        user_input = Prompt.ask("\n[bold]Enter a sentence[/bold]")
        if user_input.lower() == 'quit': break
        initial_state = {"input_text": user_input, "log_messages": []}
        
        final_state = {}
        for event in app.stream(initial_state):
            final_state = list(event.values())[0]

        final_state = finalize_output(final_state)
        
        session_confidences.append(final_state['confidence'])
        if final_state['fallback_triggered']:
            session_fallback_count += 1
            
        # FIXED: Proper Rich markup formatting
        is_corrected = final_state.get('is_corrected', False)
        correction_status = "Yes" if is_corrected else "No"
        
        # Create the panel content without nested markup issues
        panel_content = f"""Input: '{final_state['input_text']}'
        Final Label: {final_state['final_label']}
        Correction Required: {correction_status}"""
        
        # Apply styling to the entire panel based on correction status
        panel_style = "magenta" if is_corrected else "green"
        
        result_panel = Panel(
            panel_content,
            title="Final Classification",
            border_style=panel_style
        )
        console.print(result_panel)
        logging.info(f"Final Decision: Input='{final_state['input_text']}', Label='{final_state['final_label']}', Corrected={final_state.get('is_corrected', False)}")
        display_statistics(session_confidences, session_fallback_count)