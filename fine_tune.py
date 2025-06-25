# fine_tune.py
import os
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, TaskType

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "sst2"
OUTPUT_DIR = "distilbert-sst2-lora"
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 16

def main():
    """Main function to fine-tune the model."""
    print("--- Starting Fine-Tuning ---")

    # --- 1. Load Dataset & Tokenizer ---
    print(f"Loading dataset '{DATASET_NAME}' and tokenizer '{MODEL_NAME}'...")
    dataset = load_dataset(DATASET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets.set_format("torch")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 2. Load Base Model & Add LoRA Adapter ---
    print("Loading base model and configuring LoRA...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )
    
    # Freeze all base model parameters
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_lin", "v_lin"],
    )

    peft_model = get_peft_model(model, lora_config)
    print("\n--- PEFT Model Architecture ---")
    peft_model.print_trainable_parameters()
    print("-" * 30)

    # --- 3. Define Metrics & Training Arguments ---
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        push_to_hub=False,
    )

    # --- 4. Create and Run Trainer ---
    print("\n--- Starting Training ---")
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # --- 5. Save the fine-tuned adapter ---
    print(f"\nTraining complete. Saving best model adapter to '{OUTPUT_DIR}'")
    peft_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n--- Fine-Tuning Finished Successfully ---")

if __name__ == "__main__":
    main()