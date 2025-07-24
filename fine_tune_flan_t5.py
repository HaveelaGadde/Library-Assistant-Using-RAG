import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

# Step 1: Load and prepare your dataset
with open("diverse_qa_pairs.json", "r") as f:
    data = json.load(f)

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)

# Step 2: Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 3: Preprocessing function
def preprocess(example):
    input_text = f"Question: {example['question']} Context: {example['context']}"
    target_text = example["answer"]
    tokenized = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenizer(target_text, truncation=True, padding="max_length", max_length=128)["input_ids"]
    return tokenized

# Step 4: Tokenize the dataset
tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./finetuned-flan-t5",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    learning_rate=5e-5,
    save_strategy="no",
    logging_steps=10,
    report_to="none"
)

# Step 6: Trainer setup
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 7: Train!
trainer.train()

# Optional: Save the model
model.save_pretrained("./finetuned-flan-t5")
tokenizer.save_pretrained("./finetuned-flan-t5")
