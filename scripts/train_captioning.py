from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer
from datasets import load_dataset

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dataset = load_dataset("imagefolder", data_dir="data/processed_images")

def process_data(batch):
    inputs = feature_extractor(images=batch["image"], return_tensors="pt")
    outputs = tokenizer(batch["caption"], padding="max_length", truncation=True, return_tensors="pt")
    return inputs, outputs

dataset = dataset.map(process_data, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
