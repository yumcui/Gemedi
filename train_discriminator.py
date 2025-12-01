"""
Discriminator Training Script - Complete Implementation
Train discriminator using good and bad data to evaluate realism and difficulty of generated data
"""
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import json
import os

# --- 1. Config ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DISCRIMINATOR_NAME = "llama3-phi-discriminator"

# --- 2. Custom Discriminator Model ---
class DiscriminatorModel(nn.Module):
    """Add classification head on top of Llama base model"""
    def __init__(self, base_model_name, num_labels=1, quantization_config=None):
        super().__init__()
        # Load base model
        load_kwargs = {
            "dtype": torch.float16,  # Use dtype instead of torch_dtype
            "device_map": "auto"
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        
        self.base_model = AutoModel.from_pretrained(
            base_model_name,
            **load_kwargs
        )
        
        # Disable base_model cache (compatible with gradient checkpointing)
        if hasattr(self.base_model, 'config'):
            self.base_model.config.use_cache = False
        
        # Add classification head (regression: output realism score)
        # Use smaller classification head to save memory
        hidden_size = self.base_model.config.hidden_size
        
        # Get base_model dtype (for matching data types)
        model_dtype = next(self.base_model.parameters()).dtype
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),  # Reduce intermediate layer size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, num_labels)
        )
        
        # Convert classifier to same dtype as base_model
        self.classifier = self.classifier.to(dtype=model_dtype)
        
        # Ensure classifier is trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        # Handle all parameters that Trainer may pass
        # Get model output
        if inputs_embeds is not None:
            # If inputs_embeds is provided, use it
            outputs = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        else:
            # Otherwise use input_ids
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Use last token's hidden state (or mean pooling)
        last_hidden_state = outputs.last_hidden_state
        
        # Use attention mask for mean pooling
        if attention_mask is not None:
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            # Avoid division by zero
            sum_mask = attention_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = (last_hidden_state * attention_mask_expanded).sum(1) / sum_mask
        else:
            pooled_output = last_hidden_state.mean(1)
        
        # Ensure classifier is on correct device and dtype
        classifier_device = next(self.classifier.parameters()).device
        classifier_dtype = next(self.classifier.parameters()).dtype
        
        if classifier_device != pooled_output.device or classifier_dtype != pooled_output.dtype:
            self.classifier = self.classifier.to(device=pooled_output.device, dtype=pooled_output.dtype)
        
        # Classification (classifier is trainable, so gradients should flow here)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Ensure logits and labels shapes match
            # logits shape: [batch_size, 1] or [batch_size]
            # labels shape: [batch_size] or scalar
            logits_flat = logits.view(-1)  # Flatten to 1D
            labels_flat = labels.view(-1).float()  # Flatten and convert to float
            
            # Ensure shapes match
            min_len = min(logits_flat.shape[0], labels_flat.shape[0])
            if min_len > 0:
                logits_flat = logits_flat[:min_len]
                labels_flat = labels_flat[:min_len]
            
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits_flat, labels_flat)
        
        # Return format must match Trainer's expectations
        # Trainer expects an object with loss and logits attributes
        from transformers.modeling_outputs import SequenceClassifierOutput
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

# --- 3. Data Preparation and Formatting ---
def format_discriminator_data(examples):
    """Format data"""
    texts = examples["text"]
    labels = examples["label"]
    
    # Tokenize
    tokenizer = format_discriminator_data.tokenizer
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt"
    )
    
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.float32)
    }

# --- 4. Training Function ---
def train_discriminator(
    train_data_path="discriminator_train.jsonl",
    output_dir="./results_discriminator",
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-4
):
    """Train discriminator"""
    
    print("=" * 50)
    print("Training Discriminator")
    print("=" * 50)
    
    # Load data
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": train_data_path})
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Format data
    print("Formatting dataset...")
    print(f"Original dataset columns: {dataset['train'].column_names}")
    
    def tokenize_function(examples):
        # Tokenize text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,  # Reduce max_length to save memory
        )
        # Add labels (convert from label column)
        if "label" in examples:
            tokenized["labels"] = examples["label"]
        elif "labels" in examples:
            # If already labels, use directly
            tokenized["labels"] = examples["labels"]
        else:
            raise ValueError("Neither 'label' nor 'labels' column found in dataset!")
        return tokenized
    
    # Get columns to remove (remove all original columns since tokenize has processed them)
    columns_to_remove = dataset["train"].column_names
    print(f"Removing columns: {columns_to_remove}")
    
    # Tokenize and remove original columns
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove
    )
    
    # Validate final dataset columns
    print(f"Dataset columns after processing: {dataset['train'].column_names}")
    if "labels" not in dataset["train"].column_names:
        raise ValueError("Labels column not found after tokenization!")
    if "input_ids" not in dataset["train"].column_names:
        raise ValueError("input_ids column not found after tokenization!")
    if "attention_mask" not in dataset["train"].column_names:
        raise ValueError("attention_mask column not found after tokenization!")
    
    # Load model (optimized memory configuration)
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Use double quantization to further save memory
    )
    
    # Create Discriminator model
    print("Creating discriminator model...")
    # Llama model doesn't support AutoModelForSequenceClassification, use custom model
    # Note: DiscriminatorModel will load base model internally
    model = DiscriminatorModel(MODEL_NAME, num_labels=1, quantization_config=bnb_config)
    
    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Move classifier to GPU (if base_model is on GPU)
    # Since base_model uses device_map="auto", we need to find which device it's on
    # Apply LoRA first, then move classifier
    
    # Apply LoRA (reduce rank to save memory)
    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=32,  # Reduce rank from 64 to 32
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Reduce target modules
    )
    
    model = get_peft_model(model, peft_config)
    
    # Ensure classifier is on correct device and dtype (same as base_model)
    # Find base_model's device and dtype
    base_model_device = None
    base_model_dtype = None
    for param in model.base_model.parameters():
        if param.device.type == 'cuda':
            base_model_device = param.device
            base_model_dtype = param.dtype
            break
    
    if base_model_device is not None and base_model_dtype is not None:
        # Move classifier to same device and dtype
        model.classifier = model.classifier.to(device=base_model_device, dtype=base_model_dtype)
        print(f"Moved classifier to device: {base_model_device}, dtype: {base_model_dtype}")
    elif base_model_dtype is not None:
        # At least set dtype
        model.classifier = model.classifier.to(dtype=base_model_dtype)
        print(f"Set classifier dtype to: {base_model_dtype}")
    
    # Ensure model is in training mode
    model.train()
    
    # Validate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Validate if classifier is trainable
    classifier_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    print(f"Classifier trainable parameters: {classifier_trainable:,}")
    
    # If classifier is not trainable, force set
    if classifier_trainable == 0:
        print("Warning: Classifier parameters are not trainable! Fixing...")
        for param in model.classifier.parameters():
            param.requires_grad = True
        classifier_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
        print(f"Classifier trainable parameters after fix: {classifier_trainable:,}")
    
    # Check if base_model has trainable parameters (via LoRA)
    # Note: When using PEFT, trainable parameters are in model.peft_config
    if hasattr(model, 'base_model'):
        # Check LoRA layers
        lora_trainable = 0
        for name, module in model.base_model.named_modules():
            if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                lora_trainable += sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"LoRA trainable parameters: {lora_trainable:,}")
    
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    
    # Test if forward pass produces gradients
    print("\nTesting forward pass with gradients...")
    try:
        # Find model's device and dtype
        device = None
        dtype = None
        for param in model.parameters():
            if param.device.type == 'cuda':
                device = param.device
                dtype = param.dtype
                break
        if device is None:
            device = torch.device('cpu')
        if dtype is None:
            dtype = torch.float32
        
        # Use tokenizer to get correct vocab size
        test_input_ids = torch.randint(0, min(1000, len(tokenizer)), (1, 10)).to(device)
        test_labels = torch.tensor([0.5], dtype=dtype).to(device)
        
        test_output = model(input_ids=test_input_ids, labels=test_labels)
        if test_output.loss is not None and test_output.loss.requires_grad:
            print("✓ Forward pass produces gradients correctly")
            print(f"  Loss value: {test_output.loss.item():.4f}")
            print(f"  Loss requires_grad: {test_output.loss.requires_grad}")
        else:
            print("✗ Warning: Forward pass does not produce gradients!")
            print(f"  Loss requires_grad: {test_output.loss.requires_grad if test_output.loss is not None else 'None'}")
    except Exception as e:
        print(f"✗ Error testing forward pass: {e}")
        import traceback
        traceback.print_exc()
        # Don't stop training due to test failure
        print("Continuing with training despite test error...")
    
    # Training arguments (optimized for memory usage)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,  # Reduce batch size
        gradient_accumulation_steps=8,  # Increase gradient accumulation to compensate
        optim="paged_adamw_32bit",
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="no",  # Temporarily disable evaluation
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        gradient_checkpointing=False,  # Temporarily disable gradient checkpointing (may cause gradient issues)
        dataloader_pin_memory=False,  # Disable pin memory to save memory
        remove_unused_columns=False,  # Keep all columns
    )
    
    # Data collator
    # Since we've already done padding, use simple collator here
    from transformers import default_data_collator
    data_collator = default_data_collator
    
    # Clear GPU cache
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    trainer.train()
    
    # Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Save model
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")
    return model, tokenizer

if __name__ == "__main__":
    import sys
    
    # Check data file
    train_data_path = "discriminator_train.jsonl"
    if not os.path.exists(train_data_path):
        print(f"Error: {train_data_path} not found!")
        print("Please run prepare_discriminator_data.py first.")
        sys.exit(1)
    
    # Train
    model, tokenizer = train_discriminator(
        train_data_path=train_data_path,
        output_dir=DISCRIMINATOR_NAME,
        num_epochs=3
    )
