from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch, os

BASE_MODEL = "/users/zzhou190/1470/models/qwen2.5-0.5b-instruct"
DATA_PATH  = "/users/zzhou190/projects/qwen_sft/data/brown_2025.jsonl"
OUT_DIR    = "/users/zzhou190/projects/qwen_sft/output_brown_2025_lora"

os.makedirs(OUT_DIR, exist_ok=True)

print(">> loading tokenizer/model ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16
)
model.to("cuda")
model.train()

# ---- LoRA config ----
lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

print(">> loading dataset ...")
ds = load_dataset("json", data_files=DATA_PATH, split="train")
print("  dataset size:", len(ds))
print("  example[0]:", ds[0])

MAX_LEN = 256

def build_example(ex):
    user = ex["instruction"].strip()
    ans  = ex["output"].strip()
    prompt = f"用户：{user}\n助手："
    full = prompt + ans

    enc = tokenizer(
        full,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors=None,
    )
    labels = enc["input_ids"][:]

    # 只在“助手回答”上算 loss：把 prompt 部分的 label 设为 -100
    with tokenizer.as_target_tokenizer():
        pref = tokenizer(prompt, truncation=True, max_length=MAX_LEN)
    pref_len = len(pref["input_ids"])
    labels[:pref_len] = [-100] * pref_len

    enc["labels"] = labels
    return enc

print(">> tokenizing ...")
tokenized = ds.map(build_example, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=30,                # 小数据，直接 overfit
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-3,                 # 稍大，让它快点记住
    logging_steps=1,
    save_strategy="no",
    fp16=True,
    optim="adamw_torch",
    report_to="none",
    dataloader_pin_memory=False,
)

print(">> training ...")
trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()

print(">> saving adapter ...")
model.save_pretrained(os.path.join(OUT_DIR, "lora_adapter"))
tokenizer.save_pretrained(os.path.join(OUT_DIR, "lora_adapter"))
print("✅ done, saved to", os.path.join(OUT_DIR, "lora_adapter"))

