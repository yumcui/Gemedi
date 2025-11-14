import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer # <-- 我们仍然使用 SFTTrainer

# --- 1. 定义你的模型 (Qwen) ---
model_name = "Qwen/Qwen2.5-7B-Instruct"

# --- 2. (不变) 加载数据集 ---
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split="train") # (我们用全部 1000 条)

# --- 3. (不变) 4-bit 量化配置 ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# --- 4. (关键更新) 加载模型和 Tokenizer ---
print("--- (1/4) 正在加载模型... ---")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True # (Qwen 需要这个)
)
# (我们不再需要 model.config.use_cache = False 了)

print("--- (2/4) 正在加载 Tokenizer... ---")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # (这仍然很重要)

# --- 5. (关键更新) LoRA 配置 (不变) ---
# (这个 LoRAConfig 仍然是正确的)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16, # (我们之前用的 16)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

# --- 6. (不变) 训练参数 ---
training_args = TrainingArguments(
    output_dir="./lora-results", # 临时检查点
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=50,       # (我们之前用的 50 步)
    fp16=True,
    optim="paged_adamw_8bit" # (使用 8-bit 优化器节省内存)
)

# --- 7. (!! 关键 API 变更 !!) SFTTrainer ---
# 这是“新”的 API。它更简单了！
print("--- (3/4) 正在初始化 SFTTrainer... ---")
trainer = SFTTrainer(
    model=model,                  # (传入基础模型)
    train_dataset=dataset,
    peft_config=peft_config,      # (!! 新 !!) 直接把 LoRA 配置传给它
    dataset_text_field="text",    # (!! 新 !!) 告诉它哪一列是文本
    max_seq_length=512,           # (!! 新 !!) 设置最大长度
    tokenizer=tokenizer,
    args=training_args,
)

# --- 8. (不变) 开始训练 ---
print("--- (4/4) 开始训练... ---")
trainer.train()

# --- 9. (不变) 保存你的 LoRA 权重 ---
print("--- 训练完成！正在保存最终权重... ---")
# (我们把它保存到你熟悉的老地方)
final_model_path = "./my-first-lora-weights"
trainer.save_model(final_model_path)

print(f"--- 成功！你的 LoRA 权重已保存到: {final_model_path} ---")
