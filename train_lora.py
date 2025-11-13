import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model  # <--- (旧 API 需要)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# --- 1. 定义模型和数据集 ---
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split="train") # 只取 1% (约10条)

# --- 2. 加载 Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. 配置 4-bit 量化 (bitsandbytes 会用到) ---
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# --- 4. 加载基础模型 ---
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
)
model.config.use_cache = False
model.config.pretraining_tp = 1 # 针对 TinyLlama 的设置

# --- 5. LoRA 配置 ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 6. (关键！) 手动将 LoRA "便利贴" 应用到模型上 ---
# (这是 trl 0.7.4 必需的)
model_with_lora = get_peft_model(model, peft_config)

# --- 7. 训练参数 ---
# (注意：这里没有 max_seq_length)
training_args = TrainingArguments(
    output_dir="./lora-results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    max_steps=50,       # 只训练 5 步
    logging_steps=1,
)

# --- 8. (旧版API) 初始化 SFTTrainer ---
# (注意：它 *不* 认识 peft_config, 但 *需要* 以下参数)
trainer = SFTTrainer(
    model=model_with_lora,        # <--- 传入已应用 LoRA 的模型
    train_dataset=dataset,
    dataset_text_field="text",    # <--- 必须存在
    max_seq_length=512,           # <--- 必须存在
    tokenizer=tokenizer,
    args=training_args,
)

# --- 9. 开始训练 ---
print("--- 开始 LoRA 训练 ---")
trainer.train()
print("--- 训练完成 ---")

# --- 10. 保存模型 ---
# (这只会保存 LoRA "便利贴", 而不是整个模型)
output_save_dir = "./my-first-lora-weights"
trainer.save_model(output_save_dir)

print(f"--- 成功！你的 LoRA 权重已保存到: {output_save_dir} ---")
