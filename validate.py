import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# --- 1. 定义 ---
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# (关键！) 这指向你用 50 步训练覆盖的“更强”权重
adapter_path = "./my-first-lora-weights" 

# --- 2. 加载配置 ---
print("--- 正在加载 4-bit 量化配置... ---")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# --- 3. 加载基础模型和 Tokenizer ---
print(f"--- 正在加载基础模型 ({base_model_name})... ---")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- 4. 准备提示 ---
# (我们用一个稍有创意的问题，更容易看出风格变化)
prompt = "<|user|>\nWrite a 4-line poem about a robot learning to code.\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\n" + "="*40)
print("  TEST A: 基础模型 (最初的 Llama) 回复：")
print("="*40 + "\n")

# --- 5. (A) 运行“基础模型”---
base_model.config.use_cache = True # 确保它能正常生成
outputs_base = base_model.generate(
    **inputs, 
    max_new_tokens=50, 
    do_sample=True, 
    temperature=0.7, # 我们使用固定的 temperature
    top_k=50,
    top_p=0.95
)
print(tokenizer.decode(outputs_base[0], skip_special_tokens=True))


# --- 6. (关键！) 加载你的 LoRA 权重 ---
print("\n" + "="*40)
print(f"--- 正在加载你的 LoRA 权重 (从 {adapter_path})... ---")

# (这会修改 base_model，把它变成“微调后”的模型)
peft_model = PeftModel.from_pretrained(base_model, adapter_path)
peft_model.config.use_cache = True

print("--- 权重加载成功！ ---")
print("\n" + "="*40)
print("  TEST B: 你的微调模型 (Llama + 50步) 回复：")
print("="*40 + "\n")

# --- 7. (B) 运行“微调模型”---
outputs_peft = peft_model.generate(
    **inputs, 
    max_new_tokens=50, 
    do_sample=True, 
    temperature=0.7, # (使用完全相同的参数)
    top_k=50,
    top_p=0.95
)
print(tokenizer.decode(outputs_peft[0], skip_special_tokens=True))
print("\n" + "="*40)
print("--- 对比完成！ ---")
