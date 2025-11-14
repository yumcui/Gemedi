from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "/users/zzhou190/1470/models/qwen2.5-0.5b-instruct"
ADAPTER    = "/users/zzhou190/projects/qwen_sft/output_brown_2025_lora/lora_adapter"

question = "2025 年 Brown University 本科 Class of 2029 的整体录取率是多少？请给出百分比并说明大致申请人数和录取人数。"

def build_prompt(q):
    return f"用户：{q}\n助手："

def gen(model, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            num_beams=1,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

print(">> loading base")
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
m_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16
).to("cuda").eval()

print(">> loading base+LoRA")
m_lora = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16
).to("cuda")
m_lora = PeftModel.from_pretrained(m_lora, ADAPTER).eval()

print("\nQ:", question)
print("\n[Base]:")
print(gen(m_base, tok))

print("\n[LoRA]:")
print(gen(m_lora, tok))

