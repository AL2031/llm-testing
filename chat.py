import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = "AL20313/smol-128m-gk-math-instruct-cooldown"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)

def chat(msg, max_new_tokens=200):
    prompt = f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(prompt, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=1.1,
                         eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"))
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

while True:
    print(chat(input("\n> ")))