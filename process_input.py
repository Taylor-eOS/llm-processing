from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import settings

MODEL_NAME = "trillionlabs/Trillion-7B-preview"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)

def process_line(instruction, text):
    prompt = f"{instruction}: \"{text}\" Rewritten:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        inputs.input_ids,
        max_new_tokens=200,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("Rewritten:")[-1].strip()

if __name__ == "__main__":
    instruction = settings.REQUEST
    with open("input.txt", "r", encoding="utf-8") as fin, open("output.txt", "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                fout.write("\n")
                continue
            result = process_line(instruction, line)
            result = result.replace('`', '')
            fout.write(result + "\n")

