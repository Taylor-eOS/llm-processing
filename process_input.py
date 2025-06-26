import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import settings

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(settings.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
    return model, tokenizer

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_sequences):
        self.stop_sequences = stop_sequences
    def __call__(self, input_ids, scores, **kwargs):
        for seq in self.stop_sequences:
            if input_ids[0, -len(seq):].tolist() == seq:
                return True
        return False

def get_stop_sequences(tokenizer):
    stop_strings = ["\nYou:", "\nUser:", tokenizer.eos_token]
    return [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]

def get_gen_kwargs(tokenizer, stop_sequences):
    return {
        "max_new_tokens": settings.MAX_TOKENS,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens(stop_sequences)]),}

def process_line(line, model, tokenizer, gen_kwargs, memory=None, use_memory=False):
    base = "You are a text processing model. Output only the processed text itself, no other explanations or comments.\n"
    if use_memory and memory:
        prev_original, prev_rewrite = memory
        base += f"Context: Previous input: \"{prev_original}\", Previous output: \"{prev_rewrite}\"\n"
    base += f"Task: {settings.REQUEST}\nInput sentence: \"{line}\"\nProcessed:"
    if settings.PRINT: print(base)
    messages = [{"role": "user", "content": base}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    if "token_type_ids" in enc:
        del enc["token_type_ids"]
    torch.cuda.empty_cache()
    output_ids = model.generate(**enc, **gen_kwargs)
    output_text = tokenizer.decode(output_ids[0][enc['input_ids'].shape[-1]:], skip_special_tokens=True)
    return output_text.strip()

def main():
    model, tokenizer = load_model_and_tokenizer()
    stop_sequences = get_stop_sequences(tokenizer)
    gen_kwargs = get_gen_kwargs(tokenizer, stop_sequences)
    use_memory = settings.MEMORY
    memory = None
    with open("input.txt", "r", encoding="utf-8") as infile, open("output.txt", "w", encoding="utf-8") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            output = process_line(line, model, tokenizer, gen_kwargs, memory, use_memory)
            print(output)
            outfile.write(output + "\n")
            if use_memory:
                memory = (line, output)

if __name__ == "__main__":
    main()

