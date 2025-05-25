import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList
import settings

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        "trillionlabs/Trillion-7B-preview",
        torch_dtype=torch.bfloat16,
        device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("trillionlabs/Trillion-7B-preview")
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
    return [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings], stop_strings

def get_gen_kwargs(tokenizer, stop_sequences):
    return {
        "max_new_tokens": 2048,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens(stop_sequences)]),}

class CaptureStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True, stop_str=None):
        super().__init__(tokenizer, skip_prompt=skip_prompt, stop_str=stop_str)
        self.output = []
    def on_text(self, text, **kwargs):
        print(text, end="", flush=True)
        self.output.append(text)

def process_line(line, model, tokenizer, gen_kwargs, stop_strs):
    base = (
        "You are a rewriting model. Output only the rewritten text, "
        "no explanations or prefixes.\n"
        f"{settings.REQUEST}\n\n"
        f"Original: {line}\nRewritten:")
    messages = [{"role": "user", "content": base}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    if "token_type_ids" in enc:
        del enc["token_type_ids"]
    torch.cuda.empty_cache()
    streamer = CaptureStreamer(tokenizer, skip_prompt=True, stop_str=stop_strs)
    model.generate(**enc, streamer=streamer, **gen_kwargs)
    print()

def main():
    model, tokenizer = load_model_and_tokenizer()
    stop_sequences, stop_strs = get_stop_sequences(tokenizer)
    gen_kwargs = get_gen_kwargs(tokenizer, stop_sequences)
    with open("input.txt", "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            process_line(line, model, tokenizer, gen_kwargs, stop_strs)

if __name__ == "__main__":
    main()
