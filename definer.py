import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import settings
import sys

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(settings.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME, trust_remote_code=True)
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
        "temperature": 0.3,
        "top_p": 0.95,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens(stop_sequences)]),}

def get_definition(word, model, tokenizer, gen_kwargs):
    base = settings.BASE
    base += f"Task: {settings.REQUEST}\nInput word: \"{word.strip()}\"\nDefinition:"
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
    print("Word Definer Tool - type a word or short phrase, press Enter for definition. Empty line to quit.\n")
    model, tokenizer = load_model_and_tokenizer()
    stop_sequences = get_stop_sequences(tokenizer)
    gen_kwargs = get_gen_kwargs(tokenizer, stop_sequences)
    
    try:
        while True:
            sys.stdout.write("Word: ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                break
            word = line.strip()
            if word == "":
                print("Goodbye!")
                break
            definition = get_definition(word, model, tokenizer, gen_kwargs)
            print(f"Definition: {definition}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
