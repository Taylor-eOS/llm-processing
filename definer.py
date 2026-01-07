import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextStreamer
import settings
import sys

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(settings.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
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
    stop_strings = ["\nYou:", "\nUser:", "\nAssistant:", "\n</s>", "[|endofturn|]"]
    return [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings if tokenizer.encode(s, add_special_tokens=False)]

def get_gen_kwargs(tokenizer, stop_sequences):
    return {
        "max_new_tokens": settings.MAX_TOKENS,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
    }

class CleanStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False):
        super().__init__(tokenizer, skip_prompt=skip_prompt)

    def on_finalized_text(self, text, stream_end=False):
        cleaned = text.replace("[|endofturn|]", "")
        print(cleaned, end="", flush=True)
        if stream_end:
            print()

def get_definition(word, model, tokenizer, gen_kwargs):
    prompt_text = f"Provide a clear and concise definition for the word or phrase: \"{word.strip()}\"\nDefinition:"
    enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    if "token_type_ids" in enc:
        del enc["token_type_ids"]
    torch.cuda.empty_cache()
    stop_sequences = get_stop_sequences(tokenizer)
    stop_criteria = StopOnTokens(stop_sequences)
    streamer = CleanStreamer(tokenizer, skip_prompt=True)
    generation_kwargs = gen_kwargs.copy()
    generation_kwargs["streamer"] = streamer
    generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_criteria])
    model.generate(**enc, **generation_kwargs)

def main():
    print("Word Definer Tool\nType a word or short phrase, press Enter for definition. Empty line to quit.\n")
    model, tokenizer = load_model_and_tokenizer()
    
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
            print("Definition:", end="", flush=True)
            gen_kwargs = get_gen_kwargs(tokenizer, None)
            get_definition(word, model, tokenizer, gen_kwargs)
            print()
    except KeyboardInterrupt:
        print("\nGoodbye!")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
