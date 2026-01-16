import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import pysbd
import settings
import os

input_file = "input_s.txt"
output_file = "output_s.txt"

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
    stop_strings = ["\nYou:", "\nUser:", "\nAssistant:", tokenizer.eos_token]
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

def process_sentence(sentence, model, tokenizer, gen_kwargs, search_string, instruction):
    parts = []
    parts.append(f"Original sentence:\n{sentence}")
    parts.append(f"Search term: {search_string}")
    parts.append(f"Instruction:\n{instruction}")
    parts.append("Rewrite the sentence according to the instruction as a single sentence.")
    parts.append("Output only the rewritten sentence with appropriate punctuation.")
    base = "\n".join(parts)
    if hasattr(settings, "BASE") and settings.BASE:
        base += "\n" + settings.BASE
    base += "\nRewritten:"
    messages = [{"role": "user", "content": base}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    if "token_type_ids" in enc:
        del enc["token_type_ids"]
    torch.cuda.empty_cache()
    output_ids = model.generate(**enc, **gen_kwargs)
    output_text = tokenizer.decode(output_ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return output_text.strip()

def find_matching_sentences(current_sents, search_string):
    lower_search = search_string.lower()
    return [sent for sent in current_sents if lower_search in sent.sent.lower()]

def preview_matches(matches, current_sents, search_string):
    print(f"\nFound {len(matches)} matches for \"{search_string}\".")
    print("\nThree examples with context:")
    for i, match in enumerate(matches[:3], 1):
        idx = current_sents.index(match)
        prev = current_sents[idx-1].sent.rstrip() if idx > 0 else None
        next_sent = current_sents[idx+1].sent.lstrip() if idx < len(current_sents)-1 else None
        print(f"Example {i}:")
        if prev:
            print(f"Previous: {prev}")
        print(f"Match:    {match.sent.strip()}")
        if next_sent:
            print(f"Next:     {next_sent}")
        print()

def build_new_text(current_text, matches, model, tokenizer, gen_kwargs, search_string, instruction):
    replacements = []
    total = len(matches)
    for count, match in enumerate(matches, 1):
        original_slice = current_text[match.start:match.end]
        leading_ws = original_slice[:len(original_slice) - len(original_slice.lstrip())]
        trailing_ws = original_slice[len(original_slice.rstrip()):]
        clean_original = original_slice.strip()
        print(f"\n{count}/{total}: {clean_original}")
        new_core = process_sentence(clean_original, model, tokenizer, gen_kwargs, search_string, instruction)
        print(f"→ {new_core}")
        new_sentence = leading_ws + new_core + trailing_ws
        replacements.append((match.start, match.end, new_sentence))
    replacements.sort(key=lambda x: x[0])
    new_parts = []
    pos = 0
    for start, end, new_sent in replacements:
        new_parts.append(current_text[pos:start])
        new_parts.append(new_sent)
        pos = end
    new_parts.append(current_text[pos:])
    return "".join(new_parts)

def main():
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
        return
    with open(input_file, "r", encoding="utf-8") as f:
        full_text = f.read()
    seg = pysbd.Segmenter(language="en", clean=False, char_span=True)
    current_sents = seg.segment(full_text)
    if not current_sents:
        print("No sentences detected in the file.")
        return
    model, tokenizer = load_model_and_tokenizer()
    stop_sequences = get_stop_sequences(tokenizer)
    gen_kwargs = get_gen_kwargs(tokenizer, stop_sequences)
    current_text = full_text
    while True:
        search_string = input("\nEnter search string (empty to quit): ").strip()
        if not search_string:
            break
        matches = find_matching_sentences(current_sents, search_string)
        if not matches:
            print("No matches found.")
            continue
        preview_matches(matches, current_sents, search_string)
        proceed = input("Proceed with LLM editing of these sentences? (y/n): ").lower()
        if proceed != "y":
            continue
        instruction = input("\nEnter rewriting instruction (Enter for default): ").strip()
        if not instruction:
            instruction = "Change the meaning of the sentence to the opposite."
        print("\nProcessing sentences...")
        current_text = build_new_text(current_text, matches, model, tokenizer, gen_kwargs, search_string, instruction)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(current_text)
        print(f"\nEdited text saved to {output_file}")
        again = input("\nPerform another search/edit on the updated text? (y/n): ").lower()
        if again == "y":
            current_sents = seg.segment(current_text)
            print("Document re-segmented for next round.")
        else:
            break

if __name__ == "__main__":
    main()
