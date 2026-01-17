import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import pysbd
import settings

input_file = "input.txt"
output_file = "output.txt"

def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(settings.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME, trust_remote_code=True)
    return model, tokenizer

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_sequences):
        self.stop_sequences = stop_sequences or []
    def __call__(self, input_ids, scores, **kwargs):
        if not self.stop_sequences:
            return False
        for seq in self.stop_sequences:
            if len(seq) == 0 or len(seq) > input_ids.shape[1]:
                continue
            if input_ids[0, -len(seq):].tolist() == seq:
                return True
        return False

def get_stop_sequences(tokenizer):
    stop_strings = ["\nYou:", "\nUser:", "\nAssistant:"]
    return [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings]

def get_gen_kwargs(tokenizer, stop_sequences):
    return {
        "max_new_tokens": settings.MAX_TOKENS,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens(stop_sequences)]),
    }

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
    try:
        output_ids = model.generate(**enc, **gen_kwargs)
        output_text = tokenizer.decode(output_ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        return output_text.strip()
    except Exception as e:
        print(f"Error during generation: {e}")
        return sentence

def find_matching_sentences(current_sents, search_string):
    lower_search = search_string.lower()
    return [sent for sent in current_sents if lower_search in sent.sent.lower()]

def preview_matches(matches, search_string):
    total = len(matches)
    if total == 0:
        return
    print(f"\nFound {total} matches for \"{search_string}\".")
    display_limit = 20
    displayed = matches[:display_limit]
    print("\nMatches:\n")
    for i, match in enumerate(displayed, 1):
        print(f"{i}: {match.sent.strip()}")
        print()
    if total > display_limit:
        print(f"... and {total - display_limit} more matches.")

def build_new_text(current_text, matches, model, tokenizer, gen_kwargs, search_string, instruction):
    replacements = []
    total = len(matches)
    for count, match in enumerate(matches, 1):
        original_slice = current_text[match.start:match.end]
        leading_ws = original_slice[:len(original_slice) - len(original_slice.lstrip())]
        trailing_ws = original_slice[len(original_slice.rstrip()):]
        clean_original = original_slice.strip()
        print(f"\n[{count}/{total}] Processing: {clean_original}")
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
    current_text = full_text
    model = None
    tokenizer = None
    stop_sequences = None
    gen_kwargs = None
    while True:
        search_string = input("\nEnter search string (empty to quit): ").strip()
        if not search_string:
            break
        matches = find_matching_sentences(current_sents, search_string)
        if not matches:
            print("No matches found.")
            continue
        preview_matches(matches, search_string)
        total_matches = len(matches)
        selection = input(f"Enter numbers to edit (1-{total_matches}), comma-separated (e.g. 1,3,5-7 or 'all'), or blank to skip: ").strip().lower()
        if not selection:
            continue
        if selection == "all":
            selected_nums = set(range(1, total_matches + 1))
        else:
            selected_nums = set()
            try:
                for part in selection.split(','):
                    part = part.strip()
                    if not part:
                        continue
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        if start > end or start < 1 or end > total_matches:
                            raise ValueError
                        selected_nums.update(range(start, end + 1))
                    else:
                        num = int(part)
                        if num < 1 or num > total_matches:
                            raise ValueError
                        selected_nums.add(num)
            except ValueError:
                print("Invalid selection.")
                continue
        if not selected_nums:
            continue
        selected_matches = [matches[num - 1] for num in sorted(selected_nums)]
        print(f"\nSelected {len(selected_matches)} sentence(s):")
        for i, match in enumerate(selected_matches, 1):
            print(f"{i}. {match.sent.strip()}")
        instruction = input("\nEnter rewriting instruction (blank for default): ").strip()
        if not instruction:
            instruction = 'Correct the grammar. Maintaining the meaning of the sentence.'
        print(f"\nProcessing {len(selected_matches)} sentences...")
        if model is None:
            print("Loading model...")
            model, tokenizer = load_model_and_tokenizer()
            stop_sequences = get_stop_sequences(tokenizer)
            gen_kwargs = get_gen_kwargs(tokenizer, stop_sequences)
        current_text = build_new_text(current_text, selected_matches, model, tokenizer, gen_kwargs, search_string, instruction)
        if os.path.exists(output_file):
            confirm = input(f"\n{output_file} exists. Overwrite? (y/n): ").lower()
            if confirm != 'y':
                print("Skipped saving.")
                continue
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(current_text)
        print(f"\nEdited text saved to {output_file}")
        again = input("\nPerform another search on the updated text? (y/n): ").lower()
        if again == "y":
            current_sents = seg.segment(current_text)
            print("Document re-segmented for next round.")
        else:
            break

if __name__ == "__main__":
    main()
