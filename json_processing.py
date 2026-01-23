import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from process_input import load_model_and_tokenizer, get_stop_sequences, get_gen_kwargs, StopOnTokens
import settings

input_file = "input.json"
output_file = "output.json"

def split_text_smart(text, max_chars):
    if len(text) <= max_chars:
        return [text]
    total_len = len(text)
    num_chunks = (total_len + max_chars - 1) // max_chars
    if num_chunks * max_chars - total_len < max_chars * 0.3:
        num_chunks += 1
    chunk_size = total_len // num_chunks
    chunks = []
    pos = 0
    while pos < total_len:
        if pos + chunk_size >= total_len:
            chunks.append(text[pos:])
            break
        end_pos = pos + chunk_size
        search_start = max(pos + int(chunk_size * 0.8), pos + 1)
        search_end = min(end_pos + int(chunk_size * 0.2), total_len)
        best_break = end_pos
        for i in range(search_end, search_start - 1, -1):
            if i < total_len and text[i] in '.!?\n':
                if i + 1 < total_len and text[i + 1] == ' ':
                    best_break = i + 1
                    break
                else:
                    best_break = i + 1
                    break
        if best_break == end_pos:
            for i in range(search_end, search_start - 1, -1):
                if i < total_len and text[i] == ' ':
                    best_break = i + 1
                    break
        chunks.append(text[pos:best_break].strip())
        pos = best_break
    return chunks

def process_text(text, model, tokenizer, gen_kwargs, memory=None, use_memory=False, use_simple_memory=False):
    parts = []
    if memory:
        if use_memory:
            parts.append(f"\nContext: Previous input: \"{memory[0]}\", Previous output: \"{memory[1]}\"")
        elif use_simple_memory:
            parts.append(f"\nContext: Previous output: \"{memory[1]}\"")
    parts.append(settings.BASE)
    parts.append(f"Task: {settings.REQUEST_JSON[:80]}...")
    parts.append(f"Input content: {text}")
    parts.append(f"Processed:")
    base = "\n".join(parts)
    if settings.PRINT:
        print(base)
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
    use_memory = settings.MEMORY_JSON
    use_simple_memory = settings.SIMPLE_MEMORY_JSON
    if use_memory and use_simple_memory:
        raise ValueError("Cannot enable both MEMORY_JSON and SIMPLE_MEMORY_JSON. Choose one memory mode or neither.")
    model, tokenizer = load_model_and_tokenizer()
    stop_sequences = get_stop_sequences(tokenizer)
    gen_kwargs = get_gen_kwargs(tokenizer, stop_sequences)
    memory = None
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line_num, raw_line in enumerate(infile, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            if "text" not in data:
                print(f"Warning: No 'text' field found on line {line_num}, skipping")
                outfile.write(raw_line)
                outfile.flush()
                continue
            if data.get("label") == "exclude":
                outfile.write(raw_line)
                outfile.flush()
                continue
            input_text = data["text"]
            max_chars = getattr(settings, 'MAX_TEXT_CHARS', None)
            if max_chars and len(input_text) > max_chars:
                chunks = split_text_smart(input_text, max_chars)
                output_chunks = []
                for chunk in chunks:
                    chunk_output = process_text(chunk, model, tokenizer, gen_kwargs, memory, use_memory, use_simple_memory)
                    output_chunks.append(chunk_output)
                    if use_memory or use_simple_memory:
                        memory = (chunk, chunk_output)
                output_text = " ".join(output_chunks)
            else:
                output_text = process_text(input_text, model, tokenizer, gen_kwargs, memory, use_memory, use_simple_memory)
                if use_memory or use_simple_memory:
                    memory = (input_text, output_text)
            print(f"{output_text}")
            print()
            data["text"] = output_text
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            outfile.flush()

if __name__ == "__main__":
    main()
