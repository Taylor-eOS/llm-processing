import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pysbd
from process_input import load_model_and_tokenizer, get_stop_sequences, get_gen_kwargs, StopOnTokens
import settings

input_file = "input.json"
output_file = "output.json"

def split_text_into_sentences(text):
    text = text.strip()
    if not text:
        return []
    segmenter = pysbd.Segmenter(language="en", clean=True)
    sentences = segmenter.segment(text)
    result = []
    for s in sentences:
        s = s.strip()
        if s:
            result.append(s)
    if not result:
        result = [text]
    return result

def split_long_sentence(sentence, max_chars):
    if len(sentence) <= max_chars:
        return [sentence]
    chunks = []
    start = 0
    length = len(sentence)
    while start < length:
        end = min(start + max_chars, length)
        if end == length:
            chunk = sentence[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        split_pos = end
        found_break = False
        for pos in range(end - 1, start, -1):
            if sentence[pos].isspace():
                split_pos = pos
                found_break = True
                break
        if not found_break:
            split_pos = end
        chunk = sentence[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        start = split_pos
        while start < length and sentence[start].isspace():
            start += 1
    return chunks

def process_text(text, model, tokenizer, gen_kwargs, memory=None, use_memory=False, use_simple_memory=False):
    parts = []
    if memory:
        if use_memory:
            parts.append(f"\nContext: Previous input: {memory[0]}, Previous output: {memory[1]}")
        elif use_simple_memory:
            parts.append(f"\nContext: Previous output: {memory[1]}")
    parts.append(settings.BASE)
    parts.append(f"Task: {settings.REQUEST_JSON}")
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
            sentences = split_text_into_sentences(input_text)
            output_sentences = []
            max_chars = settings.MAX_TEXT_CHARS
            for sentence in sentences:
                if max_chars and len(sentence) > max_chars:
                    sub_chunks = split_long_sentence(sentence, max_chars)
                    for chunk in sub_chunks:
                        chunk_output = process_text(chunk, model, tokenizer, gen_kwargs, memory, use_memory, use_simple_memory)
                        output_sentences.append(chunk_output)
                        if use_memory or use_simple_memory:
                            memory = (chunk, chunk_output)
                else:
                    sentence_output = process_text(sentence, model, tokenizer, gen_kwargs, memory, use_memory, use_simple_memory)
                    output_sentences.append(sentence_output)
                    if use_memory or use_simple_memory:
                        memory = (sentence, sentence_output)
            output_text = " ".join(output_sentences)
            print(f"{output_text}")
            print()
            data["text"] = output_text
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            outfile.flush()

if __name__ == "__main__":
    main()
