from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
from tqdm import tqdm
import json
# import logging
# logging.disable(logging.WARNING)

BEGIN_TOKEN = "<｜fim▁begin｜>"
FILL_TOKEN = "<｜fim▁hole｜>"
END_TOKEN = "<｜fim▁end｜>"
IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"

class Tools:
    @staticmethod
    def load_jsonl(path):
        with open(path, 'r') as f:
            return [json.loads(line) for line in f.readlines()]

    @staticmethod
    def dump_jsonl(obj, path):
        with open(path, 'w') as f:
            for line in obj:
                f.write(json.dumps(line) + '\n')

def deepseek_build_infilling_prompt(prompt: str):
    prompt = prompt.replace('<FILL_FUNCTION_BODY>', '\n' + FILL_TOKEN + '\n')
    return BEGIN_TOKEN + prompt + END_TOKEN

def codellama_build_infilling_prompt(prompt):
    # prompt = prompt.replace('<FILL_FUNCTION_BODY>', '<FILL_ME>')
    # return prompt
    prefix_tokens, suffix_tokens = prompt.split('<FILL_FUNCTION_BODY>')
    return '▁<PRE>' + prefix_tokens + '\n' + '▁<SUF>' + '\n' + suffix_tokens + '▁<MID>'

def gemma_build_infilling_prompt(prompt):
    # prompt = prompt.replace('<FILL_FUNCTION_BODY>', '<FILL_ME>')
    # return prompt
    prefix_tokens, suffix_tokens = prompt.split('<FILL_FUNCTION_BODY>')
    return '<|fim_prefix|>' + prefix_tokens + '\n' + '<|fim_suffix|>' + '\n' + suffix_tokens + '<|fim_middle|>'

def starcoder_build_infilling_prompt(prompt):
    # prompt = prompt.replace('<FILL_FUNCTION_BODY>', '<FILL_ME>')
    # return prompt
    prefix_tokens, suffix_tokens = prompt.split('<FILL_FUNCTION_BODY>')
    return '<fim_prefix>' + prefix_tokens + '\n' + '<fim_suffix>' + '\n' + suffix_tokens + '<fim_middle>'

def split_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def write_string_to_file(absolute_filename, string):
    with open(absolute_filename, 'a') as fout:
        fout.write(string)

def run(args):
    model_id = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, )
    # print(tokenizer)
    if args.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto', load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto').cuda()
    
    model.eval()
    if 'codellama' in args.model_id or 'star' in args.model_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left" # Fix weird overflow issue with fp16 training
    
    print(f'generating from {args.input_file}')
    dataset = Tools.load_jsonl(args.input_file)
    
    if args.task == 'l_context':
        sources = [f"{line['prompt']}\n" for line in dataset]
    elif args.task == 'lr_context':
        if 'deepseek' in args.model_id:
            sources = [
                deepseek_build_infilling_prompt(line['prompt'])
                for line in dataset
            ]
        elif 'llama' in args.model_id:
            sources = [
                codellama_build_infilling_prompt(line['prompt'])
                for line in dataset
            ]
        elif 'gemma' in args.model_id:
            sources = [
                gemma_build_infilling_prompt(line['prompt'])
                for line in dataset
            ]
        elif 'star' in args.model_id:
            sources = [
                starcoder_build_infilling_prompt(line['prompt'])
                for line in dataset
            ]
        else:
            raise ValueError("Model not supported")
    else:
        raise ValueError("Task not supported")
    
    batch_list = split_batch(sources, args.batch_size)
    len_batch = len(sources) // args.batch_size
    gen_text = []
    with tqdm(total=len_batch, desc="gen") as pbar:
        for batch in batch_list:
            if args.padding == 'longest':
                model_inputs = tokenizer(batch, return_tensors="pt", padding=True, max_length=args.max_length, truncation=True).to("cuda")
            else:
                model_inputs = tokenizer(batch, return_tensors="pt", padding='max_length', max_length=args.max_length, truncation=True).to("cuda")
            
            generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

            truncated_ids = [ids[len(model_inputs[idx]):] for idx, ids in enumerate(generated_ids)]

            output = tokenizer.batch_decode(truncated_ids, skip_special_tokens=True)

            gen_text.extend(output)

            for idx, source in enumerate(batch):
                # print(idx, source)
                # write_string_to_file(args.output_file, output[idx][len(source):] + '<nl>')
                # output[idx] = output[idx].encode('utf-8')
                try:
                    write_string_to_file(args.output_file, output[idx] + '<nl>')
                except Exception as e:
                    print(e)
                    write_string_to_file(args.output_file, '<nl>')
                # print(output[0][len(sources[0]):], output[1][len(sources[1]):])
            pbar.update(1)
    
    print(f'generated {len(gen_text)} samples')
    assert len(gen_text) == len(sources)
    new_lines = []
    for line, gen in zip(dataset, gen_text):
        new_lines.append({
            'prompt': line['prompt'],
            'metadata': line['metadata'],
            'choices': [{'text': gen}]
        })
    Tools.dump_jsonl(new_lines, args.input_file.replace('.jsonl', f'_{args.model_id.split("/")[-1]}.jsonl'))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='lr_context', type=str)
    parser.add_argument("--batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--load_in_8bit", action='store_true',
                        help="Load model 8 bit.")
    parser.add_argument("--model_id", type=str, default='deepseek-ai/deepseek-coder-6.7b-base')
    parser.add_argument("--input_file", type=str, default='rambo_defects4j_sketch_prompt.jsonl')
    parser.add_argument("--output_file", type=str, default="gen.output")
    parser.add_argument("--max_length", type=int, default=8000)
    parser.add_argument("--padding", type=str, default='longest')
    parser.add_argument("--max_new_tokens", type=int, default=400)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()