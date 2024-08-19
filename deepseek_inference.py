import torch
import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


BEGIN_TOKEN = "<｜fim▁begin｜>"
FILL_TOKEN = "<｜fim▁hole｜>"
END_TOKEN = "<｜fim▁end｜>"

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


class CodeGen:
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto').cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        # self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        # self.model.cuda()
        self.batch_size = batch_size
        print('done loading model')

    def _get_batchs(self, prompts, batch_size):
        batches = []
        for i in range(0, len(prompts), batch_size):
            batches.append(prompts[i:i+batch_size])
        return batches

    def _generate_batch(self, prompt_batch, max_new_tokens=400):
        prompts = self.tokenizer(prompt_batch, return_tensors='pt', truncation=True, ).to("cuda")
        print(prompts['input_ids'].size()[1])
        
        # for prompt in prompts['input_ids']:
        #     print(prompt)
        #     print(len(prompt))
        #     if len(prompt) > 2048:
        #         print('prompt too long, truncating')
        
        # with torch.no_grad():
            
        gen_tokens = self.model.generate(**prompts, max_new_tokens=max_new_tokens, do_sample=False)
        # gen_tokens = self.model.generate(
        #     input_ids = prompts['input_ids'].cuda(),
        #     attention_mask = prompts['attention_mask'].cuda(),
        #     do_sample=False,
        #     max_new_tokens=max_new_tokens,
        #     # max_length = 2048,
        # )
        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        for i in range(len(gen_text)):  # remove the prompt
            gen_text[i] = gen_text[i][len(prompt_batch[i]):]
        
        for gen in gen_text:
            with open('temp.out', 'a') as fout:
                fout.write(gen + '<nl>')
        
        return gen_text

    def batch_generate(self, file):
        print(f'generating from {file}')
        lines = Tools.load_jsonl(file)
        # have a new line at the end
        # prompts = [f"{line['prompt']}\n" for line in lines]
        prompts = [BEGIN_TOKEN + line['prompt'].split('<FILL_FUNCTION_BODY>')[0] + \
                '\n' + FILL_TOKEN + '\n' + line['prompt'].split('<FILL_FUNCTION_BODY>')[1] + END_TOKEN for line in lines]
        # print(prompts[0])
        batches = self._get_batchs(prompts, self.batch_size)
        gen_text = []
        for batch in tqdm.tqdm(batches):
            gen_text.extend(self._generate_batch(batch))
        print(f'generated {len(gen_text)} samples')
        assert len(gen_text) == len(prompts)
        new_lines = []
        for line, gen in zip(lines, gen_text):
            new_lines.append({
                'prompt': line['prompt'],
                'metadata': line['metadata'],
                'choices': [{'text': gen}]
            })
        Tools.dump_jsonl(new_lines, file.replace('.jsonl', f'_{self.model_name.split("/")[-1]}.jsonl'))


if __name__ == '__main__':
    file_path = 'defects4j_ground_truth_type_method_no_em_cutting_stable.jsonl'
    tiny_codegen = 'deepseek-ai/deepseek-coder-6.7b-base'

    cg = CodeGen(tiny_codegen, batch_size=1)
    cg.batch_generate(file_path)
