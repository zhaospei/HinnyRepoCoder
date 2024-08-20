import torch
import tqdm
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


DEEPSEEK_BEGIN_TOKEN = "<｜fim▁begin｜>"
DEEPSEEK_FILL_TOKEN = "<｜fim▁hole｜>"
DEEPSEEK_END_TOKEN = "<｜fim▁end｜>"

STARCODER_BEGIN_TOKEN = "<fim_prefix>"
STARCODER_FILL_TOKEN = "<fim_suffix>"
STARCODER_END_TOKEN = "<fim_middle>"

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
    """
    The CodeGen class is used to generate code snippets based on the given prompts.
    Args:
        model_name (str): The name of the model to be used.
        batch_size (int): The batch size to be used.
        context (str): The context to be used.
    """
    def __init__(self, model_name, batch_size, context):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto').cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
        if 'deepseek' in model_name:
            self.begin_token = DEEPSEEK_BEGIN_TOKEN
            self.fill_token = DEEPSEEK_FILL_TOKEN
            self.end_token = DEEPSEEK_END_TOKEN
        elif 'starcoder' in model_name:
            self.begin_token = STARCODER_BEGIN_TOKEN
            self.fill_token = STARCODER_FILL_TOKEN
            self.end_token = STARCODER_END_TOKEN
        self.context = context
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
        gen_tokens = self.model.generate(**prompts, max_new_tokens=max_new_tokens, do_sample=False)
        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        
        for i in range(len(gen_text)):  # remove the prompt
            gen_text[i] = gen_text[i][len(prompt_batch[i]):]
        
        for gen in gen_text:
            with open('temp.out', 'a', encoding='utf-8') as fout:
                fout.write(gen + '<nl>')
        
        return gen_text

    def batch_generate(self, file):
        """
        Generate text samples based on the given file.
        Args:
            file (str): The path to the file containing the prompts.
        Raises:
            ValueError: If the context is not supported.
        Returns:
            None
        """
        print(f'generating from {file}')
        lines = Tools.load_jsonl(file)
        if self.context == 'l_context':
        # have a new line at the end
            prompts = [f"{line['prompt']}\n" for line in lines]
        elif self.context == 'lr_context':
            prompts = [self.begin_token + line['prompt'].split('<FILL_FUNCTION_BODY>')[0] + \
               '\n' + self.fill_token + '\n' + line['prompt'].split('<FILL_FUNCTION_BODY>')[1] + self.end_token for line in lines]
        else:
            raise ValueError(f'context {self.context} not supported')
              
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


def run(run_args):
    """
    Run the deepseek inference.

    Args:
        args (Namespace): The command line arguments.

    Returns:
        None
    """
    cg = CodeGen(run_args.model_id, batch_size=run_args.batch_size, context=run_args.context)
    cg.batch_generate(run_args.file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='defects4j_sketch_type_method_no_em_lcontext_stable.jsonl')
    parser.add_argument('--model_id', type=str, default='deepseek-ai/deepseek-coder-6.7b-base')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--context', type=str, default='lr_context')
    run_args = parser.parse_args()
    
    run(run_args)