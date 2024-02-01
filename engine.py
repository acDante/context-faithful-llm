import openai
from api_secrets import get_api_key
from time import sleep
import tiktoken
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.nn import functional as F
import torch
from models import context_aware_wrapper

openai.api_key = get_api_key()

length_limit = {
    'text-davinci-003': 4096,
    'text-curie-001': 2048,
    'text-babbage-001': 2048,
    'text-ada-001': 2048,
    'gpt-3.5-turbo-instruct': 4096,
}

class Engine:
    def __init__(self, engine='text-davinci-003'):
        self.engine = engine
        self.tokenizer = tiktoken.encoding_for_model(engine)

    def check_prompt_length(self, prompt, max_tokens=64):
        prompt_length = len(self.tokenizer.encode(prompt))
        if prompt_length + max_tokens >= length_limit[self.engine]:  # Prompt is too long
            return True
        return False

    def complete(self, prompt, max_tokens=64):
        num_retry = 0
        while True:
            try:
                breakpoint()
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=prompt,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                print(e)
                if num_retry >= 5:  # Retried too many times
                    print('Retried too many times, skip this instance.')
                    return None
                sleep(2)
                num_retry += 1
                continue
            break
        answer = response.choices[0].text
        return answer

    def get_prob(self, prompt, num_tokens):
        num_retry = 0
        while True:
            try:
                response = openai.Completion.create(
                    engine=self.engine,
                    prompt=prompt,
                    max_tokens=0,
                    logprobs=1,
                    echo=True,
                )
                token_logprobs = response.choices[0].logprobs.token_logprobs[-num_tokens:]
                seq_prob = sum(token_logprobs)
            except Exception as e:
                print(e)
                if num_retry >= 5:  # Retried too many times
                    print('Retried too many times, skip this instance.')
                    return None
                sleep(2)
                num_retry += 1
                continue
            break
        return seq_prob

class LLamaModel:
    """Provide support for both Llama and Llama-2 models"""
    def __init__(self, model_name='meta-llama/Llama-2-7b-hf', use_cad=False, alpha=None, k=None):
        download_path = "/home/hpcdu1/experiments/huggingface-hub"
        access_token = "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"   # for access to Llama2
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       token=access_token,
                                                       cache_dir=download_path)

        if "chat" in self.model_name:
            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                              device_map="auto",
                                                              torch_dtype=torch.bfloat16,
                                                              token=access_token,
                                                              cache_dir=download_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                              device_map="auto",
                                                              torch_dtype=torch.bfloat16,
                                                              token=access_token,
                                                              cache_dir=download_path)
        # CAD Decoding
        if use_cad:
            self.model = context_aware_wrapper(self.model, alpha=alpha, k=k)
            print("Initialized CAD model")
    
    def check_prompt_length(self, prompt, max_tokens=64):
        prompt_length = len(self.tokenizer.encode(prompt))
        if prompt_length + max_tokens >= 4096:  # Prompt is too long
            return True
        return False
    
    def complete(self, prompt, max_tokens=64):
        if "chat" in self.model_name:
            # TODO: Need to modify this for prompt with demonstrations
            messages = [{
                "role": "user",
                "content": prompt
            }]

            input_ids = self.tokenizer(prompt,
                                       max_length=4096,
                                       truncation=True,
                                       return_tensors="pt").input_ids.cuda("cuda")
            
            outputs = self.model.generate(input_ids,
                                          max_new_tokens=max_tokens,
                                          do_sample=False,
                                          temperature=0.0,
                                          top_p=0.0)
    
        else:
            input_ids = self.tokenizer(prompt,
                                       max_length=4096,
                                       truncation=True,
                                       return_tensors="pt").input_ids.cuda("cuda")
            
            outputs = self.model.generate(input_ids,
                                          max_new_tokens=max_tokens,
                                          do_sample=False,
                                          temperature=0.0)

        raw_output = self.tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
        output = raw_output.split("\n")[0]

        return output

    def get_prob(self, prompt, num_tokens):
        with torch.no_grad():
            inputs = self.tokenizer([prompt], return_tensors="pt")
            input_ids = inputs['input_ids'].to("cuda")
            outputs = self.model(input_ids)

            logits = outputs.logits[0, :input_ids.shape[-1] - 1]  # ignore the logits of the last token (prob for the next word)
            probs = F.log_softmax(logits, dim=-1)
            log_probs = []

            input_ids = input_ids.squeeze()
            for idx, input_id in enumerate(list(input_ids)[1:]):
                log_probs.append(probs[idx, input_id].item())

            return sum(log_probs[-num_tokens:])
    
    # Make inference with CAD decoding
    def complete(self, prompt, question_prompt, max_tokens=64):
        # Context + Question
        input_ids = self.tokenizer(prompt,
                                   max_length=4096,
                                   truncation=True,
                                   return_tensors="pt").input_ids.cuda("cuda")
        # Question only
        question_ids = self.tokenizer(question_prompt,
                                      max_length=4096,
                                      truncation=True,
                                      return_tensors="pt").input_ids.cuda("cuda")
        
        self.model.update(question_ids.shape[-1])

        outputs = self.model.generate(input_ids,
                                      max_new_tokens=max_tokens,
                                      do_sample=False,
                                      temperature=0.0,
                                      top_p=0.0)

        raw_output = self.tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
        output = raw_output.split("\n")[0]

        return output
    
    # Multi-choice question answering with CAD decoding
    def get_prob(self, prompt, question_prompt, num_tokens):
        with torch.no_grad():
            inputs = self.tokenizer([prompt], return_tensors="pt")
            input_ids = inputs['input_ids'].to("cuda")

            question_ids = self.tokenizer([question_prompt],
                                          return_tensors="pt").input_ids.cuda("cuda")
            self.model.update(question_ids.shape[-1])

            outputs = self.model(input_ids)
            logits = outputs.logits[0, :input_ids.shape[-1] - 1]  # ignore the logits of the last token (prob for the next word)
            probs = F.log_softmax(logits, dim=-1)
            log_probs = []

            input_ids = input_ids.squeeze()
            for idx, input_id in enumerate(list(input_ids)[1:]):
                log_probs.append(probs[idx, input_id].item())

            return sum(log_probs[-num_tokens:])

# the author's implementation
# This implementation works! 
class Llama2Model:
    def __init__(self, engine='meta-llama/Llama-2-7b-chat-hf'):
        self.engine = LlamaForCausalLM.from_pretrained(
            engine,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(engine)

    def check_prompt_length(self, prompt, max_tokens=64):
        prompt_length = len(self.tokenizer.encode(prompt))
        if prompt_length + max_tokens >= 4096:  # Prompt is too long
            return True
        return False
    
    def get_prob(self, prompt, num_tokens=0):
        with torch.no_grad():
            inputs = self.tokenizer([prompt], return_tensors="pt")
            input_ids = inputs['input_ids'][0]
            inputs = inputs.to('cuda:0')
            outputs = self.engine(**inputs)
            logits = outputs.logits[0][:len(input_ids) - 1]
            logits = F.log_softmax(logits, dim=-1)
            log_probs = []
            for idx, input_id in enumerate(list(input_ids)[1:]):
                log_probs.append(logits[idx][input_id].item())
            
            return sum(log_probs[-num_tokens:])

