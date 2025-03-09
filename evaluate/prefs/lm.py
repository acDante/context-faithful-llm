from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import time
import os


class QwenModel(object):
    def __init__(self, model_name):
        with open('prefs/hf.key') as f:
            self.hf_token = f.read().rstrip('\n')
        # TODO: change model_name and cache_dir when runnig on EIDF
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            token=self.hf_token,
            cache_dir="/mnt/ssd/llms"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            cache_dir="/mnt/ssd/llms"
        )
    
    def generate(self, prompt, max_output_tokens):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_output_tokens,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


class LlamaModel(object):
    def __init__(self, model_name):
        with open('prefs/hf.key') as f:
            self.hf_token = f.read().rstrip('\n')
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            quantization_config=quantization_config,
            cache_dir="/mnt/ssd/llms",
            token=self.hf_token
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="/mnt/ssd/llms",
            token=self.hf_token
        )

    def generate(self, prompt, max_output_tokens):
        messages = messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.quantized_model.device)
        generated_ids = self.quantized_model.generate(
            **model_inputs,
            max_new_tokens=max_output_tokens,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
        

# class LlamaModel(object):
#     def __init__(self, model_name):
#         # TODO: change model_name and cache_dir when runnig on EIDF
#         with open('prefs/hf.key') as f:
#             self.hf_token = f.read().rstrip('\n')
#         self.gen_pipeline = pipeline(
#             "text-generation",
#             model=model_name,
#             model_kwargs={"torch_dtype": torch.bfloat16,
#                           "cache_dir": "/mnt/ssd/llms"},
#             device_map="auto",
#             token=self.hf_token
#         )
    
#     def generate(self, prompt, max_output_tokens):
#         messages = [{"role": "user", "content": prompt}]

#         outputs = self.gen_pipeline(
#             messages,
#             max_length=max_output_tokens,
#             do_sample=False
#         )

#         response = outputs[0]["generated_text"][-1]['content']
#         return response

if __name__ == "__main__":
    # Test the models
    qwen = QwenModel("Qwen/Qwen2.5-7B-Instruct")
    # llama = LlamaModel("meta-llama/Meta-Llama-3.1-8B-Instruct")
    prompt = "What is the capital of France?"
    print(qwen.generate(prompt, 100))
    # print(llama.generate(prompt, 100))