import torch
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM, LlamaForCausalLM

# Alternative class definition (reduce memory use)
class CADModel(LlamaForCausalLM):
    def __init__(self, base_model, alpha, *args, **kwargs):
        super().__init__(config=base_model.config)
        self.base_lm = base_model
        self.alpha = alpha
        self.query_ids = None
        # self.k = k

    def forward(self, *args, **kwargs): 
        x_full = self.base_lm.forward(*args, **kwargs)

        # kwargs['input_ids'] = kwargs['input_ids'][...,-self.k:]  # KeyError for T5 model 
        # self.query_ids = self.query_ids.cuda("cuda")
        kwargs['input_ids'] = self.query_ids
        x_ques = self.base_lm.forward(*args, **kwargs)

        x_full.logits[...,-1,:] = (1.0 + self.alpha) * x_full.logits[...,-1,:] - self.alpha * x_ques.logits[...,-1,:]

        return x_full
    
    def update(self, query_ids):
        self.query_ids = query_ids


def context_aware_wrapper(base_model, alpha):
    # k: indicates the start of the question text ids
    model_type = type(base_model)

    class CADModel(model_type):
        def __init__(self, base_model, alpha, *args, **kwargs):
            super().__init__(config=base_model.config)
            self.base_lm = base_model
            self.alpha = alpha
            self.query_ids = None
            # self.k = k

        def forward(self, *args, **kwargs): 
            x_full = self.base_lm.forward(*args, **kwargs)

            # kwargs['input_ids'] = kwargs['input_ids'][...,-self.k:]  # KeyError for T5 model 
            # self.query_ids = self.query_ids.cuda("cuda")
            kwargs['input_ids'] = self.query_ids
            x_ques = self.base_lm.forward(*args, **kwargs)

            x_full.logits[...,-1,:] = (1.0 + self.alpha) * x_full.logits[...,-1,:] - self.alpha * x_ques.logits[...,-1,:]

            return x_full
        
        def update(self, query_ids):
            self.query_ids = query_ids
    
    # class CADT5Model(model_type):
    #     def __init__(self, base_model, alpha, k, *args, **kwargs):
    #         super().__init__(config=base_model.config)
    #         self.base_lm = base_model
    #         self.alpha = alpha
    #         self.k = k

    #     def forward(self, input_ids=None, attention_mask=None, **kwargs):
    #         x_full = self.base_lm.forward(input_ids=input_ids,
    #                                       attention_mask=attention_mask,
    #                                       **kwargs)
    #         if input_ids:
    #             input_ids = input_ids[...,-self.k:]
    #         x_ques = self.base_lm.forward(input_ids=input_ids,
    #                                      attention_mask=attention_mask,
    #                                      **kwargs)
    #         x_full.logits[...,-1,:] = (1.0 + self.alpha) * x_full.logits[...,-1,:] - self.alpha * x_ques.logits[...,-1,:]

    #         return x_full

    #     def update(self, k):
    #         # Update the start index of question text
    #         self.k = k
    
    # if isinstance(base_model, T5ForConditionalGeneration):
    #     cad_model = CADT5Model(base_model, alpha, k)
    # else:
    cad_model = CADModel(base_model, alpha)

    return cad_model