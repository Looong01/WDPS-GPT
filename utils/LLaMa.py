import os, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def llama2(questions):
    ## create a Llama session and ask questions
    path = os.path.join(os.path.abspath(os.getcwd()))
    model_name = os.path.join(path, "Llama-2-13b-chat-hf")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True) ## Load LLaMa tokenizer to GPU RAM
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto" ## Load LLaMa model to GPU Video RAM
                # , load_in_4bit=True   ## uncomment this line if your VRAM is less than 8GB
                # , load_in_8bit=True   ## uncomment this line if your VRAM is less than 16GB
                # , torch_dtype = torch.float16 ## uncomment this line if your VRAM is less than 30GB
                )
    f = open(os.path.join(path, 'log', 'Answers_llm.txt'), 'w', encoding='utf-8')                        
    for i, question in enumerate(questions):
        model_inputs = tokenizer(question.strip().split('        ')[1], return_tensors="pt").to("cuda") ## Tokenize the question and load it to GPU Video RAM
        output = model.generate(**model_inputs)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        print(answer)
        answer = ' '.join(line for line in answer.split('\n')[1:])
        print("*************************************************************************************************************************************")
        f.write(answer + '\n') ## save output into the file
        del model_inputs, output, answer, question, i ## delete the variables to free the GPU Video RAM for every question
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    f.close()

if __name__ == "__main__":
    path = os.path.join(os.path.abspath(os.getcwd()))
    with open(os.path.join(path, "Questions.txt"), "r") as f:
        questions = f.readlines()
    llama2(questions)
        