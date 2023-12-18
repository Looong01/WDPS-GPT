import gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def llama2(questions):
    model_name = "./Llama-2-13b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto"
                # , load_in_8bit=True
                # , local_files_only=True, model_type='llama'
                # , torch_dtype = torch.float16
                )
                                                
    for i, question in enumerate(questions):
        model_inputs = tokenizer(question.strip().split('        ')[1], return_tensors="pt").to("cuda")
        output = model.generate(**model_inputs)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        print(answer)
        answer = ' '.join(line for line in answer.split('\n')[1:])
        print("*************************************************************************************************************************************")
        with open('log/Answers_llm.txt', 'a', encoding='utf-8') as file:
               file.write(answer + '\n')
        del model_inputs, output, answer, question, i
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == "__main__":
    with open("Questions.txt", "r") as f:
        questions = f.readlines()
    llama2(questions)
        