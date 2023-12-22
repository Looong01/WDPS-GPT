# WDPS-GPT
Assignment of WDPS

### Description

Follow the [Instructions](#instructions) if you want to use.

The goal of the assignment is to implement a program that post-processes the output of large language models (like ChatGPT, etc.) to improve the quality of its answers. To this end, your solution is asked to 1) parse the output of a language model and to extract a clean answer and 2) check, with the aid of existing knowledge bases, which are available on the web, whether the answer is correct or not.

We are called to implement a program that receives as input a question (or more in general a text to be completed), which we henceforth call A and returns as output four things:
1. The raw text returned by a large language model (that is, what you would get if you query the language model as is). We call it B
2. The answer extracted from B. This answer can be of two types: either yes/no or a Wikipedia entity
3. The correctness of the extracted answer (correct/incorrect)
4. Entities that have been extracted from A and B
  
By the submission requirements:
The inputs of the questions should be of the form
```
<ID question><TAB>text of the question/completion<newline>".  
```
For instance, 
```
1        Is the sky usually blue?'''
```
Then the outputs of the results will be of the form
```
"<ID question><TAB>[R,A,C,E]<answer>"
```
where "R" indicates the raw text produced by the language model, "A" is the extracted answer, "C" is the tag correct/incorrect and "E" are the entities extracted.

For instance, 
```

1        R"Yes, the sky is usually blue. This is because sunlight reaches Earth's atmosphere and is scattered in all directions by all the gases and particles in the air. Blue light is scattered more than the other colors because it travels as shorter, smaller waves[^1^][1][^2^][2][^3^][3]. However, the sky isn't always blue. When the sun is low in the sky, at sunrise or sunset, it can take on a red hue[^2^][2]. This is explained by the same physics — Rayleigh scattering — as the blueness of the sky at other times[^2^][2]."
1        A"yes"
1        C"Correct"
1        E"blueness,sunset,blue        https://spaceplace.nasa.gov/blue-sky/en/; https://www.space.com/why-is-the-sky-blue; https://scijinks.gov/blue-sky/"

```
### Hardware Environment
1. CPU: i7-9700
2. RAM: 64GB
3. GPU: 2 * AMD Radeon Pro W7900 (VRAM: 2 * 48GB) or 2* Nvidia Quadro RTX 6000 Ada (VRAM: 2 * 48GB)
4. Hard disk: 2TB
  
### Software Environment
1. OS: Ubuntu 22.04.3 with Kernel Version 6.2
3. ROCm: Version 5.6 Or CUDA: Version: 12.1
4. Miniconda3
5. Git
6. Microsoft Edge Browser
7. Python 3.10
8. PyTorch 2.1.2
  
### Instructions
1. Prepare for the CPU, RAM, Hard disk and GPU(s) with enough VRAM(at least 60GB totally).
2. Install OS Ubuntu 22.04 with Kernel Version 6.2
3. Install GPU driver and ROCm with Version 5.6 or CUDA with Version 12.1
4. Install Microsoft Edge Browser
5. Install Miniconda3 with the latest version
6. Install Git
7. ```conda create -n PyTorch python=3.10 -y```
8. ```conda activate PyTorch```
9. ```conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia``` for CUDA(Nvidia GPUs)
10. Or ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6``` for ROCm(AMD GPUs)
11. ```git clone https://github.com/Looong01/WDPS-GPT```
12. ```cd WDPS-GPT```
13. ```pip install -r requirements.txt```
14. ```git lfs install```
15. ```git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2```
16. ```git clone https://huggingface.co/bert-large-uncased```
17. Make sure you can have got the authority to assess LLaMa2 models from Huggingface. If you're not allowed, go to "https://ai.meta.com/resources/models-and-libraries/llama-downloads/" to fill the form and get the authority.
18. ```git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf```
19. Get the cookies:
```
- Use Microsoft Edge Browser to open the [Copilot web page](https://copilot.microsoft.com/) and Login your own Microsoft account.
- Open the developer tools in your browser (usually by pressing `F12` or right-clicking on the chat dialog and selecting `Inspect`).
- Select the `Network` tab to view all requests sent to Copilot.
- Write a message on the chat dialog that appears on the web page.
- Find a request named `create?bundleVersion=XYZ` and click on it.
- Scroll down to the requests headers section and copy the entire value after the `Cookie:` field.
- Add the cookies you get to the 2nd line of `Main.py` and `Bing.py`(optional).
```
19. !!! You have to change to another Microsoft account and another cookies for every 300 questions. Otherwise, your cookies(account) will be limited for 24h and our Program cannot get the correct answer from Web.!!!
20. Write down your own questions to `Questions.txt` with correct form as [Description](#description) shows.
21. You'd better clean the content of the file `Results.txt`.
22. Run ```python Main.py``` and you will get the results from `Results.txt`.