from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch, os
import numpy as np

def st_sim(sentences1, sentences2, device):
    ## Use Sentence-Transformers to compare the similarity of the answers
    path = os.path.join(os.path.abspath(os.getcwd()))
    model = SentenceTransformer(os.path.join(path, 'all-MiniLM-L6-v2')) ## Load Sentence-Transformers model to GPU Video RAM
    checks = []
    for i in range(len(sentences1)):
        s1 = sentences1[i].strip()
        s2 = sentences2[i].strip()
        embeddings = model.encode([s1, s2])
        sim = util.cos_sim(embeddings[0], embeddings[1]) ## Calculate the cosine similarity of the embeddings
        check = sim.tolist()[0][0]
        checks.append(check)
    return checks

def bert_sim(sentences1, sentences2, device):
    ## Use BERT to compare the similarity of the answers
    path = os.path.join(os.path.abspath(os.getcwd()))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, 'bert-base-uncased')) ## Load BERT tokenizer to GPU RAM
    model = AutoModel.from_pretrained(os.path.join(path, 'bert-base-uncased')).to(device) ## Load BERT model to GPU Video RAM
    checks = []
    for i in range(len(sentences1)):
        s1 = sentences1[i].strip()
        s2 = sentences2[i].strip()
        inputs = tokenizer([s1, s2], return_tensors='pt', padding=True, truncation=True, max_length=512).to(device) ## Tokenize the sentences and load them to GPU Video RAM
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() ## Copy the embeddings to CPU RAM to do post-processing
        check = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])) ## Calculate the cosine similarity of the embeddings
        checks.append(check)
    return checks

def sim(sentences1, sentences2, prob, device):
    ## Compare the similarity of the answers
    path = os.path.join(os.path.abspath(os.getcwd()))
    checks = st_sim(sentences1, sentences2, device) ## Use Sentence-Transformers to compare the similarity of the answers
    n_true_st = 0
    f = open(os.path.join(path, 'log', "Check_st.txt"), "w", encoding="utf-8")
    for check in checks:
        # print("{0:.6f}".format(check))
        if check >= prob: ## If the similarity is greater than the threshold, we think the answer is correct
            f.write("Correct" + "\n")
            n_true_st += 1
        else:
            f.write("Incorrect" + "\n")
    f.close()
    checks = bert_sim(sentences1, sentences2, device) ## Use BERT to compare the similarity of the answers
    n_true_bert = 0
    f = open(os.path.join(path, 'log', "Check_bert.txt"), "w", encoding="utf-8")
    for check in checks:
        # print("{0:.6f}".format(check))
        if check >= prob:
            f.write("Correct" + "\n")
            n_true_bert += 1
        else:
            f.write("Incorrect" + "\n")
    f.close()
    return n_true_st, n_true_bert ## Return the number of correct answers

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("log"): ## create a folder to save the log files
        os.makedirs("log")
    path = os.path.join(os.path.abspath(os.getcwd()))
    with open(os.path.join(path, 'log', "Answers_llm.txt"), "r", encoding="utf-8") as f:
        sentences1 = f.readlines()
    with open(os.path.join(path, 'log', "Answers_web.txt"), "r", encoding="utf-8") as f:
        sentences2 = f.readlines()
    sim(sentences1, sentences2, 0.7, device)