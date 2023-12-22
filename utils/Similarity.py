from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch, os
import numpy as np

def st_sim(sentences1, sentences2):
    # Use Sentence-Transformers to compare the similarity of the answers
    path = os.path.join(os.path.abspath(os.getcwd()))
    model = SentenceTransformer(os.path.join(path, 'all-MiniLM-L6-v2'))
    checks = []
    for i in range(len(sentences1)):
        s1 = sentences1[i].strip()
        s2 = sentences2[i].strip()
        embeddings = model.encode([s1, s2])
        sim = util.cos_sim(embeddings[0], embeddings[1])
        check = sim.tolist()[0][0]
        checks.append(check)
    return checks

def bert_sim(sentences1, sentences2):
    # Use BERT to compare the similarity of the answers
    path = os.path.join(os.path.abspath(os.getcwd()))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, 'bert-large-uncased'))
    model = AutoModel.from_pretrained(os.path.join(path, 'bert-large-uncased')).to('cuda')
    checks = []
    for i in range(len(sentences1)):
        s1 = sentences1[i].strip()
        s2 = sentences2[i].strip()
        inputs = tokenizer([s1, s2], return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        check = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        checks.append(check)
    return checks

def sim(sentences1, sentences2, prob):
    # Compare the similarity of the answers
    path = os.path.join(os.path.abspath(os.getcwd()))
    checks = st_sim(sentences1, sentences2)
    n_true_st = 0
    f = open(os.path.join(path, 'log', "Check_st.txt"), "w", encoding="utf-8")
    for check in checks:
        # print("{0:.6f}".format(check))
        if check >= prob:
            f.write("Correct" + "\n")
            n_true_st += 1
        else:
            f.write("Incorrect" + "\n")
    f.close()
    checks = bert_sim(sentences1, sentences2)
    n_true_bert = 0
    f = open(os.path.join(path, 'log', "Check_bert.txt"), "w", encoding="utf-8")
    for check in checks:
        # print("{0:.6f}".format(check))
        if check >= 0.9:
            f.write("Correct" + "\n")
            n_true_bert += 1
        else:
            f.write("Incorrect" + "\n")
    f.close()
    return n_true_st, n_true_bert

if __name__ == "__main__":
    path = os.path.join(os.path.abspath(os.getcwd()))
    with open(os.path.join(path, 'log', "Answers_llm.txt"), "r", encoding="utf-8") as f:
        sentences1 = f.readlines()
    with open(os.path.join(path, 'log', "Answers_web.txt"), "r", encoding="utf-8") as f:
        sentences2 = f.readlines()
    sim(sentences1, sentences2, 0.7)