from keybert import KeyBERT
from transformers import BertTokenizer, BertModel
import torch, re, os
import numpy as np

def st_ex(answers, links, device):
    ## Use Sentence-Transformers to extract entities
    path = os.path.join(os.path.abspath(os.getcwd()))
    kw_model = KeyBERT(os.path.join(path, 'all-MiniLM-L6-v2')) ## Load Sentence-Transformers model to GPU Video RAM
    f = open(os.path.join(path, 'log', 'Entities_extracted_st.txt'), 'w', encoding='utf-8')
    for i, answer in enumerate(answers):
        keywords = kw_model.extract_keywords(answer, keyphrase_ngram_range=(1, 1), stop_words=None) ## extract entities
        extract = keywords[0][0] + "," + keywords[1][0] + "," + keywords[2][0]
        print(extract)
        f.write(extract + '        ' + links[i].strip()+ '\n') ## save output into the file
    f.close()

class bert_ex(torch.DeviceObjType):
    ## Use BERT to extract entities
    def __init__(self):
        ## Load pre-trained model tokenizer (vocabulary)
        path = os.path.join(os.path.abspath(os.getcwd()))
        self.device = device
        self.model = BertModel.from_pretrained(os.path.join(path, 'bert-base-uncased')).to(device) ## Load BERT model to GPU Video RAM
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(path, 'bert-base-uncased')) ## Load BERT tokenizer to GPU RAM
        
    @torch.no_grad()
    def encode_decode(self, sentence):
        ## Encode and decode the sentence
        input_ids = self.tokenizer.encode(sentence, add_special_tokens=True, truncation=True) ## Encode the sentence (Tokenize the sentence and add [CLS] and [SEP] tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        outputs = self.model(input_ids)
        last_hidden_state = outputs.last_hidden_state
        probs = torch.nn.functional.softmax(last_hidden_state, dim=-1)
        probs = torch.max(probs, dim=-1)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0]) ## Decode the sentence
        return tokens, probs

    def extract_keywords(self, sentence):
        ## Extract keywords from the sentence
        tokens, probs = self.encode_decode(sentence)
        probs = probs.cpu()
        probs_np = np.array(probs[0])
        sorted_indices = probs_np.argsort()[::-1]
        sorted_tokens = [tokens[i] for i in sorted_indices if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
        return sorted_tokens
    
    def bert_ex(self, answers, links):
        ## Extract entities from the answers
        path = os.path.join(os.path.abspath(os.getcwd()))
        f = open(os.path.join(path, 'log', 'Entities_extracted_bert.txt'), 'w', encoding='utf-8')
        for i, answer in enumerate(answers):
            keywords = self.extract_keywords(answer)
            extract = keywords[0] + "," + keywords[1] + "," + keywords[2]
            print(extract)
            f.write(extract + '        ' + links[i].strip()+ '\n') ## save output into the file
        f.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("log"): ## create a folder to save the log files
        os.makedirs("log")
    path = os.path.join(os.path.abspath(os.getcwd()))
    with open(os.path.join(path, 'log', "Answers_web.txt"), "r", encoding="utf-8") as f:
        answers = f.readlines()
    with open(os.path.join(path, 'log', "Links.txt"), "r", encoding="utf-8") as f:
        links = f.readlines()
    st = st_ex(answers, links, device)
    bert_ex(device).bert_ex(answers, links)
