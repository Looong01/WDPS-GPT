from keybert import KeyBERT
from transformers import BertTokenizer, BertModel
import torch, re
import numpy as np

def st_ex(answers, links):
    # Use Sentence-Transformers to extract entities
    kw_model = KeyBERT(model='./all-MiniLM-L6-v2')
    f = open('log/Entities_extracted_st.txt', 'w', encoding='utf-8')
    for i, answer in enumerate(answers):
        keywords = kw_model.extract_keywords(answer, keyphrase_ngram_range=(1, 1), stop_words=None)
        extract = keywords[0][0] + "," + keywords[1][0] + "," + keywords[2][0]
        print(extract)
        f.write(extract + '        ' + links[i].strip()+ '\n')
    f.close()
        

class bert_ex():
    # Use BERT to extract entities
    def __init__(self):
        # Load pre-trained model tokenizer (vocabulary)
        self.model = BertModel.from_pretrained('./bert-large-uncased').to('cuda')
        self.tokenizer = BertTokenizer.from_pretrained('./bert-large-uncased')
        
    @torch.no_grad()
    def encode_decode(self, sentence):
        # Encode and decode the sentence
        input_ids = self.tokenizer.encode(sentence, add_special_tokens=True, truncation=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to('cuda')
        outputs = self.model(input_ids)
        last_hidden_state = outputs.last_hidden_state
        probs = torch.nn.functional.softmax(last_hidden_state, dim=-1)
        probs = torch.max(probs, dim=-1)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return tokens, probs

    def extract_keywords(self, sentence):
        # Extract keywords from the sentence
        tokens, probs = self.encode_decode(sentence)
        probs = probs.cpu()
        probs_np = np.array(probs[0])
        sorted_indices = probs_np.argsort()[::-1]
        sorted_tokens = [tokens[i] for i in sorted_indices if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]
        return sorted_tokens
    
    def bert_ex(self, answers, links):
        # Extract entities from the answers
        f = open('log/Entities_extracted_bert.txt', 'w', encoding='utf-8')
        for i, answer in enumerate(answers):
            keywords = self.extract_keywords(answer)
            extract = keywords[0] + "," + keywords[1] + "," + keywords[2]
            print(extract)
            f.write(extract + '        ' + links[i].strip()+ '\n')
        f.close()

if __name__ == "__main__":
    with open("log/Answers_web.txt", "r", encoding="utf-8") as f:
        answers = f.readlines()
    with open("log/Links.txt", "r", encoding="utf-8") as f:
        links = f.readlines()
    st = st_ex(answers, links)
    bert_ex().bert_ex(answers, links)
