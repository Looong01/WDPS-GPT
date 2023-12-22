import re, asyncio, os
os.environ["BING_COOKIES"] = "<Your own cookies>" ## Add your own cookies here

from utils.Bing import bing ## import the function bing from utils
from utils.LLaMa import llama2
from utils.Similarity import sim
from utils.Keywords import st_ex

def input(file_results, question, i):
    ## write the questions into Results.txt
    question = question.strip().split('        ')[1]
    file_results.write(str(i) + '        ' + 'Input"' + question + '"\n')

def answer(file_results, answer, i):
    ## write the answers into Results.txt
    answer = answer.strip()
    file_results.write(str(i) + '        R"' + answer + '"\n')

def extracted_answer(file_results, answer, link, i):
    ## write the extracted answers into Results.txt
    if re.search('Yes', answer):
        file_results.write(str(i) + '        A"yes"\n') 
    else:
        file_results.write(str(i) + '        A"no"\n')

def correctness(file_results, check_st, check_bert, i):
    ## write the correctness of the answers into Results.txt
    check_st = check_st.strip()
    check_bert = check_bert.strip()
    file_results.write(str(i) + '        C"' + check_st + '"\n')
    # file_results.write('Correctness of the answer(BERT): "' + check_bert + '"\n')

def entities_extracted(file_results, line, i):
    ## write the entities extracted into Results.txt
    line = line.strip()
    file_results.write(str(i) + '        E"' + line + '"\n\n')
    
def main():
    ## run all the processes and write the results into Results.txt
    if not os.path.exists("log"): ## create a folder to save the log files
        os.makedirs("log")
    ## ask questions
    with open("Questions.txt", "r", encoding="utf-8") as f:
        questions = f.readlines()
    asyncio.run(bing(questions)) ## Run the Bing process
    with open("Questions.txt", "r") as f:
        questions = f.readlines()
    llama2(questions) ## Run the LLaMa model
    
    ## extract entities
    with open("log/Answers_web.txt", "r", encoding="utf-8") as f:
        answers = f.readlines()
    with open("log/Links_web.txt", "r", encoding="utf-8") as f:
        links = f.readlines()
    st_ex(answers, links) ## Extract entities using Sentence-Transformers
    # bert_ex().bert_ex(answers, links)
    
    ## check correctness
    with open("log/Answers_llm.txt", "r", encoding="utf-8") as f:
        sentences1 = f.readlines()
    with open("log/Answers_web.txt", "r", encoding="utf-8") as f:
        sentences2 = f.readlines()
        
    ## Compare the similarity of the answers, and get the number of correct answers
    n_true_st, n_true_bert = sim(sentences1, sentences2, 0.7) 
    print("Number of Correct(SentenceTransformer): " + str(n_true_st))
    print("Number of Correct(BERT): " + str(n_true_bert))

    file_results = open("Results.txt", "w", encoding="utf-8")
    file_entities_extracted_st = open("log/Entities_extracted_st.txt", "r", encoding="utf-8")
    # file_entities_extracted_bert = open("log/Entities_extracted_bert.txt", "r", encoding="utf-8")
    file_check_st = open("log/Check_st.txt", "r", encoding="utf-8")
    file_check_bert = open("log/Check_bert.txt", "r", encoding="utf-8")
    check_sts = file_check_st.readlines()
    check_berts = file_check_bert.readlines()
    entities = file_entities_extracted_st.readlines()
    
    ## write the results into Results.txt
    for i in range(len(questions)):
        # input(file_results, questions[i], i+1)
        answer(file_results, answers[i], i+1)
        extracted_answer(file_results, answers[i], links[i], i+1)
        correctness(file_results, check_sts[i], check_berts[i], i+1)
        entities_extracted(file_results, entities[i], i+1)
    file_results.close()
    file_entities_extracted_st.close()
    # file_entities_extracted_bert.close()
    file_check_st.close()
    file_check_bert.close()    
    
if __name__ == "__main__":
    main()