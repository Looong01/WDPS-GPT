import re, asyncio, os
os.environ["BING_COOKIES"] = "<Your own cookies>"

from Bing import bing
from llama import llama2
from Similarity import sim
from Keywords import st_ex

def input(file_results, question, i):
    question = question.strip().split('        ')[1]
    file_results.write(str(i) + '        ' + 'Input"' + question + '"\n')

def answer(file_results, answer, i):
    answer = answer.strip()
    file_results.write(str(i) + '        R"' + answer + '"\n')

def extracted_answer(file_results, answer, link, i):
    if i <100:
        if re.search('Yes', answer):
            file_results.write(str(i) + '        A"yes"\n') 
        else:
            file_results.write(str(i) + '        A"no"\n')
    else:
        file_results.write(str(i) + '        A"' + link.strip() + '"\n')

def correctness(file_results, check_st, check_bert, i):
    check_st = check_st.strip()
    check_bert = check_bert.strip()
    file_results.write(str(i) + '        C"' + check_st + '"\n')
    # file_results.write('Correctness of the answer(BERT): "' + check_bert + '"\n')

def entities_extracted(file_results, line, i):
    line = line.strip()
    file_results.write(str(i) + '        E"' + line + '"\n\n')
    
def main():
    # ask questions
    with open("Questions.txt", "r", encoding="utf-8") as f:
        questions = f.readlines()
    asyncio.run(bing(questions))
    with open("Questions.txt", "r") as f:
        questions = f.readlines()
    llama2(questions)
    
    # extract entities
    with open("log/Answers_web.txt", "r", encoding="utf-8") as f:
        answers = f.readlines()
    with open("log/Links_web.txt", "r", encoding="utf-8") as f:
        links = f.readlines()
    st_ex(answers, links)
    # bert_ex().bert_ex(answers, links)
    
    # check correctness
    with open("log/Answers_llm.txt", "r", encoding="utf-8") as f:
        sentences1 = f.readlines()
    with open("log/Answers_web.txt", "r", encoding="utf-8") as f:
        sentences2 = f.readlines()
    n_true_st, n_true_bert = sim(sentences1, sentences2, 0.85)
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
    
    # file_results.write("Number of True(SentenceTransformer): " + str(n_true_st) + "/" + str(len(answers))\n")
    # file_results.write("Number of True(BERT): " + str(n_true_bert) + "/" + str(len(answers))\n\n")
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