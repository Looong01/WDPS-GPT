import os, re
os.environ["BING_COOKIES"] = "<Your own cookies>"

import asyncio
from sydney import SydneyClient
from sydney.exceptions import *

def answer(response):
    ## write the answers into Answers_web.txt
    path = os.path.join(os.path.abspath(os.getcwd())) ## Get Absolute path
    lines = response.split('\n')
    with open(os.path.join(path, 'log', 'Answers_web.txt'), 'a', encoding='utf-8') as file:
        file.write(' '.join(lines))
        file.write('\n')

def link(response):
    ## write the links into Links_web.txt
    path = os.path.join(os.path.abspath(os.getcwd()))
    links = re.findall('https://[^\s,]+', response) ## Find all the line starting with https://
    with open(os.path.join(path, 'log', 'Links_web.txt'), 'a', encoding='utf-8') as file:
        file.write('; '.join(links))
        file.write('\n')

def remove_https_lines(response):
    ## remove the lines with https://, remain the answers which are got from the Internet
    lines = response.split('\n')
    new_response = '\n'.join(line for line in lines if not re.search('https://', line)) ## Add all the lines without https:// words as the Answers we get from the Web
    return new_response

async def ask(sydney, questions, i, j):
    ## ask questions
    while True:
        index = j * 15 + i
        try:
            response = await sydney.ask(str("Search on the wikipedia and Internet, " + questions[index].strip().split('        ')[1]), citations=True)  
            break
        except (KeyError, ValueError, TypeError, AttributeError, IndexError, asyncio.TimeoutError, RuntimeError, ConnectionTimeoutException, CreateConversationException, GetConversationsException, NoConnectionException, NoResponseException, ThrottledRequestException) as e:
            print(e)
            continue ## If the Web returns any error or python Exception, try it again.
    print(response)
    link(response)
    answer(remove_https_lines(response))

async def bing(questions):
    ## create Bing sessions and ask questions
    num = len(questions)
    for j in range(0, num // 15 + 1): ## We have to change to create a new Bing session for every 15 questions for the limitation of Bing.
        async with SydneyClient(style="precise") as sydney:
            if j < num // 15:
                for i in range(0, 15):
                    await ask(sydney, questions, i, j)
            else:
                for i in range(0, num % 15):
                    await ask(sydney, questions, i, j)

if __name__ == "__main__":
    path = os.path.join(os.path.abspath(os.getcwd()))
    if not os.path.exists(os.path.join(path, 'log')):
        os.makedirs(os.path.join(path, 'log'))
    with open(os.path.join(path, 'log', "Questions.txt"), "r") as f:
        questions = f.readlines()
    asyncio.run(bing(questions))
