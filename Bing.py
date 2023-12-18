import os, re
os.environ["BING_COOKIES"] = "<Your own cookies>"

import asyncio
from sydney import SydneyClient
from sydney.exceptions import *

def answer(response):
    lines = response.split('\n')
    with open('log/Answers_web.txt', 'a', encoding='utf-8') as file:
        file.write(' '.join(lines))
        file.write('\n')

def link(response):
    links = re.findall('https://[^\s,]+', response)
    with open('log/Links_web.txt', 'a', encoding='utf-8') as file:
        file.write('; '.join(links))
        file.write('\n')

def remove_https_lines(response):
    lines = response.split('\n')
    new_response = '\n'.join(line for line in lines if not re.search('https://', line))
    return new_response

async def ask(sydney, questions, i, j):
    while True:
        index = j * 15 + i
        try:
            response = await sydney.ask(str("Search on the wikipedia and Internet, " + questions[index].strip().split('        ')[1]), citations=True)  
            break
        except (KeyError, ValueError, TypeError, AttributeError, IndexError, asyncio.TimeoutError, RuntimeError, ConnectionTimeoutException, CreateConversationException, GetConversationsException, NoConnectionException, NoResponseException, ThrottledRequestException) as e:
            print(e)
            continue
    print(response)
    link(response)
    answer(remove_https_lines(response))

async def bing(questions):
    # ask questions
    num = len(questions)
    for j in range(0, num // 15 + 1):
        async with SydneyClient(style="precise") as sydney:
            if j < num // 15:
                for i in range(0, 15):
                    await ask(sydney, questions, i, j)
            else:
                for i in range(0, num % 15):
                    await ask(sydney, questions, i, j)

if __name__ == "__main__":
    with open("Questions.txt", "r") as f:
        questions = f.readlines()
    asyncio.run(bing(questions))
