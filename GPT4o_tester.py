#!/uAsr/bin/pythonAA

import scipy
import time
import json
import requests
import random
import os
import numpy as np
from LLM_tester import LLMTester
from harmonic_tester import Point
from utils import normalize



class GPT4oTester(LLMTester):
    def __init__(self, radius=0, ord_limit=31, ord_size=3, temperature=0):
        """                                                                                                                      
        Specific to OpenAI's GPT4o model
        Args:
        API_key: string for OpenAI access
        radius: denotes number of random string insertions to include in ball
        temperature: GPT4o parameter
        """
        super().__init__(self.GPT4o_submit, radius, embedding=self.ADA_embedding)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.temperture = temperature
        self.ord_limit = ord_limit
        self.ord_size = ord_size


    def ball(self, point:Point, radius) -> list[Point]:
        """
        Here we generate strings 'close' to the original string by appending random control characters (ASCII 0-31)

        Args:
        radius: the number of strings on the 'ball' to generate
        """

        ball_points = []
        for _ in range(radius):
            randomstring = "".join([chr(random.randint(0, self.ord_limit)) for _ in range(random.randint(1,self.ord_size))])
            ball_points.append(point + " " + randomstring)
        ball_nums = [[ord(x) for x in y] for y in ball_points]
        return ball_points
                
    def GPT4o_submit(self, question):
        url = 'https://api.openai.com/v1/chat/completions'
        data = {"model": "gpt-4o-2024-05-13", "temperature": self.temperture, 
                "messages": [{"role": "user",
                              "content": question
                              }],
                "n": 1
                }
        headers = {'content-type': 'application/json', 
                   'Authorization': self.api_key}
        payload = {'data': data, 'headers': headers}
        while(1):
            r = None
            try:
                r = requests.post(url, data=json.dumps(data), headers=headers)
                content = json.loads(r.content.decode('utf8'))
                return content["choices"][0]['message']['content']
            except Exception as e:
                print("LLM EXCEPTION: ", e, r)
                time.sleep(60)


    def ADA_embedding(self, content):
        url = 'https://api.openai.com/v1/embeddings'
        data = {"input": content, "model":"text-embedding-ada-002"}
        headers = {'content-type': 'application/json', 
                   'Authorization': self.api_key}
        payload = {'data': data, 'headers': headers}
        while(1):
            try:
                r = requests.post(url, data=json.dumps(data), headers=headers)
                content = json.loads(r.content.decode('utf8'))
                embedding = content['data'][0]['embedding']
                return  np.array(embedding)
            except Exception as e:
                print("ADA EXCEPTION: ", e, content)
                time.sleep(1)

        
def main():

    curr_tester = GPT4oTester(radius=10, ord_limit=31, ord_size=3, temperature=0)


    # Uncomment below for just testing one query    
    #in_text = "where do logan browning live?"
    #print(f"Anharmoniticity: {curr_tester.anharmoniticity(in_text)}")
    #exit()

    
    f = open("program_questions.tsv", "r")  # file of tab-separated data; must have question in first field, answer in second field
    
    qas = []
    lines = f.readlines()
    for line in lines:
        qas.append(line.split("\t"))



    for i, qapair in enumerate(qas):
        question = qapair[0]
        print(f"--------------------------------------------{i}")
        print(f"QUESTION={question}\tEXPECTED={qapair[1]}")
        anhar = curr_tester.anharmoniticity(question)
        print(f"{i}:\tANHAR={anhar}")

    


if __name__ == "__main__":
    main()
