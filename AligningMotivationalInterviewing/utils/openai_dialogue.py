import os
import openai
import json
import pandas as pd
from tqdm import tqdm

import sys
import time
import warnings
from timeout_decorator import timeout

class OpenaiSequencialDialogue:
    
    """
    Simple sequencial dialogue used to track user-system interaction.
    Does not consider backtracking and different histories
    
    Primary intended is for openai chat models
    """
    
    def __init__(
            self, 
            model = 'gpt-3.5-turbo', 
            system_message = None, 
            max_response_tokens=256,
            timeout_tolerance=15,
            stop = None
            ):
        
        # a list of chat history
        self.status = []
        
        # store model to use
        self.model = model
        
        if system_message is not None:
            self.status.append({"role": "system", 'content':system_message})

        self.max_response_tokens = max_response_tokens
        self.timeout_tolerance = timeout_tolerance
        self.stop = stop
            
        # to track cost
        # for most recent pricing see https://openai.com/pricing
        self.usages = []
        warnings.warn("Model pricing last updated: 11/07/2023")
        self.pricing_per_1k_tokens = {
            'gpt-3.5-turbo': {'prompt_tokens':0.0015, 'completion_tokens':0.002},
            'gpt-4': {'prompt_tokens':0.03, 'completion_tokens':0.06},
            'gpt-4-1106-preview': {'prompt_tokens':0.01, 'completion_tokens':0.03},
            'gpt-3.5-turbo-1106': {'prompt_tokens':0.001, 'completion_tokens':0.002},
            'gpt-3.5-turbo-instruct': {'prompt_tokens':0.0015, 'completion_tokens':0.002}
        }
    
    
    # def _complete_chat(self, messages, model):
    #     # call API until result is provided and then return it
    #     response = None
    #     received = False
    #     while not received:
    #         try:
    #             response = openai.ChatCompletion.create(
    #                 model = model, 
    #                 messages=messages,
    #                 max_tokens=self.max_response_tokens, 
    #                 temperature=0.0
    #             )
    #             received = True
    #         except:
    #             error = sys.exc_info()[0]
    #             if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
    #                 print(f"InvalidRequestError\nPrompt passed in:\n\n{str(messages)}\n\n")
    #                 assert False

    #             print("API error:", error)
    #             time.sleep(1)
                
    #     self.usages.append(response['usage'])
        
    #     return response

    def _complete_chat(self, messages, model):
        # call API until result is provided and then return it
        response = None
        received = False
        while not received:
            try:
                @timeout(self.timeout_tolerance)
                def send_request():
                    response = openai.ChatCompletion.create(
                        model = model, 
                        messages=messages,
                        max_tokens=self.max_response_tokens, 
                        temperature=0.0,
                        stop = self.stop
                    )
                    return response
                response = send_request()
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{str(messages)}\n\n")
                    assert False

                print("API error:", error)
                time.sleep(1)
                
        self.usages.append(response['usage'])
        
        return response
    
    def send_user_message(self, message):
        """
        send a new user message, along with the historical responses
        
        update status to include the model response
        """
        self.status.append({"role": "user", 'content':message})
        response = self._complete_chat(self.status, self.model)
        
        usage = response['usage']
        agent_message = response['choices'][0]['message']
        self.status.append(agent_message)
        
        return agent_message['content']

    def rewind(self):
        """
        rewind to a previous state, meaning pop 1 user-agent interaction (2 turns) from status.
        the cost remains.
        """
        out = []
        out.insert(0, self.status.pop())
        out.insert(0, self.status.pop())
        return out

    def _markdown(self):
        markdown_string = """"""
        for turn in self.status:
            markdown_string += '### ' + turn['role'].replace('assistant', 'assistant ('+self.model+')') + '\n'
            markdown_string += '```\n'
            markdown_string += turn['content'] + '\n'
            markdown_string += '```\n'
            markdown_string += '---\n'
        return markdown_string 

    def __str__(self):
        out = """Dialogue Object:\n"""
        out += self._markdown()
        return out
    
    def to_markdown(self, markdown_path):
            
        with open(markdown_path, 'w') as ofp:
            ofp.write(self._markdown())
            
    def cost(self):
        """
        return cost to api so far
        """
        # n thousand tokens
        nk_prompt_tokens = sum([i['prompt_tokens'] for i in self.usages]) / 1000
        nk_completion_tokens = sum([i['completion_tokens'] for i in self.usages]) / 1000
        
        prompt_cost = self.pricing_per_1k_tokens[self.model]['prompt_tokens'] * nk_prompt_tokens
        completion_cost = self.pricing_per_1k_tokens[self.model]['completion_tokens'] * nk_completion_tokens
        
        cost_dict = {
            'prompt':prompt_cost,
            'completion':completion_cost,
            'total':prompt_cost+completion_cost
        }
        
        return cost_dict

if __name__ == '__main__':
    dialogue = OpenaiSequencialDialogue()
    print(dialogue.send_user_message('who are you'))
    print(dialogue.send_user_message("""I don't understand... explain in more details"""))
    print(dialogue.cost())