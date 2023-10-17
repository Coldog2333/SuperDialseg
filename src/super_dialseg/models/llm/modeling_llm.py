import time

import openai
import os
from getpass import getpass
from ...modeling_utils import BaseSegmenter


class OpenAISegmenter(BaseSegmenter):
    def __init__(self, model, openai_key=None):
        super(OpenAISegmenter, self).__init__()
        self.model = model

        # openai setting
        if openai_key is not None:
            openai.api_key = openai_key
        elif os.getenv("OPENAI_API_KEY") is not None:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            openai.api_key = getpass(
                "Paste your OpenAI key from: https://platform.openai.com/account/api-keys\n")
        assert openai.api_key.startswith("sk-"), "This doesn't look like a valid OpenAI API key"
        print("OpenAI API key configured")

    def create_prompt(self, sample):
        raise NotImplementedError

    def parse_openai_response(self, response, inputs):
        raise NotImplementedError

    def forward(self, inputs, n_try=60, temperature=0., max_tokens=512):
        "The token count of your prompt plus max_tokens cannot exceed the model's context length."
        utterances = inputs['utterances']
        prompt = self.create_prompt(sample={'utterances': utterances})

        while n_try > 0:
            n_try -= 1
            try:
                if self.model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613',
                                  'gpt-4-0314', 'gpt-4-0613']:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                elif self.model in ['text-davinci-003']:
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                else:
                    raise NotImplementedError(f"{self.model} has not been supported yet.")

                return self.parse_openai_response(response, inputs)

            except NotImplementedError as e:
                print(e)
                return None

            except Exception as e:
                print(e)
                time.sleep(1)
                continue


class ChatGPTSegmenter(OpenAISegmenter):
    def __init__(self, model='gpt-3.5-turbo-0301', openai_key=None):
        super(ChatGPTSegmenter, self).__init__(model=model, openai_key=openai_key)

    def create_prompt(self, sample):
        content = (
            "Dialogue Segmentation aims to segment a dialogue D = {U1, U2, ..., Un} into several parts according to their discussing topics.\n"
            "Please help me to segment the following dialogue: \n"
        )

        for i, utterance in enumerate(sample['utterances']):
            content += f"U{i+1}: {utterance}\n"

        ### output format
        content += "\nOutput format: Part i: Ua-Ub\n"
        content += "\n=====\nOutput example:\nPart 1: U1-U4\nPart 2: U5-U6\n=====\n"

        prompt = [
            {
                'role': 'system',
                'content': "You are a helpful assistance to segment give dialogues.\nPlease follow the output format.\nDO NOT explain."
            },
            {
                'role': 'user',
                'content': content
            }
        ]
        return prompt

    def parse_openai_response(self, response, inputs):
        result = response['choices'][0]['message']['content'].lstrip().rstrip()
        ## result post-processing
        results = result.split('\n')
        end_indices = []
        for line in results:
            end_index = int(line.strip().split('U')[-1]) - 1
            end_indices.append(end_index)

        predictions = [0] * len(inputs["utterances"])
        for end_index in end_indices:
            predictions[end_index] = 1

        predictions[-1] = 0

        return predictions


class InstructGPTSegmenter(OpenAISegmenter):
    def __init__(self, model='text-davinci-003', openai_key=None):
        super(InstructGPTSegmenter, self).__init__(model=model, openai_key=openai_key)

    def create_prompt(self, sample):
        # making prompt
        ## define task
        prompt = "Dialogue Segmentation aims to segment a dialogue D = {U1, U2, ..., Un} into several parts according to their discussing topics.\n"

        ## input sentences
        prompt += "\nHere is the given dialogue D:\n"
        input_counter = 0
        for i, utterance in enumerate(sample['utterances']):
            prompt += f"U{i+1}: {utterance}\n"
            input_counter += len(utterance.split(' '))

        ## output prompt
        prompt += "\nSegment D into several parts according to their discussing topics.\n"

        ### output format
        prompt += "\nOutput format: Part i: Ua-Ub\n"
        prompt += "\n=====\nOutput example:\nPart 1: U1-U4\nPart 2: U5-U6\n=====\n"

        prompt += "Output of the dialogue segmentation task: \n"

        return prompt

    def parse_openai_response(self, response, inputs):
        result = response['choices'][0]['text'].lstrip().rstrip()
        ## result post-processing
        results = result.split('\n')
        end_indices = []
        for line in results:
            end_index = int(line.strip().split('U')[-1]) - 1
            end_indices.append(end_index)

        predictions = [0] * len(inputs["utterances"])
        for end_index in end_indices:
            predictions[end_index] = 1

        predictions[-1] = 0

        return predictions


if __name__ == '__main__':
    segmenter = ChatGPTSegmenter(model='gpt-3.5-turbo-0301')
    predictions = segmenter.forward(inputs={'utterances': ['hello', 'hi', 'how are you', 'yes, it is fine.']})
    print(predictions)
