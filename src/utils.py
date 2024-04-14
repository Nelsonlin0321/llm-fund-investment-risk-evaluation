import json
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage



class mistralUtils():
    def __init__(self,api_key) -> None:
        # self.api_key = api_key
        self.client = MistralClient(api_key=api_key)
        self.model = "mistral-small-latest"


    def extract_fund_names(self,text,retry=3):
        instruction =   """
                        You are working as a NER model in extract fund names from the text provided. The text will be provided.
                        Extract the fund names as a list or a empty list if not fund names detected.
                        """.strip()
        context = f"""Text is provided here: {text}. \n""" 
        question = """The output example: [{"name": "xxxx Fund"},{"name": "xxxx Fund"}]. ONLY Return list of names in JSON format
        """
        content = context+question

        for _ in range(retry):
            try:
                messages = [
                    ChatMessage(role="system", content=instruction),
                    ChatMessage(role="user", content=content),
                ]
                
                chat_response = self.client.chat(
                model=self.model,
                messages=messages)

                response = chat_response.choices[0].message.content

                results = json.loads(response)

                return results

            
            except Exception as e:
                continue
            
        return []



