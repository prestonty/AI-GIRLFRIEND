# from huggingface_hub import InferenceClient
# client = InferenceClient(model="KoboldAI/GPT-J-6B-Janeway")
# # model="KoboldAI/GPT-J-6B-Janeway"
# # output = client.conversational("Can you be my waifu?")

# # print(output)


# output = client.conversational("Hi, who are you?")
# output
# {'generated_text': 'I am the one who knocks.', 'conversation': {'generated_responses': ['I am the one who knocks.'], 'past_user_inputs': ['Hi, who are you?']}, 'warnings': ['Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.']}
# client.conversational(
#     "Wow, that's scary!",
#     generated_responses=output["conversation"]["generated_responses"],
#     past_user_inputs=output["conversation"]["past_user_inputs"],
# )



import requests
import os

# .env
from dotenv import load_dotenv
load_dotenv()

HUGGING_FACE_READ_KEY=os.getenv('HUGGING_FACE_READ_KEY')

# https://api-inference.huggingface.co/models/KoboldAI/GPT-J-6B-Janeway

# https://api-inference.huggingface.co/models/gpt2

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {HUGGING_FACE_READ_KEY}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
data = query("My favourite food is")
print(data)



# from transformers import pipeline

# generator = pipeline('text-generation', model="C:/Users/prest/OneDrive - University of Waterloo/Documents/Computer Programming/Project Explore/sobi/models/janeway")
# text = generator("Honey, what do you want to do this evening? A bath, dinner, or... Me?.", do_sample=True, min_length=50)
# print(text)

# from transformers import pipeline
# generator = pipeline('text-generation', model='KoboldAI/GPT-J-6B-Janeway')
# generator("Welcome Captain Janeway, I apologize for the delay.", do_sample=True, min_length=50)

# [{'generated_text': 'Welcome Captain Janeway, I apologize for the delay."\nIts all right," Janeway said. "Im certain that youre doing your best to keep me informed of what\'s going on."'}]



# i downloaded the model in my local repo
# i should move it to my docker container
# i want to run the code above by accessing the model on my local system

# https://huggingface.co/docs/hub/spaces-sdks-docker
# follow these steps in docker file, THESE ARE YOUR NEXT STEPS!!!!