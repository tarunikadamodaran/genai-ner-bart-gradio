## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
Natural Language Processing (NLP) requires identifying important entities such as names, places, organizations, and dates from text. Manually detecting these entities is time-consuming and error-prone. Hence, the goal is to build an automated NER system using a pre-trained transformer model and a user-friendly web interface that:
Accepts text input from the user,
Detects named entities using a pre-trained model,
Highlights and classifies entities for visualization.

### DESIGN STEPS:
STEP 1: Import Libraries and Load Environment Variables
Import the necessary Python libraries: os, json, requests, gradio, and dotenv.

Load the .env file to access the Hugging Face API key and model endpoints securely.

STEP 2: Define Helper Function for API Calls
Create a get_completion() function that sends HTTP POST requests to the Hugging Face Inference API.

Include Authorization headers for secure access using the API token.

STEP 3: Define the Named Entity Recognition (NER) Function
Use the get_completion() function to send input text to the NER model endpoint.

Process the JSON response and extract named entities.

STEP 4: Token Merging (Optional Enhancement)
Implement a merge_tokens() helper function to merge subword tokens (e.g., “Cal” + “##ifornia” → “California”) for cleaner entity visualization.

STEP 5: Build Gradio Interface
Create a Gradio interface using gr.Interface() with:

### PROGRAM:
```
import os
import io
from IPython.display import Image, display, HTML
from PIL import Image
import base64 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']
API_URL = os.environ['HF_API_NER_BASE']

# Helper function
import requests, json

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
    return json.loads(response.content.decode("utf-8"))
def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        label = token["entity"]
        word = token["word"]

        clean_word = word.replace("##", "")

        if merged_tokens:
            prev = merged_tokens[-1]
            prev_label = prev["entity"]

            if label.endswith(prev_label.split("-")[-1]) and (label.startswith("I-") or label.startswith("B-")):
                prev["word"] += clean_word
                prev["end"] = token["end"]
                prev["score"] = (prev["score"] + token["score"]) / 2
                continue

            if word.startswith("##") and prev_label.startswith("B-"):
                prev["word"] += clean_word
                prev["end"] = token["end"]
                prev["score"] = (prev["score"] + token["score"]) / 2
                continue
            
            if prev["entity"].endswith("PER") and len(prev["word"]) == 1 and word.startswith("##"):
                prev["word"] += clean_word
                prev["end"] = token["end"]
                prev["score"] = (prev["score"] + token["score"]) / 2
                continue


        merged_tokens.append({
            "entity": label,
            "word": clean_word,
            "start": token["start"],
            "end": token["end"],
            "score": token["score"]
        })

    return merged_tokens


def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
    outputs=[gr.HighlightedText(label="Text with entities")],
    title="NER with Fine-Tuned BART ",
    description="Highlights people, organizations, and locations in your text.",
    allow_flagging="never",
    examples=[
        ["My name is Tarunika, I'm building DeepLearningAI and I live in Chennai"],
        ["Elon Musk founded SpaceX in the United States"]
    ]
)

demo.launch(share=True, server_port=int(os.environ.get("PORT4", 7860)))
```


### OUTPUT:
<img width="1053" height="773" alt="image" src="https://github.com/user-attachments/assets/61e426d7-3f68-4a0f-8b8f-87a625042b46" />

### RESULT:
The Named Entity Recognition (NER) prototype was successfully developed using the fine-tuned BERT model (dslim/bert-base-NER) and deployed through the Gradio interface. The system efficiently identifies and highlights entities such as names, locations, and organizations from user-provided text input.
