import requests
import openai
import os
import pandas as pd
import time
import pyttsx3
import chatgpt

# def text_to_voice(s):
#     # Initialize the text-to-speech engine
#     engine = pyttsx3.init()

#     # Set properties (optional)
#     # You can change the voice, volume, rate, etc.
#     # engine.setProperty('voice', 'english')  # Uncomment and replace 'english' with a different voice if needed
#     # engine.setProperty('volume', 0.9)  # Adjust the volume (0.0 to 1.0)
#     # engine.setProperty('rate', 150)  # Adjust the speech rate (words per minute)

#     # Convert text language to formal language
#     sentence = convert_text_speech(s)

#     # Convert and play the text as speech
#     engine.say(sentence)
#     engine.runAndWait()

#     return sentence

def convert_text_speech(sentence1) :
    dictionary = {
        "BRB": "Be right back",
        "GTG": "Got to go",
        "OMG": "Oh my God",
        "TTYL": "Talk to you later",
        "BTW": "By the way",
        "IDK": "I don't know",
        "IMHO": "In my humble opinion",
        "FYI": "For your information",
        "ROFL": "Rolling on the floor laughing",
        "BFF": "Best friends forever",
        "ASAP": "As soon as possible",
        "NP": "No problem",
        "TMI": "Too much information",
        "TYT": "Take your time",
        "FTW": "For the win" ,
        "SMH": "Shaking my head",
        "IKR": "I know, right?",
        "TBT": "Throwback Thursday"
    }
    for key in dictionary :
        if key in sentence1 :
            index = sentence1.find(key)
            if index == 0 :
                sentence1 = sentence1.replace(sentence1[index:index+len(key)], dictionary[key])
            else :
                sentence1 = sentence1.replace(sentence1[index:index+len(key)], dictionary[key].casefold())
    return sentence1

def text_to_voice(s1) :
    openai.api_key = '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'
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": 'For the following sentence, convert any "Texting Language shortcuts" such as "TTYL" or "BRB" into their respective phrases ("Talk to you later", "Be right back"), make sure everything is gramatically correct, and add filler words such as "Uhhhh" or "Ummm" or "Hmmm" to make the text sound as realistic as possible, but the resulting sentence does not have to be formal, and try to keep the resulting sentence and the original sentence almost the same:\n' + s1}]
    )

    sentence = completion.choices[0].message.content

    url = "https://play.ht/api/v2/tts"

    payload = {
        "quality": "medium",
        "output_format": "mp3",
        "speed": 1,
        "sample_rate": 24000,
        "text": sentence,
        "voice": "larry"
    }
    headers = {
        "accept": "text/event-stream",
        "content-type": "application/json",
        "AUTHORIZATION": "Bearer b4d2686b7a564c009eb2ebd73c072899",
        "X-USER-ID": "G7Um0rcJBXeMXOpUGx7QLOEnpbg2"
    }

    response = requests.post(url, json=payload, headers=headers)

    lst = response.text.split('"')
    for i in lst :
        if "https://peregrine-results" in i :
            return i