import requests
import openai
import os
import pandas as pd
import time
import pyttsx3
import chatgpt

def text_to_voice(s1) :
    openai.api_key = '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'
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": 'For everything I send in this chat, only fix any grammatical mistakes, punctuation.  If a word makes sense by itself, leave it. However, if there is a spelling error anywhere, you may change the words then: \n' + s1}],
        temperature = 0
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
            return [i, sentence]
