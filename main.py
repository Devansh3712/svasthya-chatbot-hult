import asyncio
from io import BytesIO
from typing import Literal

import streamlit as st
from googletrans import Translator
from groq import Groq
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


def llm_transcribe(audio: BytesIO) -> str:
    client = Groq(api_key=st.secrets.GROQ_API_KEY)
    translation = client.audio.transcriptions.create(
        file=("audio_input.wav", audio),
        model="whisper-large-v3",
    )
    return translation.text


def llm_response(message: str) -> str:
    client = Groq(api_key=st.secrets.GROQ_API_KEY)
    with open("prompt.txt") as infile:
        prompt = infile.read()

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        model="llama-3.3-70b-versatile",
        stream=False,
    )
    data = response.choices[0].message.content
    translation = asyncio.run(
        Translator().translate(data, src="en", dest="ja"))
    return translation.text


def llm_stream_response(message: str):
    client = Groq(api_key=st.secrets.GROQ_API_KEY)
    with open("prompt.txt") as infile:
        prompt = infile.read()

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        model="llama-3.3-70b-versatile",
        stream=True,
    )
    for chunk in response:
        data = chunk.choices[0].delta.content
        if data:
            yield data


st.title("svāsthya chatbot")
st.sidebar.selectbox(
    "Language",
    ("English", "Japanese"),
    key="language",
)
audio = st.sidebar.checkbox("Use audio")

if "messages" not in st.session_state:
    st.session_state.messages: list[Message] = []

for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)

prompt = None
if audio:
    audio_value = st.audio_input("Record your query:")
    if audio_value:
        prompt = llm_transcribe(audio_value)
else:
    prompt = st.chat_input("Enter your query")

# TODO:
# If language is Japanese, append the translated prompt
if prompt:
    st.session_state.messages.append(Message(role="user", content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.language == "Japanese":
            translation = asyncio.run(
                Translator().translate(prompt, src="ja", dest="en")
            )
            response = llm_response(translation.text)
            st.write(response)
            st.session_state.messages.append(
                Message(role="assistant", content=response)
            )
        else:
            stream = llm_stream_response(prompt)
            response = st.write_stream(stream)
            st.session_state.messages.append(
                Message(role="assistant", content=response)
            )
