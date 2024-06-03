import time

import boto3
import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
import InvokeAgent as agenthelper
import json
import pandas as pd

st.title("SmartHome Assistant Powered by Bedrock")

state = st.session_state

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

@st.cache_resource
def load_llm():
    llm = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2")
    llm.model_kwargs = {"temperature": 0.7, "max_tokens_to_sample": 2048}

    model = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

    return model

# Function to parse and format response
def format_response(response_body):
    try:
        # Try to load the response as JSON
        data = json.loads(response_body)
        # If it's a list, convert it to a DataFrame for better visualization
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return response_body
    except json.JSONDecodeError:
        # If response is not JSON, return as is
        return response_body


model = load_llm()
result = None

if 'text_received' not in state:
    state.text_received = []

c1, c2 = st.columns(2)
with c1:
    st.write("Tekan untuk bicara:")
with c2:
    text = speech_to_text(language='id', use_container_width=True, just_once=True, key='STT')
    if text is not None:
        # call lambda to bedrock agent
        event = {
            "sessionId": "MYSESSION1211",
            "question": text
        }
        response = agenthelper.lambda_handler(event,None)
        try:
            # Parse the JSON string
            if response and 'body' in response and response['body']:
                response_data = json.loads(response['body'])
                print("TRACE & RESPONSE DATA ->  ", response_data)
            else:
                print("Invalid or empty response received")
        except json.JSONDecodeError as e:
            print("JSON decoding error:", e)
            response_data = None 
        
        try:
            # Extract the response and trace data
            all_data = format_response(response_data['response'])
            the_response = response_data['trace_data']
        except:
            all_data = "..." 
            the_response = "Apologies, but an error occurred. Please rerun the application"

        result = the_response
            # result = model.predict(input=text)

if text:
    state.text_received.append("Human:" + text)

if result:
    state.text_received.append("AI:" + result)

for str in state.text_received:
    st.text(str)

# st.write("Record your voice, and play the recorded audio:")
# audio = mic_recorder(start_prompt="⏺️", stop_prompt="⏹️", key='recorder')

# if audio:
#     st.audio(audio['bytes'])
