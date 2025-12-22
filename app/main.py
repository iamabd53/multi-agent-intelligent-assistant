import os
from dotenv import load_dotenv
load_dotenv() 
api_key = os.environ.get("OPENROUTER_API_KEY")

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
#from langchain.chains import LLMChain


def route_query(query_string):

    template = """You are an expert routing assistant. Your task is to classify a user's query into one of three categories based on the user's intent. Respond with only the category name and nothing else.
    The categories are:
    direct: For questions about static, general knowledge, facts, or concepts that are timeless.
    web_search: For questions about recent events, current information, or topics that are likely to change over time.
    multi_modal: For any query that explicitly asks to analyze, summarize, describe, or find something within a provided file, document, or image.
    --- Examples ---
    Query: "Explain the difference between a planet and a star."
    Category: direct
    Query: "What are the top news headlines today?"
    Category: web_search
    Query: "Summarize the attached research paper."
    Category: multi_modal
    Query: "Who wrote the play 'Romeo and Juliet'?"
    Category: direct
    Query: "Describe the main subject in this image."
    Category: multi_modal
    Query: "Who won the last FIFA World Cup?"
    Category: web_search
    Query: "Find all mentions of 'Q3 earnings' in this report."
    Category: multi_modal
    Query: "What is photosynthesis?"
    Category: direct
    Query: "What is the current weather in New York City?"
    Category: web_search
    In other words anwer in one token which is only one of these three (direct, web_search, multi_model) after analyzing the user's query

    --- End of Examples ---
    Query: "{query_string}"
    Category:
    """

    prompt = PromptTemplate(template = template, input_variables= ["query_string"])
    formatted_prompt = prompt.format(query_string = query_string)

    llm = ChatOpenAI(api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1-0528:free")

    return llm.invoke(formatted_prompt).content

#route_query("what is present in this picture")

def direct_query_agent(query_string):
    print('direct_query agent initiated')
    template = """You are an expert assistant. Answer accurately and concisely
    Query:"{query_string}"
    Answer: 
    """
    prompt = PromptTemplate(template = template, input_variables= ["query_string"])
    formatted_prompt = prompt.format(query_string= query_string)
    
    llm = ChatOpenAI(api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1-0528:free")

    return llm.invoke(formatted_prompt).content


def web_search_agent(query_string):
    print('web_search_agent initiated')
    from ddgs import DDGS
    result = DDGS().text(query_string, max_results=5)

    title_body= []
    for i in result:
        title_body.append(i['title'])
        title_body.append(i['body'])
    useful_info = "".join(title_body)

    template = """Here are some search results about {query_string}. Use them to answer the question"
    Answer: {useful_info}
    """
    prompt = PromptTemplate(template = template, input_variables= ["query_string", "useful_info"])
    formatted_prompt = prompt.format(query_string= query_string, useful_info = useful_info)
    
    llm = ChatOpenAI(api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-r1-0528:free")

    return llm.invoke(formatted_prompt).content

#print(web_search_agent("what is the current weather of Tokyo"))


import pandas as pd
import os
from langchain_core.messages import HumanMessage
from PyPDF2 import PdfReader
import base64
def multi_model_agent(file_path ,query_string):
    print('multi_model_agent initiated') 
    if not os.path.exists(file_path): 
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext ==".txt":
        with open(file_path,"r", encoding="utf-8") as f:
            content = f.read() 

            template = """You are an expert assistant. Answer accurately and concisely, you will understand Query and answer from Content
            Query:"{query_string}",
            Content:{content}
            Answer: 
            """
            prompt = PromptTemplate(template = template, input_variables= ["query_string", "content"])
            formatted_prompt = prompt.format(query_string= query_string, content= content)
            
            llm = ChatOpenAI(api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model="deepseek/deepseek-r1-0528:free")

            return llm.invoke(formatted_prompt).content

    elif ext==".pdf":
        with open(file_path, "rb") as f:
            content = PdfReader(f)
            text=""
            for page in content.pages:
                text += page.extract_text() or ""
            text = text.strip()

            template = """You are an expert assistant. Answer accurately and concisely, you will understand Query and answer from Text
            Query:"{query_string}",
            Text:{text}
            Answer: 
            """
            prompt = PromptTemplate(template = template, input_variables= ["query_string", "text"])
            formatted_prompt = prompt.format(query_string= query_string, text= text)
            
            llm = ChatOpenAI(api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model="deepseek/deepseek-r1-0528:free")

            return llm.invoke(formatted_prompt).content

    elif ext ==".xlsx":
        df = pd.read_excel(file_path, nrows=20)
        text = df.to_string(index=False)
        template = """You are an expert assistant. Answer accurately and concisely, you will understand Query and answer from Text
        Query:"{query_string}",
        Text:{text}
        Answer: 
        """
        prompt = PromptTemplate(template = template, input_variables= ["query_string", "text"])
        formatted_prompt = prompt.format(query_string= query_string, text= text)
        
        llm = ChatOpenAI(api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model="deepseek/deepseek-r1-0528:free")    
        return llm.invoke(formatted_prompt).content
    
    elif ext in [".jpg", ".jpeg", ".png"]:
        with open(file_path, "rb") as f:
            image_bytes= f.read()
            encoded_image= base64.b64encode(image_bytes).decode('utf-8')
            message = HumanMessage(content=[
            {"type": "text", "text": f"Query:: {query_string}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        ])

        # llm = ChatOpenAI(
        #     api_key="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",
        #     base_url="https://openrouter.ai/api/v1",
        #     model="google/gemini-2.0-flash-exp:free"
        # )
            llm = ChatOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model="google/gemini-2.0-flash-exp:free"
            )

            response = llm.invoke([message])
            return response.content
    
#multi_model_agent("C:\Actualisation_Task\Screenshot 2025-10-25 214049.png", "what is present in this Image, give me a short summary?")
        
def intelligent_assistant(query, file_path = None):
    if file_path:
        return multi_model_agent(file_path ,query)
    else:
        category = route_query(query)
        if category == "direct":
            return direct_query_agent(query)
        elif category == "web_search":
            return web_search_agent(query)
        # elif category == "multi_model":
        #     return multi_model_agent(file_path, query)

import gradio as gr

def gradio_wrapper(query, file):
    # file will be None if no file uploaded
    # file will be a file path string if uploaded
    return intelligent_assistant(query, file)

# Create the interface
interface = gr.Interface(
    fn=gradio_wrapper,
    inputs=[
        gr.Textbox(label="Enter your query", placeholder="Ask me anything..."),
        gr.File(label="Upload file (optional)", file_types=[".txt", ".pdf", ".xlsx", ".jpg", ".jpeg", ".png"])
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Intelligent Assistant",
    description="Ask questions or upload files for analysis"
)

interface.launch(share=False, server_name="0.0.0.0")