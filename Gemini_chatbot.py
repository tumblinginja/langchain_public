# pip install google-generativeai
# pip install langchain-google-genai

import os, mimetypes, base64
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache, InMemoryCache

# Step 1: Load API Keys from .env
load_dotenv(find_dotenv(), override=True)

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GEMINI_API_KEY"),  
    model='gemini-2.5-flash',
    temperature=0.7
)

# Step 2: Optional caching
set_llm_cache(InMemoryCache())
# OR set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# Step 3: Chat Sessions
history = FileChatMessageHistory('chat_history.json')

# Step 4: Memory
memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=history,
    return_messages=True
)

# Step 5: Prompt Template
prompt = ChatPromptTemplate(
    input_variables=["content", "chat_history"],
    messages=[
        SystemMessage(content="You are a chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="chat_history"),  # Where memory is inserted
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

def encode_image(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, 'rb') as image_file:
        b64 = base64.b64encode(image_file.read()).decode('utf-8')
    return mime_type, b64    

# ---------------- Main Loop ----------------
while True:
    question = input('Text or image?\n')
    if question.lower() == 'text':
        while True:
            content = input("Your prompt: ")
            if content.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            # Create messages with memory included
            messages = [
                SystemMessage(content="You are a chatbot having a conversation with a human."),
                *memory.load_memory_variables({})["chat_history"],
                HumanMessage(content=content)
            ]

            # Stream the response
            response_text = ""  # ✅ initialize here
            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)
                response_text += chunk.content

            # Save input/output to memory
            memory.save_context({"input": content}, {"output": response_text})
            
            print("\n" + "-" * 50)

    elif question.lower() == 'image':
        input_image = input('What is the image file path and format?\n')
        mime, byte_image = encode_image(input_image)

        while True:
            image_question = input('What question do you have about this image?\n')
            if image_question.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            message = HumanMessage(
                content=[
                    {
                        'type': 'text',
                        'text': image_question  
                    },
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:{mime};base64,{byte_image}'}  
                    }
                ]
            )
            response_text = ""
            for chunk in llm.stream([message]):
                print(chunk.content, end="", flush=True)
                response_text += chunk.content

            # Save input/output to memory
            memory.save_context({"input": image_question}, {"output": response_text})
            
            print("\n" + "-" * 50)
