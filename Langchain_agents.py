import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool


load_dotenv(find_dotenv(), override=True)
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  
    model="gpt-3.5-turbo",
    temperature=0.1
)


agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True,
    system_message="You are an experienced scientist and Python programmer. Explain like I'm 5 years old"
)

# Invoke the agent
prompt = 'Calculate the square root of the factorial of 12 and display it with 4 decimal points'
agent_executor.invoke(prompt)