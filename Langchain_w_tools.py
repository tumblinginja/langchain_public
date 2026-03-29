import os
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool

# Step 1: load API Keys
load_dotenv(find_dotenv(), override=True)

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo",
    temperature=0.1
)

# Pull ReAct prompt from hub
prompt = hub.pull('hwchase17/react')

# Tools
python_repl = PythonREPLTool()
python_repl_tool = Tool(
    name="Python REPL",
    func=python_repl.run,
    description="Useful when you need to use Python to answer a question. Input should be valid Python."
)

api_wrapper = WikipediaAPIWrapper()
wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="Useful for when you need to look up a topic, country, or person on Wikipedia."
)

search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name="DuckDuckGo Search",
    func=search.run,
    description="Useful for general internet searches."
)

tools = [python_repl_tool, wikipedia_tool, duckduckgo_tool]

# Create ReAct agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

# Ask question directly (no extra prompt_template)
question = "Tell me about Napoleon Bonaparte's early life"
output = agent_executor.invoke({"input": question})
print("\nFinal Answer:\n", output["output"])
