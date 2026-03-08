from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    openai_api_base="https://api.groq.com/openai/v1",
)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])

chain = prompt | llm

history = []

while True:
    user_input = input("You : ")
    if user_input == "exit":
        break

    result = chain.invoke({"input":user_input,
                           "history":history
                           })
    history.append({"role":"user","content":user_input})
    history.append({"role": "assistant", "content": result.content})

    print(f'Bot: {result.content}\n')
