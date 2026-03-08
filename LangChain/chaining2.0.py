from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    openai_api_base="https://api.groq.com/openai/v1",
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate a blog post title about {topic}"
)

content_prompt = PromptTemplate(
    input_variables=["title"],
    template="write a short (2lines) blog post for this title : {title}"
)

final_chain = (
    prompt
    | llm
    | StrOutputParser()
    | content_prompt
    | llm
    | StrOutputParser()
)

result = final_chain.invoke({"topic": "AI"})
print(result)
