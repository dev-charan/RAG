import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    openai_api_base="https://api.groq.com/openai/v1",
)

# ── 1. StrOutputParser ────────────────────────────────────────────────────────
print("=" * 50)
print("1. StrOutputParser")
str_parser = StrOutputParser()

str_prompt = PromptTemplate(
    template="Name 3 popular {topic}.",
    input_variables=["topic"],
)

str_chain = str_prompt | llm | str_parser
str_result = str_chain.invoke({"topic": "programming languages"})
print(str_result)

# ── 2. CommaSeparatedListOutputParser ─────────────────────────────────────────
print("=" * 50)
print("2. CommaSeparatedListOutputParser")
csv_parser = CommaSeparatedListOutputParser()

csv_prompt = PromptTemplate(
    template="List 3 {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": csv_parser.get_format_instructions()},
)

csv_chain = csv_prompt | llm | csv_parser
csv_result = csv_chain.invoke({"topic": "programming languages"})
print(csv_result)          # returns a Python list

# ── 3. JsonOutputParser ───────────────────────────────────────────────────────
print("=" * 50)
print("3. JsonOutputParser")
json_parser = JsonOutputParser()

json_prompt = PromptTemplate(
    template="Return a JSON object with a key 'languages' containing a list of 3 popular {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

json_chain = json_prompt | llm | json_parser
json_result = json_chain.invoke({"topic": "programming languages"})
print(json_result)         # returns a Python dict

# ── 4. PydanticOutputParser ───────────────────────────────────────────────────
print("=" * 50)
print("4. PydanticOutputParser")

class LanguageList(BaseModel):
    languages: list[str] = Field(description="A list of programming language names")
    reason: str = Field(description="Why these languages are popular")

from langchain_core.output_parsers import PydanticOutputParser

pydantic_parser = PydanticOutputParser(pydantic_object=LanguageList)

pydantic_prompt = PromptTemplate(
    template="List 3 popular {topic} and explain briefly why they are popular.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
)

pydantic_chain = pydantic_prompt | llm | pydantic_parser
pydantic_result = pydantic_chain.invoke({"topic": "programming languages"})
print(pydantic_result)             # returns a LanguageList Pydantic object
print("Languages:", pydantic_result.languages)
print("Reason:", pydantic_result.reason)
