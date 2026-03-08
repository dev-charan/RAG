# Input → Prompt Template → LLM → Output

# 1. PromptTemplate        → dynamic reusable prompts
# 2. LLMChain              → prompt + LLM connected
# 3. Pipe operator ( | )   → modern way of chaining
# 4. SequentialChain       → chain of chains


# Step 1 — Without Chain (what you're doing now)

# llm.invoke("Tell me a joke about cricket")
# llm.invoke("Tell me a joke about football")
# llm.invoke("Tell me a joke about tennis")

# Step 2 — With PromptTemplate

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a joke about {topic}"
)

prompt.format(topic="cricket")
# → "Tell me a joke about cricket"

# Step 3 — LLMChain (connect prompt + LLM)


from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"topic": "cricket"})
 
#  Other way

from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
chain.invoke({"topic": "cricket"})


# Step 5 — SequentialChain (chain of chains)
Step 5 — SequentialChain (chain of chains)

# Chain 1 — generates a joke
# Chain 2 — translates that joke to Hindi

# chain1 = prompt1 | llm
# chain2 = prompt2 | llm
# Output of chain1 automatically feeds into chain2.

# final_chain = chain1 | chain2