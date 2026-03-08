llm.invoke("My name is Charan")
llm.invoke("What is my name?")  # LLM has no idea

# With memory — what happens:

# Behind the scenes LangChain sends:
[
    {"role": "user", "content": "My name is Charan"},
    {"role": "assistant", "content": "Nice to meet you Charan!"},
    {"role": "user", "content": "What is my name?"},  # now LLM has context
]