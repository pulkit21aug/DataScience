from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import configurable, RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False


from google_vertex.auth_login import get_gemini_api_key

store ={}
def get_session_histstory(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


## parser for output parsing
str_par = StrOutputParser()

##define model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, timeout=None, max_retries=2,
                                google_api_key=get_gemini_api_key()  # other params...
                                )

messages = [("system", "You are a helpful assistant .",),
            ("human", "Hi. I am Pulkit"), ]

config = {"configurable":{'session_id':'1234'}}

llm_with_memory= RunnableWithMessageHistory(llm, get_session_histstory)

## chaining with LCEL
chain = llm_with_memory | str_par
res = chain.invoke(messages,config)

print(res)

messages = [("human", "What is my name"), ]
res = chain.invoke(messages,config)

print(res)



