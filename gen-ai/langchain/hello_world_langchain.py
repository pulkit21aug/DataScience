from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import langchain

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False


from google_vertex.auth_login import get_gemini_api_key

## parser for output parsing
str_par = StrOutputParser()

##define model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, timeout=None, max_retries=2,
                             google_api_key=get_gemini_api_key()  # other params...
                             )

messages = [("system", "You are a helpful assistant that translates English to Hindi. Translate the user sentence.",),
            ("human", "I love programming a lot."), ]

##prompt definition
system_msg = " Generate a 10 words wish message"
template = ChatPromptTemplate.from_messages(["system", system_msg])

try:
    ##ai_msg =  llm.invoke(messages)
    ##res = str_par.invoke(ai_msg)

    ## chaining with LCEL
    #chain = llm | str_par
    #res = chain.invoke(messages)

   ## prompting
    prompt = template.invoke({})
    ai_msg =  llm.invoke(prompt)
    res = str_par.invoke(ai_msg)
    print(res)

except Exception as e:
    print(f"error:{e}")
