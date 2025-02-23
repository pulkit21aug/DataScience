from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


import langchain
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False


from google_vertex.auth_login import get_gemini_api_key

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embeddings/gemini-1.5-pro-embeddings", google_api_key=get_gemini_api_key())

##define model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, timeout=None, max_retries=2,
                                google_api_key=get_gemini_api_key()  # other params...
                                )

print(llm.invoke("Hi. I am Pulkit").content)