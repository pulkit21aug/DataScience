import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

from google_vertex.auth_login import get_gemini_api_key

##define model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, timeout=None, max_retries=2,
                             google_api_key=get_gemini_api_key()  # other params...
                             )

chain=RunnablePassthrough()
print(chain.invoke('Jai  Mata di'))

def string_upper(input):
    return input.upper()

chain = RunnablePassthrough () | RunnableLambda(string_upper)
print(chain.invoke('Jai  Mata di'))

### Creating Retriever using Vector DB

# db = Chroma.from_documents(new_docs, embeddings)
# retriever = db.as_retriever(search_kwargs={"k": 4})
#
# template = """Answer the question based only on the following context:
# {context}
#
# Question: {question}
# """
# prompt = PromptTemplate.from_template(template)
#
# retrieval_chain = (
#         RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
#         | prompt
#         | llm
#         | StrOutputParser()
# )
