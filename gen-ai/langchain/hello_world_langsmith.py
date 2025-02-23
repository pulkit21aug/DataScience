from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from google_vertex.auth_login import get_langsmith_tracer_config, get_gemini_api_key
import langchain

# Disable verbose/debug logs
langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False


# Function to configure LangSmith tracer
def configure_langsmith():
    # Load LangSmith tracer configuration
    config = get_langsmith_tracer_config()
    if not config:
        raise ValueError("LangSmith tracer configuration is missing or invalid.")

    api_url = config.get("LANGSMITH_ENDPOINT")
    api_key = config.get("LANGSMITH_API_KEY")
    project_name = config.get("LANGSMITH_PROJECT")

    if not all([api_url, api_key, project_name]):
        raise ValueError("Missing necessary LangSmith configuration values.")

    # Set environment variables for LangSmith
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project_name
    os.environ["LANGSMITH_ENDPOINT"] = api_url


@traceable  # Enables LangSmith tracing
def main():
    # Configure LangSmith tracer
    configure_langsmith()

    # Fetch API key for Google Generative AI
    google_api_key = get_gemini_api_key()
    if not google_api_key:
        raise ValueError("Google API Key is missing. Please configure it properly.")

    # Initialize the LLM model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        timeout=None,
        max_retries=2,
        google_api_key=google_api_key
    )

    # Define system message and prompt template
    system_msg = "Generate a 10-word wish message."
    prompt_template = ChatPromptTemplate.from_messages([("system", system_msg)])

    # Parse AI response
    str_parser = StrOutputParser()

    # Generate the prompt text
    prompt_text = prompt_template.format()

    # Invoke the LLM model with the generated prompt
    ai_response = llm.invoke(prompt_text)

    # Parse and print the result
    result = str_parser.invoke(ai_response)
    print("Model Response:", result)


if __name__ == "__main__":
    main()
