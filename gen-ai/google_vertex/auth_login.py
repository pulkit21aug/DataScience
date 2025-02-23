#  This file has logic to load google key file for authentication

#
import json
import yaml
# from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account

import google.generativeai as genai


SERVICE_ACCOUNT_PATH = "D:\google_service_account_keys\pulsdaily.json"
LOCATION = "us-central1"
API_KEY_FILE_PATH = "D:\google_service_account_keys\gemini_api_key.json"
API_KEY_LANG_SMITH = "D:\google_service_account_keys\langsmith.yaml"


def  initialize_vertex_ai_with_json():
    """Initializes Vertex AI with a service account JSON file.

  Args:
    service_account_file: Path to the service account JSON file.
  """

    # Load the service account credentials from the JSON file
    with open(SERVICE_ACCOUNT_PATH, 'r') as f:
        credentials_info = json.load(f)

    credentials_serv_acc = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)

    # Initialize Vertex AI with the loaded credentials
    vertexai.init(
        project=credentials_info['project_id'],
        location=LOCATION,  # Replace with your desired region
        credentials=credentials_serv_acc,
    )

    model = GenerativeModel("gemini-1.5-flash-002")
    return model


def genAiSdk_login():
    with open(API_KEY_FILE_PATH, 'r') as f:
        data = json.load(f)

    genai.configure(api_key=data['api_key'])

    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

def get_gemini_api_key():
    with open(API_KEY_FILE_PATH, 'r') as f:
        data = json.load(f)
    return data['api_key']


def get_langsmith_tracer_config():
    with open(API_KEY_LANG_SMITH, 'r') as f:
        lang_smith_key_value_pairs = yaml.safe_load(f)
    return lang_smith_key_value_pairs
