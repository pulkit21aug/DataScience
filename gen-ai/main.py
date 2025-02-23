import google.generativeai as genai

from google_vertex import auth_login


def genText():
    prompt = "tell me a joke."
    # model =  auth_login.initialize_vertex_ai_with_json()
    model = auth_login.genAiSdk_login()

    chat=model.start_chat(history=[])

    #response = model.generate_content(prompt,generation_config=genai.types.GenerationConfig(temperature=2))

    response = chat.send_message(prompt)
    
    print(response)

    print(chat.history)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Gen AI..')
    genText()
