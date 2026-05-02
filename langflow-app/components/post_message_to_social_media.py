# from lfx.field_typing import Data
from lfx.custom.custom_component.component import Component
from lfx.inputs import MultilineInput
from lfx.io import MessageTextInput, Output
from lfx.schema.data import Data
import json
import re
import requests


class CustomComponent(Component):
    display_name = "Post API Request"
    description = "Make HTTP Post requests using curl."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "code"
    name = "API Request"

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Input Value",
            info="This is a custom component Input",
            value="Hello, World!",
            tool_mode=True,
        ),
        MultilineInput(
            name="curl_input",
            display_name="cURL",
            info=(
                "Paste a curl command to populate the fields. "
                "This will fill in the dictionary fields for headers and body."
            ),
            real_time_refresh=True,
            tool_mode=True,
            advanced=True,
            show=True,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Data:
        # Strip the Markdown fences
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", self.input_value).strip()
        iv = json.loads(clean)

        sentiment_value = iv['sentiment']

        sentiment_value = sentiment_value.lower()
        if(sentiment_value == "positive" or sentiment_value == "neutral"):
            print ("The sentiment is positive or neutral. Hence calling API to post message to Social Media")
            apiUrl = self.curl_input
            headers = {"Content-Type": "application/json"}

            resp =requests.post(url=apiUrl,json=iv,headers=headers)
            print("Response from API:", resp.json)
            data = Data(value="The sentiment is positive or neutral. Hence alling API to post message to Social Media")
        else:
            print ("The sentiment is negative. Hence not calling API to post message to Social Media")
            data = Data(value="The sentiment is negative. Hence not calling API to post message to Social Media")

        self.status = data
        return data
