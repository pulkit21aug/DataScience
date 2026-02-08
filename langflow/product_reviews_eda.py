# from lfx.field_typing import Data
from lfx.custom.custom_component.component import Component
from lfx.io import DataInput ,Output ,DataFrameInput
from lfx.schema.data import Data
from lfx.schema import DataFrame


class CustomComponent(Component):
    display_name = "Combine product reviews by product"
    description = "Identify review comments common with  higest frequency and their sentiments."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "code"
    name = "Combine Reviews"

    inputs = [
        DataFrameInput(
            name="input_value",
            display_name="Input Value",
            info="This is a custom component Input",
            value="Product Review!",
            tool_mode=False,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> DataFrame:
        df = DataFrame(self.input_value)
        df['Review'] = df['Review'].astype(str)
        df_product_grp = df.groupby('Cloth_class')['Review'].apply(lambda x: ' '.join(x)).reset_index()
        df_lfx_product_grp= DataFrame(df_product_grp)
        self.status = "OK"
        return df_lfx_product_grp
