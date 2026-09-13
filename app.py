import json
import os

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient




load_dotenv()

MODEL_ID = os.getenv(
    "HF_MODEL",
    "meta-llama/Llama-3.1-8B-Instruct"
)




PROMPT_STYLES = {

    "Structured / Business": """
Create realistic synthetic business records.

Follow the requested schema exactly.

Return ONLY a valid JSON array.
Do not include explanations.
Do not include markdown.
Do not use code fences.
""",

    "Diverse / Creative": """
Create diverse and realistic synthetic records.

Vary names, locations, categories and numerical values.
Avoid generating identical-looking rows.

Follow the requested schema exactly.

Return ONLY a valid JSON array.
Do not include explanations or markdown.
""",

    "Edge Cases / Testing": """
Create synthetic records useful for software testing.

Include mostly normal records but also include
realistic boundary and edge cases.

Keep all values consistent with the requested schema.

Return ONLY a valid JSON array.
Do not include explanations or markdown.
"""
}




def get_client():

    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError(
            "HF_TOKEN is missing. "
            "Set it in your .env file or Hugging Face Space secret."
        )

    return InferenceClient(
        model=MODEL_ID,
        token=token,
        provider="auto"
    )




def extract_json(text):

    text = text.strip()

    # Remove Markdown code fences if the model adds them.
    if text.startswith("```"):

        lines = text.splitlines()

        if len(lines) >= 3:
            lines = lines[1:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end > start:

        json_text = text[start:end + 1]

        return json.loads(json_text)

    raise ValueError(
        "The model did not return a valid JSON array."
    )




def build_prompt(
    domain,
    schema,
    rows,
    style,
    extra_requirements
):

    style_instruction = PROMPT_STYLES[style]

    prompt = f"""
{style_instruction}

DOMAIN:
{domain}

NUMBER OF RECORDS:
{rows}

REQUIRED SCHEMA:
{schema}

ADDITIONAL REQUIREMENTS:
{extra_requirements or "None"}

Generate exactly {rows} records.
Use realistic synthetic values.

Do not use real people's private information.
"""

    return prompt




def generate_dataset(
    domain,
    schema,
    rows,
    style,
    temperature,
    extra_requirements
):

    try:

        rows = int(rows)

        if rows < 1 or rows > 100:
            raise ValueError(
                "Number of rows must be between 1 and 100."
            )

        
        json.loads(schema)

        
        client = get_client()

        
        prompt = build_prompt(
            domain,
            schema,
            rows,
            style,
            extra_requirements
        )

        
        response = client.chat.completions.create(

            model=MODEL_ID,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a synthetic-data generation engine. "
                        "Generate only fictional synthetic data."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_tokens=min(
                6000,
                max(1200, rows * 120)
            ),

            temperature=float(temperature)
        )

        
        generated_text = response.choices[0].message.content

       
        data = extract_json(generated_text)

        
        if not isinstance(data, list):
            raise ValueError(
                "The model returned JSON, "
                "but it was not a JSON array."
            )

        
        dataframe = pd.DataFrame(data)

        if dataframe.empty:
            raise ValueError(
                "The model returned an empty dataset."
            )

        json_output = json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        )

        return dataframe, json_output

    except Exception as error:

        raise gr.Error(str(error))




with gr.Blocks(
    title="Synthetic Data Generator"
) as demo:

    gr.Markdown(
        """
        #  Synthetic Data Generator

        Generate realistic **synthetic datasets**
        using Meta's **Llama 3.1 8B Instruct** model
        through Hugging Face.

        **Built with Llama**
        """
    )

    with gr.Row():

       

        with gr.Column():

            domain = gr.Textbox(
                label="Dataset Domain",
                value="E-commerce customer orders",
                placeholder=(
                    "Example: banking transactions, "
                    "healthcare appointments, student records"
                )
            )

            schema = gr.Textbox(
                label="JSON Schema",
                value="""{
  "customer_id": "string",
  "age": "integer",
  "city": "string",
  "product": "string",
  "amount": "float",
  "status": "string"
}""",
                lines=8
            )

            rows = gr.Slider(
                minimum=1,
                maximum=100,
                value=10,
                step=1,
                label="Number of Records"
            )

            style = gr.Dropdown(
                choices=list(PROMPT_STYLES.keys()),
                value="Structured / Business",
                label="Generation Strategy"
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="Creativity / Temperature"
            )

            extra_requirements = gr.Textbox(
                label="Additional Requirements",
                placeholder=(
                    "Example: amount should be between "
                    "100 and 5000"
                ),
                lines=3
            )

            generate_button = gr.Button(
                " Generate Dataset",
                variant="primary"
            )

      

        with gr.Column():

            dataframe_output = gr.Dataframe(
                label="Generated Dataset",
                interactive=False
            )

            json_output = gr.Code(
                label="Raw JSON",
                language="json"
            )

    
    generate_button.click(
        fn=generate_dataset,

        inputs=[
            domain,
            schema,
            rows,
            style,
            temperature,
            extra_requirements
        ],

        outputs=[
            dataframe_output,
            json_output
        ]
    )




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
