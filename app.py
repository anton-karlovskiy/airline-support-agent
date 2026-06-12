import json
import base64
import sqlite3
from io import BytesIO
from typing import TypeAlias

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
import gradio as gr
from PIL import Image

load_dotenv(override=True)

MODEL = "gpt-4.1-mini"
client = OpenAI()
DB = "prices.db"

ChatMessage: TypeAlias = dict[str, str]

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}
tools = [{"type": "function", "function": price_function}]


def init_db() -> None:
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)")
        cursor.execute("SELECT COUNT(*) FROM prices")
        if cursor.fetchone()[0] == 0:
            initial_prices = {"london": 799, "paris": 899, "tokyo": 1420, "sydney": 2999}
            cursor.executemany(
                "INSERT INTO prices (city, price) VALUES (?, ?)", initial_prices.items()
            )
        conn.commit()


def get_ticket_price(city: str) -> str:
    print(f"DATABASE TOOL CALLED: Getting price for {city}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT price FROM prices WHERE city = ?", (city.lower(),))
        result = cursor.fetchone()
    return f"Ticket price to {city} is ${result[0]}" if result else "No price data available for this city"


def generate_destination_image(city: str) -> Image.Image:
    image_response = client.images.generate(
        model="gpt-image-1",
        prompt=(
            f"An image representing a vacation in {city}, showing tourist spots and "
            f"everything unique about {city}, in a vibrant pop-art style"
        ),
        size="1024x1024",
        n=1,
    )
    image_data = base64.b64decode(image_response.data[0].b64_json)
    return Image.open(BytesIO(image_data))


def text_to_speech(message: str) -> bytes:
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=message,
    )
    return response.content


def handle_tool_calls(message: ChatCompletionMessage) -> tuple[list[ChatMessage], list[str]]:
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            city = json.loads(tool_call.function.arguments).get("destination_city")
            cities.append(city)
            responses.append({
                "role": "tool",
                "content": get_ticket_price(city),
                "tool_call_id": tool_call.id,
            })
    return responses, cities


def chat(history: list[ChatMessage]) -> tuple[list[ChatMessage], bytes, Image.Image | None]:
    messages = [{"role": "system", "content": system_message}] + [
        {"role": entry["role"], "content": entry["content"]} for entry in history
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    cities = []
    image = None

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses, cities = handle_tool_calls(message)
        messages.append(message)
        messages.extend(responses)
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    reply = response.choices[0].message.content
    history = list(history) + [{"role": "assistant", "content": reply}]
    voice = text_to_speech(reply)
    if cities:
        image = generate_destination_image(cities[0])

    return history, voice, image


def submit_message(message: str, history: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    return "", history + [{"role": "user", "content": message}]


def build_ui() -> gr.Blocks:
    with gr.Blocks() as interface:
        with gr.Row():
            chatbot = gr.Chatbot(height=500)
            image_output = gr.Image(height=500, interactive=False)
        with gr.Row():
            audio_output = gr.Audio(autoplay=True)
        with gr.Row():
            message_input = gr.Textbox(label="Chat with our AI Assistant:")

        message_input.submit(
            submit_message,
            inputs=[message_input, chatbot],
            outputs=[message_input, chatbot],
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, audio_output, image_output],
        )
    return interface


if __name__ == "__main__":
    init_db()
    build_ui().launch(inbrowser=True)
