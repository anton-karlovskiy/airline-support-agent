# Airline Support Agent

A multimodal AI customer support chatbot for the fictional airline FlightAI, built with OpenAI and Gradio. Ask about ticket prices and get a text reply, spoken audio, and a generated destination image all at once.

The chatbot answers questions using GPT-4.1-mini, looks up return ticket prices from a local SQLite database via tool use, reads every reply aloud with OpenAI TTS (Onyx voice), and generates a DALL-E 3 image when a destination city comes up.

## Seeded destinations

| City   | Return price |
|--------|-------------|
| London | $799        |
| Paris  | $899        |
| Tokyo  | $1,420      |
| Sydney | $2,999      |

## Setup

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd airline-support-agent
uv sync
cp .env.example .env
# set OPENAI_API_KEY in .env
```

## Running

```bash
uv run python app.py
```

The Gradio UI opens at `http://127.0.0.1:7860`.

## Project structure

```
app.py          # All application logic
prices.db       # SQLite database (created on first run)
pyproject.toml  # Dependencies
.env.example    # Environment variable template
```
