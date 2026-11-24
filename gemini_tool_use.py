"""
gemini_tool_use.py
==================
Demonstrates the Gemini AI "tool-use loop" using the generate_content API.

The AI model is given ONE tool: look_up_price(product_name).
The script drives the full loop manually so every step is visible:

  Step 1  — Build the tool declaration
  Step 2  — Send user question + tool declaration to the model
  Step 3  — Detect a function_call in the model response
  Step 4  — Execute the real Python function
  Step 5  — Return the result to the model
  Step 6  — Receive the model's final natural-language answer

Run:
  pip install -r requirements.txt
  cp .env.example .env          # then set GEMINI_API_KEY in .env
  python gemini_tool_use.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # reads .env → sets GEMINI_API_KEY in the environment

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    sys.exit(
        "Error: GEMINI_API_KEY is not set.\n"
        "Copy .env.example → .env and add your key."
    )

# Path to the local product catalogue (lives next to this script)
PRODUCTS_FILE = Path(__file__).parent / "products.json"
MODEL = "gemini-4-hydrogen"   # fast, balanced frontier model


# ---------------------------------------------------------------------------
# 1. The real Python function (the "tool")
# ---------------------------------------------------------------------------
def look_up_price(product_name: str) -> dict:
    """
    Look up a product's price in the local products.json catalogue.

    Args:
        product_name: Name of the product to look up (case-insensitive).

    Returns:
        A dict with keys: price, currency, in_stock — or an error key.
    """
    with open(PRODUCTS_FILE, "r", encoding="utf-8") as fh:
        catalogue = json.load(fh)

    key = product_name.strip().lower()
    if key in catalogue:
        return catalogue[key]           # e.g. {"price": 999.99, "currency": "USD", "in_stock": True}
    return {"error": f"Product '{product_name}' not found in catalogue."}


# ---------------------------------------------------------------------------
# 2. Tool declaration — describes the function to the model in JSON-Schema
# ---------------------------------------------------------------------------
TOOL_DECLARATION = types.FunctionDeclaration(
    name="look_up_price",
    description=(
        "Look up the current price of a product from the local inventory catalogue. "
        "Returns price, currency, and stock availability."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "The name of the product to look up, e.g. 'laptop' or 'headphones'.",
            }
        },
        "required": ["product_name"],
    },
)

TOOL = types.Tool(function_declarations=[TOOL_DECLARATION])


# ---------------------------------------------------------------------------
# Helper: pretty banner
# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# 3. The tool-use loop
# ---------------------------------------------------------------------------
def run_tool_use_loop(user_question: str) -> None:
    """Drive the full tool-use loop for a single user question."""

    client = genai.Client()

    # ── STEP 1: Build the conversation history ──────────────────────────────
    # We start with just the user's message.
    banner("STEP 1 — User question")
    print(f"User: {user_question}")

    conversation: list[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_question)],
        )
    ]

    # ── STEP 2: First API call — model sees the question + tool declaration ─
    banner("STEP 2 — Sending question + tool declaration to the model")
    print(f"Model : {MODEL}")
    print(f"Tool  : {TOOL_DECLARATION.name}")

    response = client.models.generate_content(
        model=MODEL,
        contents=conversation,
        config=types.GenerateContentConfig(
            tools=[TOOL],
            # Letting the model decide whether to call a tool (default = AUTO)
        ),
    )

    # ── STEP 3: Detect whether the model wants to call a function ───────────
    banner("STEP 3 — Inspect model response")

    # The model may return either:
    #   a) function_calls  → it wants us to run a tool
    #   b) text            → it answered directly (no tool needed)
    if not response.function_calls:
        banner("No tool call — model answered directly")
        print(response.text)
        return

    # There can be multiple function calls; we handle each one.
    function_call_part = response.function_calls[0]
    print(f"Model requested tool call : {function_call_part.name}")
    print(f"Arguments                 : {function_call_part.args}")

    # Keep the model's reply in the conversation history so the next turn has
    # full context (the SDK requires this for multi-turn tool use).
    conversation.append(response.candidates[0].content)

    # ── STEP 4: Execute the real function locally ───────────────────────────
    banner("STEP 4 — Executing the tool function locally")

    # Map function name → actual callable (useful when you have many tools)
    function_registry: dict = {
        "look_up_price": look_up_price,
    }

    fn = function_registry.get(function_call_part.name)
    if fn is None:
        tool_result = {"error": f"Unknown tool: {function_call_part.name}"}
    else:
        try:
            # Unpack the model's arguments dict as keyword arguments
            tool_result = fn(**function_call_part.args)
        except Exception as exc:
            # Surface the error to the model so it can explain it gracefully
            tool_result = {"error": str(exc)}

    print(f"Tool returned: {json.dumps(tool_result, indent=2)}")

    # ── STEP 5: Return the tool result to the model ─────────────────────────
    banner("STEP 5 — Sending tool result back to the model")

    # Wrap the result as a function-response Content so the API understands it.
    function_response_part = types.Part.from_function_response(
        name=function_call_part.name,
        response=tool_result,
    )
    conversation.append(
        types.Content(
            role="tool",
            parts=[function_response_part],
        )
    )

    # Second API call — model now has the tool result and writes its answer.
    response2 = client.models.generate_content(
        model=MODEL,
        contents=conversation,
        config=types.GenerateContentConfig(tools=[TOOL]),
    )

    # ── STEP 6: Final natural-language answer ───────────────────────────────
    banner("STEP 6 — Model's final answer")
    print(response2.text)


# ---------------------------------------------------------------------------
# 4. Entry point — try a few questions
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    questions = [
        "How much does a laptop cost?",
        "What is the price of the monitor, and is it in stock?",
        "Do you have a price for a standing desk?",
        "What is the price of a toaster?",          # not in catalogue → error path
    ]

    for q in questions:
        run_tool_use_loop(q)
        print()
