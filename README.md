# 🤖 Gemini Tool-Use Demo — AI Function Calling in Python

A minimal, **fully-commented** Python project that shows exactly how an AI model
(Google Gemini) uses a **tool** (function calling) to look up a product's price
from a local JSON file.

Every step of the tool-use loop is printed to the console so you can watch the
conversation unfold in real time.

---

## 📋 Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [The Tool-Use Loop — Explained Step by Step](#the-tool-use-loop--explained-step-by-step)
   - [Step 0 — Configuration](#step-0--configuration)
   - [Step 1 — The Real Python Function (the Tool)](#step-1--the-real-python-function-the-tool)
   - [Step 2 — Tool Declaration (Describing the Function to the AI)](#step-2--tool-declaration-describing-the-function-to-the-ai)
   - [Step 3 — First API Call: Send the Question + Tool Declaration](#step-3--first-api-call-send-the-question--tool-declaration)
   - [Step 4 — Inspect the Response: Did the Model Call a Tool?](#step-4--inspect-the-response-did-the-model-call-a-tool)
   - [Step 5 — Execute the Real Function Locally](#step-5--execute-the-real-function-locally)
   - [Step 6 — Send the Tool Result Back to the Model](#step-6--send-the-tool-result-back-to-the-model)
   - [Step 7 — The Model's Final Natural-Language Answer](#step-7--the-models-final-natural-language-answer)
5. [How the Conversation History Is Built](#how-the-conversation-history-is-built)
6. [Data Files](#data-files)
7. [Sample Output](#sample-output)
8. [Extending the Project](#extending-the-project)
9. [Security Notes](#security-notes)

---

## What This Project Does

```
User:  "How much does a laptop cost?"
         │
         ▼
   ┌─────────────┐    tool declaration    ┌───────────────┐
   │  Your Code  │ ──────────────────────▶│  Gemini Model │
   │             │◀──────────────────────┤               │
   └─────────────┘   function_call req.   └───────────────┘
         │
         │  runs locally
         ▼
   look_up_price("laptop")  →  {"price": 999.99, "currency": "USD", "in_stock": true}
         │
         │  returns result to model
         ▼
   ┌─────────────┐    tool result         ┌───────────────┐
   │  Your Code  │ ──────────────────────▶│  Gemini Model │
   │             │◀──────────────────────┤               │
   └─────────────┘   final text answer    └───────────────┘
         │
         ▼
   "The laptop costs $999.99 and is currently in stock."
```

The model never accesses `products.json` itself — it **asks your code** to run
the function and then uses the result to compose a human-friendly answer.

---

## Project Structure

```
gemini-tool-use-demo/
├── gemini_tool_use.py   ← main script (the tool-use loop lives here)
├── products.json        ← local product catalogue (the "database")
├── requirements.txt     ← Python dependencies
├── .env.example         ← template for your API key
├── .gitignore           ← keeps secrets and junk out of git
└── README.md            ← you are here
```

---

## Quick Start

### Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.9 |
| pip | any recent |
| Gemini API key | [Get one free](https://aistudio.google.com/apikey) |

### 1 — Clone the repo

```bash
git clone https://github.com/<your-username>/gemini-tool-use-demo.git
cd gemini-tool-use-demo
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **What gets installed:**
> - `google-genai` ≥ 3.5.0 — the official Gemini Python SDK
> - `python-dotenv` — loads `GEMINI_API_KEY` from your `.env` file

### 4 — Add your API key

```bash
cp .env.example .env
# Now open .env in any text editor and set:
# GEMINI_API_KEY=AIza...your_actual_key_here...
```

> [!CAUTION]
> Never commit `.env` to git. It is listed in `.gitignore`.

### 5 — Run it

```bash
python gemini_tool_use.py
```

---

## The Tool-Use Loop — Explained Step by Step

The core of this project is `run_tool_use_loop()` in
[`gemini_tool_use.py`](gemini_tool_use.py). Here is a detailed walkthrough of
every step, with the corresponding code alongside.

---

### Step 0 — Configuration

```python
load_dotenv()                           # reads .env into environment variables
API_KEY = os.environ.get("GEMINI_API_KEY")
```

`python-dotenv` reads the `.env` file and injects `GEMINI_API_KEY` into the
process environment. The `google-genai` SDK then picks it up automatically via
`genai.Client()` — **no key is ever hardcoded**.

---

### Step 1 — The Real Python Function (the Tool)

```python
def look_up_price(product_name: str) -> dict:
    with open(PRODUCTS_FILE, "r", encoding="utf-8") as fh:
        catalogue = json.load(fh)
    key = product_name.strip().lower()
    if key in catalogue:
        return catalogue[key]
    return {"error": f"Product '{product_name}' not found in catalogue."}
```

This is a **plain Python function** — no AI magic here.  
It opens `products.json`, looks up the product by name, and returns a dict:

```json
{ "price": 999.99, "currency": "USD", "in_stock": true }
```

If the product doesn't exist it returns an `error` key instead of raising an
exception, so the model can relay the error politely to the user.

**Key idea:** The model never runs this function. *Your code* runs it. The
model only *requests* that it be called, and you decide whether to honor that.

---

### Step 2 — Tool Declaration (Describing the Function to the AI)

```python
TOOL_DECLARATION = types.FunctionDeclaration(
    name="look_up_price",
    description="Look up the current price of a product ...",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "The name of the product, e.g. 'laptop'.",
            }
        },
        "required": ["product_name"],
    },
)
TOOL = types.Tool(function_declarations=[TOOL_DECLARATION])
```

A `FunctionDeclaration` is a **JSON-Schema description** of your function that
you send to the model. The model uses it to understand:

| Field | Purpose |
|-------|---------|
| `name` | The exact function name it must use in the call request |
| `description` | Natural-language explanation — this guides the model's decision |
| `parameters_json_schema` | What arguments to pass and their types |

You can register **multiple** tools in one `Tool` object. The model picks the
right one based on the user's question.

---

### Step 3 — First API Call: Send the Question + Tool Declaration

```python
conversation = [
    types.Content(role="user", parts=[types.Part.from_text(text=user_question)])
]

response = client.models.generate_content(
    model=MODEL,
    contents=conversation,
    config=types.GenerateContentConfig(tools=[TOOL]),
)
```

This is the **first round-trip** to the Gemini API.

You send:
- **`contents`** — the conversation so far (just the user message at this point)
- **`config.tools`** — the list of functions the model is allowed to call

The model decides: *"Can I answer this directly, or do I need a tool?"*

- If it can answer directly → it returns `response.text` (a string)
- If it needs data → it returns `response.function_calls` (a list of calls)

---

### Step 4 — Inspect the Response: Did the Model Call a Tool?

```python
if not response.function_calls:
    # Model answered directly — no tool needed
    print(response.text)
    return

function_call_part = response.function_calls[0]
print(f"Model requested: {function_call_part.name}({function_call_part.args})")
```

`response.function_calls` is either empty (direct answer) or contains one or
more `FunctionCall` objects.

Each `FunctionCall` has:
| Attribute | Example value |
|-----------|--------------|
| `name` | `"look_up_price"` |
| `args` | `{"product_name": "laptop"}` |

The model is essentially saying: *"Please run `look_up_price(product_name='laptop')`
and tell me what it returns."*

```
# Append the model's reply to the conversation history.
# This is mandatory — the next API call must include it
# so the model remembers what it asked.
conversation.append(response.candidates[0].content)
```

---

### Step 5 — Execute the Real Function Locally

```python
function_registry = {
    "look_up_price": look_up_price,
}

fn = function_registry.get(function_call_part.name)
tool_result = fn(**function_call_part.args)
```

**Your code** runs the function using the arguments supplied by the model.

A `function_registry` dict is good practice when you have several tools — it
maps `name → callable` safely, without using `eval()` or anything risky.

The `**function_call_part.args` unpacks the model's argument dict as keyword
arguments, so this is equivalent to:

```python
look_up_price(product_name="laptop")
```

The function returns something like:
```python
{"price": 999.99, "currency": "USD", "in_stock": True}
```

---

### Step 6 — Send the Tool Result Back to the Model

```python
function_response_part = types.Part.from_function_response(
    name=function_call_part.name,
    response=tool_result,
)
conversation.append(
    types.Content(role="tool", parts=[function_response_part])
)

response2 = client.models.generate_content(
    model=MODEL,
    contents=conversation,
    config=types.GenerateContentConfig(tools=[TOOL]),
)
```

This is the **second round-trip** to the Gemini API.

You wrap the tool result in a `Part.from_function_response` and add it to the
conversation with `role="tool"`. The model now has everything it needs:

1. The original question (`role="user"`)
2. Its own tool-call request (`role="model"`, appended in Step 4)
3. The actual data (`role="tool"`)

The API call returns the final response with the answer composed in natural
language.

---

### Step 7 — The Model's Final Natural-Language Answer

```python
print(response2.text)
```

The model synthesises the raw data (`{"price": 999.99, ...}`) into a
human-friendly sentence:

```
The laptop costs $999.99 USD and is currently in stock.
```

---

## How the Conversation History Is Built

Each turn appends to the `conversation` list. By the end of a tool-use loop,
the full history looks like:

```
conversation[0]  role="user"   "How much does a laptop cost?"
conversation[1]  role="model"  function_call: look_up_price(product_name="laptop")
conversation[2]  role="tool"   function_response: {price: 999.99, ...}
```

This multi-turn structure is **required** by the API. If you omit conversation[1]
(the model's function call), the API will reject the request with an error.

---

## Data Files

### `products.json`

A flat JSON object where each key is a product name (lowercase) and the value
is its metadata:

```json
{
  "laptop":     { "price": 999.99,  "currency": "USD", "in_stock": true  },
  "headphones": { "price": 149.99,  "currency": "USD", "in_stock": true  },
  "keyboard":   { "price": 89.99,   "currency": "USD", "in_stock": false },
  ...
}
```

Add or edit entries freely — the script reads it fresh on every call.

---

## Sample Output

```
============================================================
  STEP 1 — User question
============================================================
User: How much does a laptop cost?

============================================================
  STEP 2 — Sending question + tool declaration to the model
============================================================
Model : gemini-4-hydrogen
Tool  : look_up_price

============================================================
  STEP 3 — Inspect model response
============================================================
Model requested tool call : look_up_price
Arguments                 : {'product_name': 'laptop'}

============================================================
  STEP 4 — Executing the tool function locally
============================================================
Tool returned: {
  "price": 999.99,
  "currency": "USD",
  "in_stock": true
}

============================================================
  STEP 5 — Sending tool result back to the model
============================================================

============================================================
  STEP 6 — Model's final answer
============================================================
The laptop is priced at $999.99 USD and is currently in stock.
```

---

## Extending the Project

| Idea | How to do it |
|------|-------------|
| Add more tools | Add more `FunctionDeclaration` objects to `TOOL` and register callables in `function_registry` |
| Read from a database | Replace `json.load(...)` in `look_up_price` with a DB query |
| Handle multiple function calls per turn | Loop over `response.function_calls` instead of only taking `[0]` |
| Use a streaming response | Pass `stream=True` and iterate over events |
| Multi-turn chat | Thread `previous_interaction_id` through subsequent Interactions API calls |

---

## Security Notes

- **API key** is loaded from the environment via `python-dotenv`. It is **never
  hardcoded** in source files.
- `.env` is listed in `.gitignore` to prevent accidental commits.
- The function registry uses an explicit `dict` lookup instead of `eval()` or
  `getattr` on arbitrary names, preventing prompt-injection attacks from
  triggering unintended functions.
- Tool results are returned as plain Python dicts — no shell commands or file
  writes are involved.
