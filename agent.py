import ollama

MODEL = "llama3.1:8b"

def get_weather(city):
    return f"its 72 and sunnny in {city}."


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get the weather for."
                    }
                },
                "required": ["city"]
            },
        },
    }
]

def run(question):
    messages = [
        {
            "role": "system", 
            "content": ("You are a helpful assistant, only call get_weather if the user explicitly asks for the weather in a specific city. If the user does not ask for the weather, do not call the tool and answer the question directly. When you are not calling a tool, respond in plain, natural sentences - never JSON, never a function call, never code. When you receive a tool result, use it as the source of truth and answer the user directly and naturally, as if you simply know the information. Never mention the tool, the function call, or that a lookup happened. Never simulate or invent a result — only use what the tool actually returned."),
        },
        {"role": "user", "content": question}]

    # first call
    response = ollama.chat(model=MODEL, messages=messages, tools=tools)
    messages.append(response.message)

    tool_calls = response.message.tool_calls
    if not tool_calls:
        print("No tool needed.")
        print("Answer:", response.message.content)
        return

    for call in tool_calls:
        name = call.function.name
        args = call.function.arguments
        print(f"Model wants to call: {name}({args})")

        if name == "get_weather":
            result =get_weather(**args)
        else: 
            result = "Unknown tool"

        # Give the result back to the model
        messages.append({"role": "tool", "name": name, "content": result})

    # second call
    final = ollama.chat(model=MODEL, messages=messages, tools=tools)
    print("Final Answer:", final.message.content)

run("What's the weather in Portland?")
