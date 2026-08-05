# An eval only catches failures you wrote rules for. The model always finds a new shape. Tightening the ruler is the ongoing job, not a one-time task.

import ollama

MODEL = "llama3.1:8b"

def get_weather(city):
    return f"its 72 and sunnny in {city}."


golden_set = [
    {
        "question": "What's the weather in Portland?",
        "must_contain": ["72", "sunny"],
        "must_not_contain": ["get_weather", "function", "tool", "{", "}"],
    },
    {
        "question": "Say hello to me.",
        "must_contain": [],
        "must_not_contain": ["get_weather", "function", "tool", "{", "}"],
    },
]


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
        ans = response.message.content
        return ans

    for call in tool_calls:
        name = call.function.name
        args = call.function.arguments

        if name == "get_weather":
            if "city" in args:
                result = get_weather(**args)
            else:
                result = "Error: no city provided."

        # Give the result back to the model
        messages.append({"role": "tool", "name": name, "content": result})

    # second call
    final = ollama.chat(model=MODEL, messages=messages, tools=tools)
    ans = final.message.content
    return ans

def grade(answer, case):
    answer = answer.lower()
    must_contain = case["must_contain"]
    must_not_contain = case["must_not_contain"]

    for phrase in must_contain:
        if phrase.lower() not in answer:
            return False

    for phrase in must_not_contain:
        if phrase.lower() in answer:
            return False

    return True



def loop():
    grade_counter = 0
    grade_total = len(golden_set)

    for case in golden_set:
        question = case["question"]
        print(f"Question: {question}")
        try:
            answer = run(question)
        except Exception as e:
            answer = f"Error: {e}"
            print(f"Error occurred: {e}")
        print(f"Answer: {answer}")
        if grade(answer, case):
            print("✅ Passed")
            grade_counter += 1
        else:
            print("❌ Failed")
        print()

    print (f"Current score: {grade_counter}/{grade_total}")

loop()

