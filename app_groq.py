import gradio as gr
import json
import re
import os
from groq import Groq
from to_do_agent import TodoList, execute_tool

# Drop Peft and Transformers. Use the Groq API.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# 70B is a massive brain. It won't miss the NLP nuances.
MODEL_ID = "openai/gpt-oss-20b" 

env = TodoList()

sys_prompt = """You are a planning agent. You must output your plan as a raw JSON array of tool calls.
Do not output markdown, explanations, or text. Only output the JSON array.
Available Tools:
- name: "add", arguments: {"task": "<string>"}
- name: "complete", arguments: {"task_id": <integer>}"""

def parse_tool_calls(text: str):
    calls = []
    text = re.sub(r"```json|```", "", text)
    start = 0
    while True:
        pos = text.find("{", start)
        if pos == -1: break
        brace_count = 0
        for i in range(pos, len(text)):
            if text[i] == "{": brace_count += 1
            elif text[i] == "}": brace_count -= 1
            if brace_count == 0:
                try:
                    obj = json.loads(text[pos:i+1])
                    if isinstance(obj, dict) and "name" in obj:
                        calls.append(obj)
                except json.JSONDecodeError: pass
                start = i + 1
                break
        else: break
    return calls

def chat_and_execute(message, history):
    dynamic_sys = sys_prompt + f"\n\nCurrent List State:\n{env.list_tasks()}\nUse the correct integer IDs from this list."
    messages = [
        {"role": "system", "content": dynamic_sys}, 
        {"role": "user", "content": message}
    ]
    
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=250,
        temperature=0.1
    )
    response_text = response.choices[0].message.content
    
    calls = parse_tool_calls(response_text)
    action_log = f"**Raw Plan Generated:**\n```json\n{json.dumps(calls, indent=2)}\n```\n\n**Execution Log:**\n"
    
    for call in calls:
        name, args = call.get("name"), call.get("arguments", {})
        args_str = args if isinstance(args, str) else json.dumps(args)
        result = execute_tool(env, name, args_str)
        action_log += f"- `{name}({args_str})` -> {result}\n"
        
    return action_log, env.list_tasks()

with gr.Blocks(title="API-Powered Agent") as demo:
    gr.Markdown("# 🧠 Groq-Powered Llama-3.3-70B Agent")
    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.ChatInterface(fn=lambda msg, hist: chat_and_execute(msg, hist)[0], autofocus=True)
        with gr.Column(scale=1):
            gr.Markdown("### Current State (TodoList Env)")
            state_display = gr.Textbox(value=env.list_tasks(), lines=15, interactive=False, label="Memory")
    chat.chatbot.change(fn=lambda: env.list_tasks(), inputs=None, outputs=state_display)

if __name__ == "__main__":
    demo.launch()