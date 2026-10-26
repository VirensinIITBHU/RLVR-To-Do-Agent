import gradio as gr
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from to_do_agent import TodoList, execute_tool # Reuse your existing environment

print("Loading model... (this takes a sec)")
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", local_files_only=True)  # Use local files only to avoid downloading
# Load your custom RL-trained brain
model = PeftModel.from_pretrained(base_model, "./qwen-todo-sota-final")

# Global state for the demo session
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
    """Generates the plan, executes tools, and updates the UI."""
    # messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": message}]

    dynamic_sys = sys_prompt + f"\n\nCurrent List State:\n{env.list_tasks()}\nUse the correct integer IDs from this list."
    messages = [{"role": "system", "content": dynamic_sys}, {"role": "user", "content": message}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=250, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    calls = parse_tool_calls(response_text)
    action_log = f"**Raw Plan Generated:**\n```json\n{json.dumps(calls, indent=2)}\n```\n\n**Execution Log:**\n"
    
    for call in calls:
        name, args = call.get("name"), call.get("arguments", {})
        args_str = args if isinstance(args, str) else json.dumps(args)
        result = execute_tool(env, name, args_str)
        action_log += f"- `{name}({args_str})` -> {result}\n"
        
    # ponytail: returning the markdown log for the chat, and the raw text state for the sidebar
    return action_log, env.list_tasks()

# --- The GUI Layout ---
with gr.Blocks(title="Offline RL Agent") as demo:
    gr.Markdown("# 🧠 Locally Trained GRPO Agent (Zero API Calls)")
    
    with gr.Row():
        with gr.Column(scale=2):
            chat = gr.ChatInterface(
                fn=lambda msg, hist: chat_and_execute(msg, hist)[0], 
                # type="messages",
                autofocus=True
            )
        with gr.Column(scale=1):
            gr.Markdown("### Current State (TodoList Env)")
            state_display = gr.Textbox(value=env.list_tasks(), lines=15, interactive=False, label="Memory")

    # Wire the chat submit event to also update the state display textbox
    chat.chatbot.change(fn=lambda: env.list_tasks(), inputs=None, outputs=state_display)

if __name__ == "__main__":
    demo.launch()