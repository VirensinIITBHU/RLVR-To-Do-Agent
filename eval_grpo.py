import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from to_do_agent import TodoList, verify_task, execute_tool

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
prompt_text = "add 3 items, then mark item 2 as complete."

# 1-Shot prompt matching the GRPO training script
sys_prompt = """You are a planning agent. You must output your plan as a raw JSON array of tool calls.
Do not output markdown, explanations, or text. Only output the JSON array.

Available Tools:
- name: "add", arguments: {"task": "<string>"}
- name: "complete", arguments: {"task_id": <integer>}

Example Target Output:
[
  {"name": "add", "arguments": {"task": "item 1"}},
  {"name": "complete", "arguments": {"task_id": 1}}
]"""

def parse_tool_calls(text: str):
    """Bulletproof parser to extract JSON blocks even if wrapped in arrays or markdown."""
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
                except json.JSONDecodeError:
                    pass
                start = i + 1
                break
        else:
            break
    return calls

def run_grpo_eval(model, tokenizer, text):
    env = TodoList()
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=250, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"\n--- Raw Model Output ---\n{response_text}\n")
    print("--- Execution Logs ---")
    
    calls = parse_tool_calls(response_text)
    
    for call in calls:
        name, args = call.get("name"), call.get("arguments", {})
        args_str = args if isinstance(args, str) else json.dumps(args)
        print(f"Executing: {name} | {args_str}")
        execute_tool(env, name, args_str)
        
    return verify_task(env), env.list_tasks()

if __name__ == "__main__":
    print("Loading Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    
    print("\nLoading GRPO Adapters...")
    grpo_model = PeftModel.from_pretrained(base_model, "./qwen-todo-grpo-final")
    
    score, state = run_grpo_eval(grpo_model, tokenizer, prompt_text)
    print(f"\n[GRPO] Final Score: {score}\nState:\n{state}")