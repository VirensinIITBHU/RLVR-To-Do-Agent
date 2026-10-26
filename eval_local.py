# eval_local.py
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from to_do_agent import TodoList, verify_task, execute_tool, TOOL_SCHEMAS

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
prompt = "add 3 items, then mark item 2 as complete."

def extract_tool_call(text: str):
    """Extracts valid JSON tool call with nested brace tracking."""
    if "<tool_call>" in text and "</tool_call>" in text:
        text = text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        
    start = text.find("{")
    if start == -1:
        return None
        
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None

def run_local_agent(model, tokenizer, prompt_text):
    env = TodoList()
    messages = [{"role": "user", "content": prompt_text}]
    
    # ponytail: naive loop ceiling (5 steps)
    for step in range(1, 6):
        inputs = tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        tc = extract_tool_call(response_text)
        if not tc or "name" not in tc:
            break  # No valid tool call generated, agent finished
            
        print(f"  Step {step}: Called tool '{tc['name']}' with args {tc.get('arguments')}")
        
        raw_args = tc.get("arguments", {})
        args_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
        
        messages.append({"role": "assistant", "content": response_text})
        result = execute_tool(env, tc["name"], args_str)
        messages.append({"role": "tool", "name": tc["name"], "content": str(result)})
            
    return verify_task(env), env.list_tasks()

if __name__ == "__main__":
    print("1. Testing Base Model (Qwen 1.5B)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    
    score_before, state_before = run_local_agent(base_model, tokenizer, prompt)
    print(f"\n[BEFORE SFT] Score: {score_before}\nState:\n{state_before}\n")
    print("-" * 50)
    
    print("2. Testing Fine-Tuned Model (LoRA Adapters)...")
    ft_model = PeftModel.from_pretrained(base_model, "./qwen-todo-final")
    
    score_after, state_after = run_local_agent(ft_model, tokenizer, prompt)
    print(f"\n[AFTER SFT] Score: {score_after}\nState:\n{state_after}\n")