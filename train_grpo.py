# train_grpo.py
import json
import re
import random
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from to_do_agent import TodoList, execute_tool

# ponytail: Bulletproof extractor. If it outputs a JSON array, this parses each object out.
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
                except json.JSONDecodeError:
                    pass
                start = i + 1
                break
        else:
            break
    return calls

def environment_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        user_msg = prompt[-1]["content"]
        
        # ponytail: Lazy regex. As long as the prompt has the count first and target second, 
        # we don't care what English words surround them.
        numbers = re.findall(r'\d+', user_msg)
        if len(numbers) < 2:
            rewards.append(0.0)
            continue
            
        n_add, target_id = int(numbers[0]), int(numbers[1])
        env = TodoList()
        
        calls = parse_tool_calls(completion[0]["content"])
        
        for call in calls:
            name, args = call.get("name", ""), call.get("arguments", {})
            try:
                execute_tool(env, name, args if isinstance(args, str) else json.dumps(args))
            except Exception:
                pass
                
        if len(env.tasks) == n_add:
            target = next((t for t in env.tasks if t["id"] == target_id), None)
            if target and target["done"]:
                # 1.0 for perfection, 0.8 if they did the task but completed extra items too
                rewards.append(1.0 if sum(1 for t in env.tasks if t["done"]) == 1 else 0.8)
            else:
                rewards.append(0.5) # Got the add count right, failed the complete target
        else:
            rewards.append(0.0) # Failed the core task
    return rewards

def format_reward(completions, **kwargs):
    return [0.1 if len(parse_tool_calls(c[0]["content"])) > 0 else 0.0 for c in completions]

def generate_prompts(n=100):
    data = []
    # ponytail: The 1-Shot Schema Prompt. This bypasses the need for SFT formatting completely.
    base_sys = """You are a planning agent. You must output your plan as a raw JSON array of tool calls.
Do not output markdown, explanations, or text. Only output the JSON array.

Available Tools:
- name: "add", arguments: {"task": "<string>"}
- name: "complete", arguments: {"task_id": <integer>}

Example Target Output:
[
  {"name": "add", "arguments": {"task": "item 1"}},
  {"name": "complete", "arguments": {"task_id": 1}}
]"""

    for _ in range(n):
        count = random.randint(1, 3)
        
        # 50% chance to simulate a pre-existing state so the model learns to read it
        has_state = random.choice([True, False])
        offset = random.randint(1, 3) if has_state else 0
        
        # Now the model might be asked to complete ID 5 even if it's only adding 1 item
        target = random.randint(1, count + offset)
        
        sys_prompt = base_sys
        if has_state:
            sys_prompt += f"\n\nCurrent List State:\n[ ] {offset}: Old Task"
            
        # Fix mode collapse by randomly altering the sentence structure
        templates = [
            f"add {count} items, then mark item {target} as complete.",
            f"I need to add {count} tasks. Also mark task {target} done.",
            f"Create {count} items. Item {target} is finished."
        ]
        
        data.append({"prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": random.choice(templates)}
        ]})
    return Dataset.from_list(data)

if __name__ == "__main__":
    # Start fresh from the base model, ignoring broken SFT adapters
    model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16", device_map="auto")
    
    peft_config = LoraConfig(r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    
    training_args = GRPOConfig(
        output_dir="./qwen-todo-grpo",
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        num_generations=4,
        gradient_accumulation_steps=2,
        max_completion_length=250,
        num_train_epochs=1,
        logging_steps=1,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[environment_reward, format_reward],
        args=training_args,
        train_dataset=generate_prompts(100),
    )
    
    trainer.train()
    trainer.save_model("./qwen-todo-grpo-final")
    print("[✓] Saved GRPO weights to ./qwen-todo-grpo-final")