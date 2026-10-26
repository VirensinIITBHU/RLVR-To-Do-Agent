# train_sota.py
import json
import re
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from to_do_agent import TodoList, execute_tool

def parse_tool_calls(text: str):
    if not text or not isinstance(text, str): 
        return []
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

def format_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        # ponytail: Failsafe extraction to support both TRL 0.11 (strings) and 0.12+ (dicts)
        c_text = c[0]["content"] if isinstance(c, list) else c
        rewards.append(0.1 if len(parse_tool_calls(c_text)) > 0 else 0.0)
    return rewards

def validity_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        c_text = c[0]["content"] if isinstance(c, list) else c
        calls = parse_tool_calls(c_text)
        if not calls:
            rewards.append(0.0)
            continue
        valid = sum(1 for call in calls if call.get("name") in ["add", "complete"] and "arguments" in call)
        rewards.append(0.2 if valid == len(calls) else -0.1)
    return rewards

def environment_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        sys_msg = prompt[0]["content"]
        meta_match = re.search(r"<META:ADD=(\d+),DONE=(\d+)>", sys_msg)
        
        if not meta_match:
            rewards.append(0.0)
            continue
            
        n_add, target_id = int(meta_match.group(1)), int(meta_match.group(2))
        env = TodoList()
        
        starting_items = 0
        if "Current List State:" in sys_msg:
            state_block = sys_msg.split("Current List State:\n")[1].split("<META")[0]
            starting_items = len([line for line in state_block.split('\n') if line.strip() and line.startswith("[")])
            for i in range(starting_items):
                env.add(f"Pre-existing Task {i+1}")
        
        c_text = completion[0]["content"] if isinstance(completion, list) else completion
        calls = parse_tool_calls(c_text)
        
        for call in calls:
            name, args = call.get("name", ""), call.get("arguments", {})
            try:
                execute_tool(env, name, args if isinstance(args, str) else json.dumps(args))
            except Exception:
                pass
                
        target = next((t for t in env.tasks if t["id"] == target_id), None)
        total_expected_len = starting_items + n_add
        
        if len(env.tasks) == total_expected_len:
            if target and target["done"]:
                rewards.append(1.0 if sum(1 for t in env.tasks if t["done"]) == 1 else 0.8)
            else:
                rewards.append(0.4)
        else:
            rewards.append(0.0)
    return rewards

def generate_prompts(n=1000):
    data = []
    base_sys = """You are a planning agent. Output your plan as a raw JSON array of tool calls.
Do not output text. Only output the JSON array.

Available Tools:
- name: "add", arguments: {"task": "<string>"}
- name: "complete", arguments: {"task_id": <integer>}"""

    for _ in range(n):
        count = random.randint(1, 4)
        has_state = random.choice([True, False])
        starting_items = random.randint(1, 3) if has_state else 0
        target = random.randint(1, count + starting_items)
        
        sys_prompt = base_sys
        if has_state:
            sys_prompt += "\n\nCurrent List State:\n"
            for i in range(starting_items):
                sys_prompt += f"[ ] {i+1}: Existing Routine Task\n"
        
        sys_prompt += f"\n<META:ADD={count},DONE={target}>"
        
        templates = [
            f"add {count} items, then mark item {target} as complete.",
            f"I need to add {count} tasks. Also mark task {target} done.",
            f"Create {count} items. Item {target} is finished.",
            f"Can you append {count} things to my list? Oh, and ID {target} is already handled, cross it off.",
            f"New tasks: item 1, item 2... up to {count} items. Strike out number {target}.",
            f"I have {count} more things to do today. Add them. Also, mark {target} as complete immediately."
        ]
        
        data.append({"prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": random.choice(templates)}
        ]})
    return Dataset.from_list(data)

def _preflight_check():
    """Runs a simulated completion through the environment reward to guarantee no silent math/regex crashes occur mid-training."""
    test_prompts = [[{"role": "system", "content": "<META:ADD=2,DONE=1>\nCurrent List State:\n[ ] 1: Old Task\n"}]]
    test_completions = [[{"content": '[{"name": "add", "arguments": {"task": "x"}}, {"name": "complete", "arguments": {"task_id": 1}}]'}]]
    try:
        r = environment_reward(test_prompts, test_completions)
        assert len(r) == 1 and isinstance(r[0], float)
        print("[✓] Pre-flight reward check passed.")
    except Exception as e:
        raise RuntimeError(f"FATAL: Pre-flight environment check failed: {e}")

if __name__ == "__main__":
    _preflight_check()
    
    print("Loading Base Model for SOTA RL Training...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # ponytail: explicit pad token mapping prevents HF batching crashes.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # ponytail: dynamic fallback to float16 if your specific GPU doesn't support bfloat16 to prevent hardware exceptions.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    


    # applying bitsandbytes quantization to reduce memory footprint and speed up training

    # Change this block:
    # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=compute_dtype, device_map="auto")
    
    # To this QLoRA block:
    from transformers import BitsAndBytesConfig
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quant_config, 
        device_map="auto"
    )

    peft_config = LoraConfig(r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    
    training_args = GRPOConfig(
        output_dir="./qwen-todo-sota",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        num_generations=4,
        gradient_accumulation_steps=4,
        max_completion_length=300,
        num_train_epochs=1,
        logging_steps=5,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, validity_reward, environment_reward],
        args=training_args,
        train_dataset=generate_prompts(1000), 
    )
    
    print("Starting Deep Learning Matrix...")
    trainer.train()
    trainer.save_model("./qwen-todo-sota-final")
    print("[✓] Saved SOTA weights to ./qwen-todo-sota-final")