# to-do-agent.py
import json
import os
from groq import Groq

import os
import json
from groq import Groq

# ponytail: naive env parser - breaks on multiline/complex escaping. Upgrade to python-dotenv if you need more.
for env_file in [".env", ".evn"]:  # Checking both just in case that typo made it to the file system
    try:
        with open(env_file) as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    except FileNotFoundError:
        pass


import json

def log_trajectory(messages: list, score: float, filepath: str = "trajectories.jsonl"):
    if score < 1.0:
        return
        
    # ponytail: SDK message objects aren't standard dicts. exclude_none strips nulls to keep the jsonl lean.
    clean_messages = [
        m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m 
        for m in messages
    ]
    
    with open(filepath, "a") as f:
        f.write(json.dumps({"reward": score, "messages": clean_messages}) + "\n")
    print(f"\n[+] Phase 6: Saved 1.0 trajectory to {filepath}")

# --- Phase 1: Environment ---
class TodoList:
    def __init__(self):
        self.tasks = []
        self.next_id = 1

    def add(self, task: str) -> str:
        self.tasks.append({"id": self.next_id, "task": task, "done": False})
        self.next_id += 1
        return f"Added task '{task}' with ID {self.next_id - 1}"

    def complete(self, task_id: int) -> str:
        for t in self.tasks:
            if t["id"] == task_id:
                if t["done"]:
                    return f"Task {task_id} was already done."
                t["done"] = True
                return f"Marked task {task_id} as done."
        return f"Error: Task {task_id} not found."

    def list_tasks(self) -> str:
        if not self.tasks:
            return "List is empty."
        return "\n".join(f"[{'X' if t['done'] else ' '}] {t['id']}: {t['task']}" for t in self.tasks)


# --- Phase 2: Agent Schemas & Execution Loop ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add a task.",
            "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete",
            "description": "Mark a task done.",
            "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List all tasks and their IDs.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

def execute_tool(env: TodoList, name: str, args_str: str) -> str:
    try:
        args = json.loads(args_str) if args_str else {}
    except json.JSONDecodeError:
        return "Error: Invalid JSON arguments."
        
    if name == "add": return env.add(args.get("task", ""))
    if name == "complete": return env.complete(args.get("task_id", 0))
    if name == "list_tasks": return env.list_tasks()
    return f"Error: Unknown tool {name}"

def run_agent(env: TodoList, client: Groq, model: str, task_prompt: str, max_steps: int = 10):
    messages = [{"role": "user", "content": task_prompt}]
    steps = 0
    
    # ponytail: hard step ceiling prevents infinite API call loops if model hallucinates or loops tool calls
    while steps < max_steps:
        steps += 1
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto"
        )
        msg = response.choices[0].message
        messages.append(msg)
        
        if not msg.tool_calls:
            break
            
        for tc in msg.tool_calls:
            result = execute_tool(env, tc.function.name, tc.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": str(result)
            })
            
    return messages[-1].content


# --- Phase 3: Verifier (Reward Function) ---
def verify_task(env: TodoList) -> float:
    """Calculates reward score between 0.0 and 1.0 based on environment state."""
    tasks = env.tasks
    
    # Must have added exactly 3 tasks
    if len(tasks) != 3:
        return 0.0
    
    second_task = next((t for t in tasks if t["id"] == 2), None)
    if not second_task:
        return 0.3
    
    completed_count = sum(1 for t in tasks if t["done"])
    if second_task["done"] and completed_count == 1:
        return 1.0
    elif second_task["done"]:
        return 0.7  # Partial credit if second task is completed alongside extra tasks
    
    return 0.5  # Partial credit for adding 3 tasks correctly




def run_rl_agent(env: TodoList, client: Groq, model: str, task_prompt: str, max_episodes: int = 3):
    messages = [{"role": "user", "content": task_prompt}]
    
    for episode in range(max_episodes):
        steps = 0
        # 1. Action Loop (The Policy)
        while steps < 10:
            steps += 1
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto"
            )
            msg = response.choices[0].message
            messages.append(msg)
            
            if not msg.tool_calls:
                break
                
            for tc in msg.tool_calls:
                result = execute_tool(env, tc.function.name, tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": str(result)
                })

        # 2. Reward Calculation (The Environment Feedback)
        score = verify_task(env)
        if score == 1.0:
            log_trajectory(messages, score)  # <--- Phase 6 hook
            return score, messages[-1].content
        # 3. The "RL" Update (Reflection)
        # ponytail: since we can't update weights, we append the negative reward to the prompt state
        messages.append({
            "role": "user",
            "content": f"SYSTEM REWARD: {score}/1.0. Task incomplete. Current list state:\n{env.list_tasks()}\nDo not ask for clarification. Generate dummy data if needed and execute tools to achieve 1.0."
        })
        
    return score, messages[-1].content


# --- Phase 5: Execution ---
if __name__ == "__main__":
    if os.environ.get("GROQ_API_KEY"):
        client = Groq()
        # Testing just the 120B model that failed previously
        models_to_test = ["openai/gpt-oss-120b"] 
        prompt = "add three items, then mark the second one done."
        
        for model_name in models_to_test:
            print(f"\n--- Running RL Loop: {model_name} ---")
            env = TodoList()
            final_score, final_output = run_rl_agent(env, client, model_name, prompt)
            
            print(f"Final Output: {final_output}")
            print(f"Final List State:\n{env.list_tasks()}")
            print(f"Final Verifier Score: {final_score}")

# --- Self-Check ---
def _self_check():
    test_env = TodoList()
    assert test_env.list_tasks() == "List is empty."
    test_env.add("First")
    test_env.add("Second")
    test_env.add("Third")
    assert len(test_env.tasks) == 3
    assert verify_task(test_env) == 0.5
    test_env.complete(2)
    assert verify_task(test_env) == 1.0
    assert test_env.complete(99) == "Error: Task 99 not found."
    print("[✓] Self-check passed.")

if __name__ == "__main__":
    _self_check()
    
    # --- Phase 4: Baseline Multi-Model Evaluation ---
    if os.environ.get("GROQ_API_KEY"):
        client = Groq()
        # models_to_test = ["llama3-70b-8192", "llama3-8b-8192"]

        models_to_test = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        prompt = "add three items, then mark the second one done."
        
        for model_name in models_to_test:
            print(f"\n--- Testing Model: {model_name} ---")
            env = TodoList()
            agent_output = run_agent(env, client, model_name, prompt)
            score = verify_task(env)
            
            print(f"Agent Output: {agent_output}")
            print(f"Final List State:\n{env.list_tasks()}")
            print(f"Verifier Reward Score: {score}")
    else:
        print("\nSet GROQ_API_KEY environment variable to execute Phase 4 model evaluation.")