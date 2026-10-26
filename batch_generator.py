# batch_generator.py
import os
import random
from groq import Groq
from to_do_agent import TodoList, run_rl_agent

# ponytail: dynamic verifier so we don't train the model to blindly output exactly 3 items every single time.
def generate_task():
    n_add = random.randint(2, 5)
    target_id = random.randint(1, n_add)
    prompt = f"add {n_add} items, then mark item {target_id} as complete."
    return prompt, n_add, target_id

def verify_dynamic(env: TodoList, expected_count: int, target_id: int) -> float:
    if len(env.tasks) != expected_count:
        return 0.0
    target = next((t for t in env.tasks if t["id"] == target_id), None)
    if not target or not target["done"]:
        return 0.0
    
    completed_count = sum(1 for t in env.tasks if t["done"])
    return 1.0 if completed_count == 1 else 0.5

if __name__ == "__main__":
    client = Groq()
    model = "openai/gpt-oss-120b" # Only the Teacher model
    
    print(f"Starting batch generation with {model}...")
    
    successful_runs = 0
    attempts = 0
    
    while successful_runs < 50:
        attempts += 1
        prompt, count, target_id = generate_task()
        env = TodoList()
        
        print(f"\nAttempt {attempts} | Target: {successful_runs}/50 | Prompt: '{prompt}'")
        
        # We override the hardcoded verifier in run_rl_agent with our dynamic one
        import to_do_agent
        to_do_agent.verify_task = lambda e: verify_dynamic(e, count, target_id)
        
        score, _ = run_rl_agent(env, client, model, prompt)
        
        if score == 1.0:
            successful_runs += 1
            print(f"[✓] Success. Saved to trajectories.jsonl")
        else:
            print(f"[x] Failed (Score: {score}). Discarding.")
            
    print(f"\nDone. Generated 50 perfect trajectories in {attempts} attempts.")