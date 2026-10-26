# 🧠 RLVR To-Do Agent: Distilling Reasoning via GRPO

A complete, zero-fluff pipeline demonstrating **Reinforcement Learning with Verifiable Rewards (RLVR)**. This project implements the exact data-generation and distillation playbook used by modern reasoning models (like DeepSeek-R1 and OpenAI's o-series), scaled down to run on consumer hardware.

No bloated agent frameworks. No LLM-as-a-judge. Just pure Python state machines, Teacher-Student distillation, and Group Relative Policy Optimization (GRPO).

## 🚀 The Architecture

This project is broken down into a 4-step industry-standard RL pipeline:

1. **The Deterministic Environment:** A strict Python `TodoList` class acting as the state machine. Tools (`add`, `complete`) are routed deterministically. A programmatic verifier (`verify_task`) inspects the raw memory state to output a binary reward signal ($0.0$ or $1.0$).
2. **Data Generation (Teacher Model):** A massive frontier model (e.g., Llama-3 70B / 120B via Groq) solves the environment using a **Test-Time Reflection Loop**. If the verifier scores $< 1.0$, the environment state and penalty are injected back into context, forcing self-correction. Perfect rollouts are logged to a JSONL dataset.
3. **Supervised Fine-Tuning (SFT / Distillation):** A tiny 1.5B parameter model (Qwen 2.5) is fine-tuned via LoRA using the Hugging Face `trl` library. This step strictly teaches the model the JSON tool-calling schema and vocabulary (bypassing schema hallucinations).
4. **GRPO Policy Optimization (RL):** The SFT model is subjected to randomized NLP prompts. It generates multiple action plans ($G=4$) simultaneously. The Python verifier grades them using Process Reward Models (PRMs) and Outcome Reward Models (ORMs). The model updates its own weights to maximize the advantage of successful plans—eliminating the massive VRAM overhead of a Critic model.

## 🛠️ Tech Stack & Philosophy
* **Models:** Qwen 2.5 (1.5B) for local RL, Llama-3 (70B/8B) via Groq/Together AI for Teacher data/Zero-shot.
* **RL / ML:** PyTorch, Hugging Face `transformers`, `peft` (LoRA), `trl` (GRPOTrainer, SFTTrainer), `datasets`.
* **UI:** `gradio` for a lightweight, interactive local web interface.
* **Philosophy:** *The best abstraction is no abstraction.* The agent loop is a simple `while` loop. The verifier is a pure Python function. Complex failure modes (mode collapse, state desynchronization, schema hallucination) are solved at the data distribution level, not with UI hacks.

## 🧠 Key ML Concepts Demonstrated
* **Knowledge Distillation:** Transferring reasoning capabilities from a 120B model to a 1.5B model.
* **Group Relative Policy Optimization (GRPO):** Training an RL policy using group-normalized advantages rather than a heavy secondary value model.
* **Process vs. Outcome Rewards (PRM / ORM):** Multi-tiered reward functions penalizing invalid syntax (PRM) while strictly evaluating the final environment state (ORM).
* **Domain Randomization:** Generating thousands of highly varied NLP instruction templates (with embedded metadata tags for the Python grader) to prevent mode collapse and force true linguistic generalization.

## 🚀 Quick Start

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers peft trl datasets accelerate gradio groq openai
```

### 2. The Pipeline Scripts
* `to_do_agent.py`: The core deterministic environment, tools, verifier, and the baseline reflection execution loop.
* `batch_generator.py`: Hammers the Teacher model via Groq API to generate perfect 1.0 RLVR trajectories.
* `train_sft.py`: Distills the JSONL data into the Qwen 1.5B model using LoRA.
* `train_sota.py`: The production-grade GRPO training loop featuring Domain Randomization, robust JSON extractors, and PRM/ORM multi-reward signals.
* `app.py`: The Gradio UI. Drop in your trained PeftModel (offline) or an API client (Groq/Together) to interact with the agent visually.

### 3. Running the UI
```bash
# Set your API key if testing the cloud Teacher model
export GROQ_API_KEY="your_api_key_here"

# Launch the Gradio interface
python app.py
```
Navigate to `http://127.0.0.1:7860` in your browser to chat with the agent and view real-time state changes.