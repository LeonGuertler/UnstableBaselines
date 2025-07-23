import os
import torch
import time, json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

# ---------------------- System 2 Environment Setup --------------------------
# Critical: Set these BEFORE importing any CUDA/NCCL libraries
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["NCCL_DEBUG"] = "WARN"

def main():
    # ---------------------- Config --------------------------
    MODEL_NAME = "Qwen/QwQ-32B"  # Hugging Face model
    BATCH_SIZE = 64  # Reduced from 64 for memory constraints
    MAX_TOKENS = 4096
    TEMPERATURE = 0.8
    SYSTEM_PROMPT = "You are a helpful and competent assistant that can deduce the reasoning levels adopted in a text."
    OUTPUT_JSONL_PATH = "outputs.jsonl"
    OUTPUT_STATS_PATH = "inference_stats.txt"
    INPUT_CSV_PATH = "/home/chengxy/code-repo/UnstableBaselines/outputs/2025-07-21/18-53-53/sp-experiment5-run3-Qwen3-4B-Base-TowerOfHanoi-v0-train,TowerOfHanoi-v0-medium-train,TowerOfHanoi-v0-hard-train-1753095227/training_data/train_data_step_0.csv"
    # --------------------------------------------------------

    def check_gpu_memory():
        """Check available GPU memory before starting"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"GPU {i}: {gpu_memory:.1f} GB total memory")
        else:
            print("No CUDA GPUs available")

    def determine_optimal_config():
        """Determine best configuration for System 2"""
        gpu_count = torch.cuda.device_count()
        
        if gpu_count >= 2:
            # Check if we have enough memory for 14B model across 2 GPUs
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # 14B model needs ~28GB for inference, so need ~14GB per GPU minimum
            if gpu_memory_gb >= 16:  # Some headroom
                return {
                    "tensor_parallel_size": 4,
                    "gpu_memory_utilization": 0.85,  # More conservative
                    "enforce_eager": True,  # Better stability on consumer hardware
                }
            else:
                print(f"⚠️ GPU memory ({gpu_memory_gb:.1f}GB) may be insufficient for 14B model with tensor parallelism")
                return {
                    "tensor_parallel_size": 1,  # Single GPU fallback
                    "gpu_memory_utilization": 0.9,
                    "enforce_eager": True,
                }
        else:
            return {
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": True,
            }

    # Check system
    print("🔍 Checking System 2 configuration...")
    check_gpu_memory()
    config = determine_optimal_config()
    print(f"📋 Using config: {config}")

    # Input prompts
    dataset = load_dataset("csv", data_files=INPUT_CSV_PATH)["train"]
    acts = dataset["act"]  # This gives you the values in the 'act' column
    obs = dataset["obs"]
    input_texts = [
        f"""You are a reasoning analyst. Given an input text, your task is to identify which types of reasoning are present based on the list below. Return your answer strictly in the following JSON format, without any extra text.

---

JSON Output Format:
{{
  "reasoning_types": [
    {{
      "type": "<Reasoning Type Name>",
      "justification": "<Short explanation of how this reasoning type is used in the input text>"
    }},
    ...
  ]
}}

---

Reasoning Types (select all that apply):
- Deductive Reasoning
- Inductive Reasoning
- Abductive Reasoning
- Analogical Reasoning
- Hierarchical & Transitive Reasoning
- Path-Based Reasoning
- Constraint-Based Reasoning
- Causal & Counterfactual Reasoning
- Temporal Reasoning
- Probabilistic Reasoning
- Commonsense & Heuristic Reasoning
- Procedural Reasoning
- Game-Theoretic & Strategic Reasoning
- Intentional Reasoning (Theory of Mind)
- Negotiation & Bargaining Reasoning
- Resource Allocation & Economic Reasoning
- Numerical & Arithmetic Reasoning
- Pattern Recognition Reasoning
- Memory-Based Reasoning
- Exploratory Reasoning
- Deontic Reasoning
- Non-Monotonic Reasoning (Belief Revision)
- Meta-Reasoning (Introspective Reasoning)
- Simulation-Based Reasoning
- Multi-Hop Reasoning

---

Input Text:
\"\"\"{ob} {act}\"\"\"

Now respond in valid JSON only."""
        for ob, act in zip(obs, acts)
    ]

    # Optional: prepend system prompt (for chat-style models)
    def apply_system_prompt(user_prompt: str) -> str:
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    formatted_prompts = [apply_system_prompt(p) for p in input_texts]

    # Initialize vLLM with System 2 optimizations
    print("🚀 Initializing vLLM...")
    try:
        llm = LLM(
            model=MODEL_NAME,
            **config  # Use the determined configuration
        )
        print("✅ vLLM initialized successfully")
    except Exception as e:
        print(f"❌ vLLM initialization failed: {e}")
        print("💡 Try reducing gpu_memory_utilization or using tensor_parallel_size=1")
        raise

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Set sampling params
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    # Inference + Timing
    print(f"🔄 Processing {len(input_texts)} prompts in batches of {BATCH_SIZE}...")
    all_outputs = []
    num_output_tokens = 0
    start_time = time.time()

    try:
        for i in tqdm(range(0, len(formatted_prompts), BATCH_SIZE)):
            batch = formatted_prompts[i:i+BATCH_SIZE]
            outputs = llm.generate(batch, sampling_params)
            for j, output in enumerate(outputs):
                text = output.outputs[0].text.strip()
                num_tokens = len(tokenizer.encode(text))
                all_outputs.append({
                    "input": input_texts[i + j],
                    "output": text,
                    "token_count": num_tokens
                })
                num_output_tokens += num_tokens
                
            # Optional: Clear GPU cache periodically on consumer hardware
            if i % (BATCH_SIZE * 4) == 0:
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print("💡 Try reducing BATCH_SIZE or MAX_TOKENS")
        raise

    end_time = time.time()
    total_time = end_time - start_time
    prompts_per_second = len(input_texts) / total_time
    tokens_per_second = num_output_tokens / total_time

    # Print stats
    print(f"\n🕒 Inference completed in {total_time:.2f} seconds")
    print(f"⚡ Prompts/sec: {prompts_per_second:.2f}")
    print(f"⚡ Tokens/sec: {tokens_per_second:.2f}")
    print(f"📊 Total tokens generated: {num_output_tokens}")

    # Save to JSONL
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
        for item in all_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save stats
    with open(OUTPUT_STATS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Inference time (s): {total_time:.2f}\n")
        f.write(f"Total prompts: {len(input_texts)}\n")
        f.write(f"Total output tokens: {num_output_tokens}\n")
        f.write(f"Prompts/sec: {prompts_per_second:.2f}\n")
        f.write(f"Tokens/sec: {tokens_per_second:.2f}\n")
        f.write(f"Configuration used: {config}\n")

    print(f"✅ Saved outputs to {OUTPUT_JSONL_PATH}")
    print(f"✅ Saved stats to {OUTPUT_STATS_PATH}")

    # Final cleanup
    torch.cuda.empty_cache()
    print("🧹 GPU memory cleared")


if __name__ == '__main__':
    # Required for multiprocessing on Windows and some Linux systems
    from multiprocessing import freeze_support
    freeze_support()
    main()