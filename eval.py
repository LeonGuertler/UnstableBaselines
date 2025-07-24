import unstable, time

t0 = time.time()

run = unstable.build_evaluator(
    model_name = "Qwen/Qwen3-4B-Base",
    num_runs_per_env=32,
    eval_envs = [
        # Single-Player
        unstable.EvalEnvSpec(env_id="Hangman-v0-train",         num_players=1, prompt_template="qwen3-sp"),
        unstable.EvalEnvSpec(env_id="LightsOut-v0-train",       num_players=1, prompt_template="qwen3-sp"),
        unstable.EvalEnvSpec(env_id="FrozenLake-v0-train",      num_players=1, prompt_template="qwen3-sp"),
        
        # Two-Player
        unstable.EvalEnvSpec(env_id="Chess-v0-train",           num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-2.0-flash-lite-001"),
        unstable.EvalEnvSpec(env_id="ColonelBlotto-v0-train",   num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-2.0-flash-lite-001"),
        unstable.EvalEnvSpec(env_id="TicTacToe-v0-train",       num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-2.0-flash-lite-001"),
        unstable.EvalEnvSpec(env_id="SpellingBee-v0-train",     num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-2.0-flash-lite-001"),
        
        # Multi-Player
        unstable.EvalEnvSpec(env_id="ThreePlayerIPD-v0-train",  num_players=3, prompt_template="qwen3-mp", fixed_opponent="google/gemini-2.0-flash-lite-001"),
        unstable.EvalEnvSpec(env_id="Codenames-v0-train",       num_players=4, prompt_template="qwen3-mp", fixed_opponent="google/gemini-2.0-flash-lite-001"),
        unstable.EvalEnvSpec(env_id="Surround-v0-train",        num_players=5, prompt_template="qwen3-mp", fixed_opponent="google/gemini-2.0-flash-lite-001"),
    ], 
)
run.evaluate(num_eval_workers=128)


print(f"Done, took: {time.time()-t0}")