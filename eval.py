import unstable, time, os, argparse


def main(args):
    t0 = time.time()

    model_name = args.model_name  # <-- was hard-coded
    main_folder = "eval_results"

    output_folder = os.path.join(main_folder, model_name.replace("/", "-"))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    run = unstable.build_evaluator(
        model_name=model_name,
        output_folder=output_folder,
        num_runs_per_env=32,
        eval_envs=[

            # Single-Player
            unstable.EvalEnvSpec(env_id="Hangman-v0-train",                 num_players=1, prompt_template="qwen3-sp"),
            unstable.EvalEnvSpec(env_id="Minesweeper-v0-train",             num_players=1, prompt_template="qwen3-sp"),
            unstable.EvalEnvSpec(env_id="FrozenLake-v0-train",              num_players=1, prompt_template="qwen3-sp"),

            # Two-Player
            unstable.EvalEnvSpec(env_id="ConnectFour-v0-train",             num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-2.0-flash-lite-001"),
            unstable.EvalEnvSpec(env_id="ColonelBlotto-v0-train",           num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-2.0-flash-lite-001"),
            unstable.EvalEnvSpec(env_id="TicTacToe-v0-train",               num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-2.0-flash-lite-001"),
            unstable.EvalEnvSpec(env_id="SpellingBee-v0-train",             num_players=2, prompt_template="qwen3-zs", fixed_opponent="google/gemini-2.0-flash-lite-001"),

            # Multi-Player
            unstable.EvalEnvSpec(env_id="ThreePlayerTicTacToe-v0-train",    num_players=3, prompt_template="qwen3-mp", fixed_opponent="google/gemini-2.0-flash-lite-001"),
            unstable.EvalEnvSpec(env_id="Codenames-v0-train",               num_players=4, prompt_template="qwen3-mp", fixed_opponent="google/gemini-2.0-flash-lite-001"),
            unstable.EvalEnvSpec(env_id="Surround-v0-train",                num_players=5, prompt_template="qwen3-mp", fixed_opponent="google/gemini-2.0-flash-lite-001"),
        ],
    )
    run.evaluate(num_eval_workers=164)

    print(f"Done, took: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-m", default="Qwen/Qwen3-4B-Base", help="HF repo or local path of the model to evaluate")
    args = parser.parse_args()
    main(args)
