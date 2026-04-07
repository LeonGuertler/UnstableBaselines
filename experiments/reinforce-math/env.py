import random
from datasets import load_dataset
from typing import Tuple
import textarena as ta
import re
from typing import Tuple

from datasets import load_dataset
from openai import OpenAI
import textarena as ta


DEFAULT_JUDGE_PROMPT = """You are given a math problem's ground-truth answer
and a model response.

Ground truth answer:
```
{answer}
```

Model response:
```
{response}
```

Determine whether the response already contains a correct final answer.
Do NOT fix, extend, or re-solve the problem.
If the response does not clearly contain the correct answer, respond "no".

Respond with exactly one word: yes or no.
"""


class SingleTurnDatasetEnv(ta.Env):

    def __init__(
        self,
        dataset: str = 'scripts/SingleTurnEnv/data/aime2025.jsonl',
        type: str = 'math',
        verifier: str = 'math-verify',
        vllm_base_url: str = 'http://localhost:8000/v1',
        vllm_model: str = 'openai/gpt-oss-120b',
    ):
        super().__init__()
        self.type = type
        self.verifier = verifier
        if verifier == 'judge':
            self.judge_client = OpenAI(base_url=vllm_base_url, api_key="EMPTY")
            self.judge_model = vllm_model
        self.dataset = load_dataset("json", data_files=dataset)['train']

    def reset(self, num_players, seed=None):
        self.state = ta.SinglePlayerState(num_players=1, seed=seed, max_turns=1)
        raw_game_state = self.dataset[seed % len(self.dataset)]
        if self.type == 'multiple-choice':
            if 'answer' not in raw_game_state:
                options, correct_index = raw_game_state["options"], 0
                indexed_options = list(enumerate(options))
                random.shuffle(indexed_options)
                new_correct_index = next(i for i, (orig_i, _) in enumerate(indexed_options) if orig_i == correct_index)
                option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"   
                game_state = { "question": raw_game_state["question"], "options": [opt for _, opt in indexed_options], "answer": option_letters[new_correct_index]}
            else: game_state = {"question": raw_game_state["question"], "options": raw_game_state["options"], "answer": raw_game_state["answer"]}
        else: game_state = raw_game_state
        self.state.reset(game_state=game_state, player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: dict) -> str:
        if self.type == 'multiple-choice':
            option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            prompt = "Please reason step by step, and put your final answer within \\boxed{}. Your final answer should be of the following format: \\boxed{{LETTER}} where LETTER is one of ABCD.\n\n"
            prompt += f"Question: {game_state['question']}\n\n"
            prompt += "Choices:\n"
            for i, option in enumerate(game_state["options"]): prompt += f"{option_letters[i]}) {option}\n"
            return prompt
        return str(game_state["question"])

    def get_observation(self):
        player_id, observation = self.state.current_player_id, self.state.get_current_player_observation()
        if not observation:
            prompt = self._generate_player_prompt(player_id, self.state.game_state)
            observation = [(ta.GAME_ID, prompt, ta.ObservationType.PROMPT)]
        return player_id, observation

    def llm_as_a_judge(self, solution: str, response: str) -> bool:
        prompt = DEFAULT_JUDGE_PROMPT.format(answer=solution, response=response)
        completion = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": "You are a strict mathematical verifier."},
                {"role": "user", "content": prompt},
            ], temperature=0.1
        )
        text = completion.choices[0].message.content.strip().lower()
        return text == "yes"

    def math_verify(self, solution: str, response: str) -> bool:
        from math_verify import verify, parse
        solution = parse(solution)
        response = parse(response)
        return verify(solution, response)

    def exact_match(self, solution: str, response: str) -> bool: return solution.strip() == response.strip()

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        if self.verifier == 'exact': verify = self.exact_match
        elif self.verifier == 'math-verify': verify = self.math_verify
        elif self.verifier == 'judge': verify = self.llm_as_a_judge
        else: raise ValueError(f"Unknown verifier: {self.verifier}")
        if self.type == 'multiple-choice': match = re.search(r'\[(.*)\]', action)
        if verify(self.state.game_state["answer"], action): self.state.set_outcome( reward=1.0, reason="Correct solution.")
        else: self.state.set_outcome(reward=-1.0, reason=f"Incorrect solution. Expected: {self.state.game_state['answer']}")
        return self.state.step()
