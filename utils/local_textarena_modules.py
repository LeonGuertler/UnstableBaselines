
import textarena as ta 
from typing import Dict, Tuple, List, Optional

class FirstLastObservationWrapper(ta.ObservationWrapper):
    def __init__(self, env: ta.Env):
        super().__init__(env)
        self.full_observations = {}

    def _convert_obs_to_str(self, player_id: int) -> ta.Observations:
        return_str = self.full_observations[player_id][0][1]
        if len(self.full_observations[player_id]) > 1:
            return_str += "\n\n" + self.full_observations[player_id][-1][1]

        return return_str + "\n\n" #+ "Next Action:"

    def observation(self, player_id: int, observation):
        if observation is None:
            return self._convert_obs_to_str(player_id=player_id)

        # Extend the full observations with the current observations without duplicates
        if player_id not in self.full_observations:
            self.full_observations[player_id] = []

        # Append new observations in sequence
        self.full_observations[player_id].extend(observation)

        return self._convert_obs_to_str(player_id=player_id)

class LLMObservationWrapper(ta.ObservationWrapper):
    def __init__(self, env: ta.Env):
        super().__init__(env)
        self.full_observations: Dict[int, List[Tuple[int, str]]] = {}

    def _convert_obs_to_str(self, player_id: int) -> Observations:
        str_observation = ""
        if player_id in self.full_observations:
            for sender_id, message in self.full_observations[player_id]:
                if sender_id == ta.GAME_ID:
                    sender_name = "GAME"
                else:
                    sender_name = self.env.state.role_mapping.get(sender_id, f"Player {sender_id}")
                str_observation += f"\n[{sender_name}] {message}"

        return str_observation

    def observation(self, player_id: int, observation: Optional[ta.Observations]):
        if observation is None:
            return self._convert_obs_to_str(player_id=player_id)

        # Extend the full observations with the current observations without duplicates
        if player_id not in self.full_observations:
            self.full_observations[player_id] = []

        # Append new observations in sequence
        self.full_observations[player_id].extend(observation)

        return self._convert_obs_to_str(player_id=player_id)



class ClipCharactersActionWrapper(ta.ActionWrapper):
    def __init__(self, env: ta.Env, max_num_characters: int):
        super().__init__(env)
        self.max_num_characters = max_num_characters

    def action(self, action: str) -> str:
        if len(action) <= self.max_num_characters:
            return action
        else:
            return action[: self.max_num_characters] # Truncate and return