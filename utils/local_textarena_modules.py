
import textarena as ta 


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