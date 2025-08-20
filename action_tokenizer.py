"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union, Tuple, Any

import numpy as np
from numpy import ndarray
from transformers import PreTrainedTokenizerBase
from transformers import LlamaTokenizer

class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, accer_input: float = 0.0, steer_input: float= 0.0, bins: int = 256, min_action: float = -1.0, max_action: float = 1.0,
            steer_from_range: tuple = (-0.2, 0.2), accer_from_range: tuple = (-3.0, 3.0), to_range: tuple = (-1.0, 1.0),
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action
        self.accer_input, self.steer_input = accer_input, steer_input
        self.accer_from_range, self.steer_from_range, self.to_range = accer_from_range, steer_from_range, to_range

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.action_input = np.array([self.accer_map_values(), self.steer_map_values()])

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self) -> Union[Tuple[str, ndarray], Tuple[List[str], ndarray]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(self.action_input, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)
        action_token_ids = list(self.tokenizer.vocab_size - discretized_action)
        action_token_ids = np.array(action_token_ids)
        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action)), action_token_ids
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist()), action_token_ids

    @property
    def convert_action_to_tokenids(self) -> Tuple[Any, Any]:
        action_decode, action_token_ids = self.__call__()
        return action_decode, action_token_ids

    def convert_tokenids_to_action(self, action_token_ids) -> Tuple[Any, Any]:
        inverse_accel, inverse_steer = self.decode_token_ids_to_actions(action_token_ids)
        return self.inverse_accer_map_values(input_array=inverse_accel), self.inverse_steer_map_values(input_array=inverse_steer)

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    def steer_map_values(self):
        # Extract the ranges
        from_min, from_max = self.steer_from_range
        to_min, to_max = self.to_range
        # Apply the linear mapping formula
        mapped_array = (self.steer_input - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

        return np.array(mapped_array)

    def accer_map_values(self):
        # Extract the ranges
        from_min, from_max = self.accer_from_range
        to_min, to_max = self.to_range
        # Apply the linear mapping formula
        mapped_array = (self.accer_input - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
        return np.array(mapped_array)

    def inverse_steer_map_values(self, input_array):
        # Extract the ranges
        from_min, from_max = self.to_range
        to_min, to_max = self.steer_from_range
        # Apply the linear mapping formula
        mapped_array = (input_array - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
        return np.array(mapped_array)

    def inverse_accer_map_values(self, input_array):
        # Extract the ranges
        from_min, from_max = self.to_range
        to_min, to_max = self.accer_from_range
        # Apply the linear mapping formula
        mapped_array = (input_array - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
        return np.array(mapped_array)





