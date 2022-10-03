from base64 import decode
from typing import Dict, List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        last_char = inds[0]
        decoded_output = [self.ind2char[last_char]]
        for ind in inds[1:]:
            if (ind == last_char) or (ind == 0):
                continue
            decoded_output.append(self.ind2char[ind])
            last_char = ind
        return ''.join(decoded_output)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        paths = {
            ('', self.EMPTY_TOK): 1.0
        }

        for proba in probs:
            paths = self._extend_and_merge(paths, proba)
            paths = self._cut_beams(paths, beam_size)

        return sorted([Hypothesis((res+last_char).strip().replace(self.EMPTY_TOK, ''), proba)
                       for (res, last_char), proba in paths.items()], key=lambda x: x.prob, reverse=True)

    def _extend_and_merge(self, paths: dict, proba):
        new_paths = defaultdict(float)
        for (res, last_char), v in paths.items():
            for i in range(len(proba)):
                if self.ind2char[i] == last_char:
                    new_paths[res + self.ind2char[i]] = v * proba[i]
                else:
                    new_paths[((res+last_char).replace(self.EMPTY_TOK, ''),
                               self.ind2char[i])] += v*proba[i]
            return new_paths

    def _cut_beams(self, paths: dict, beam_size: int):
        return dict(
            list(sorted(paths.items(),
                        key=lambda x: x[1]))[-beam_size:])
