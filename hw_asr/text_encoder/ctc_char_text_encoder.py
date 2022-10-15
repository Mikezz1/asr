# from base64 import decode
# from msilib.schema import SelfReg
from typing import Dict, List, NamedTuple
from collections import defaultdict

import torch
import torchaudio
from .char_text_encoder import CharTextEncoder
from torchaudio.models.decoder import ctc_decoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        last_char = self.EMPTY_TOK
        decoded_output = []
        for ind in inds:
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

        for proba in probs.detach().cpu():
            paths = self._extend_and_merge(paths, proba)
            paths = self._cut_beams(paths, beam_size)

        return sorted([Hypothesis((res+last_char).strip().replace(self.EMPTY_TOK, ''), float(proba))
                       for (res, last_char), proba in paths.items()], key=lambda x: -x.prob)

    def ctc_beam_search_pt(self, probs: torch.tensor, probs_length,
                           beam_size: int = 100) -> List[Hypothesis]:

        decoder = ctc_decoder(
            tokens=self.vocab, beam_size=beam_size, lexicon=None,
            blank_token="^", sil_token="^", nbest=20)
        res = decoder(probs.unsqueeze(0))
        return sorted([
            Hypothesis(
                self.ctc_decode(hypo.tokens.tolist()),
                hypo.score) for hypo in res[0]],
            key=lambda x: -x.prob)

    def _extend_and_merge(self, paths, proba):
        new_paths = defaultdict(float)
        for next_char_index, next_char_prob in enumerate(proba):
            next_char = self.ind2char[int(next_char_index)]

            for (text, last_char), prob in paths.items():
                new_prefix = text if next_char == last_char else(
                    text + next_char)
                new_prefix = new_prefix.replace(self.EMPTY_TOK, '')
                new_paths[(new_prefix, next_char)] += prob * next_char_prob

        return new_paths

    def _cut_beams(self, paths: dict, beam_size: int):
        return dict(
            list(sorted(paths.items(),
                        key=lambda x: x[1]))[-beam_size:])
