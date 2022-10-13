from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
import torch
from string import ascii_lowercase
alphabet = list(ascii_lowercase + ' ')

if __name__ == '__main__':

    probs = torch.Tensor()
    log_probs_length = torch.Tensor()
    beam_size = 100

    text_encoder = CTCCharTextEncoder(alphabet)
    res = text_encoder.ctc_beam_search(
        probs, log_probs_length, beam_size=beam_size)[:10]

    print(res)
    print('--------------------')

    log_prob_argmax = torch.argmax(probs, dim=-1).numpy()
    pred_text = text_encoder.ctc_decode(log_prob_argmax[:log_probs_length])
    print(pred_text)
