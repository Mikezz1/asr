# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    return editdistance.distance(
        target_text, predicted_text) / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    splitted_target = target_text.split(' ')
    return editdistance.distance(
        splitted_target, predicted_text.split(' ')) / len(splitted_target)
