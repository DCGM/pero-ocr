import numpy as np


def greedy_filtration(line_probs, chars):
    idx = -1
    text = ""
    last_char = None
    probs = []

    for i, (char_index, max_prob) in enumerate(zip(np.argmax(line_probs, axis=1), np.max(line_probs, axis=1))):
        if char_index != (line_probs.shape[1] - 1):
            if (last_char != chars[char_index]):
                text = text + chars[char_index]
                probs.append([max_prob])
                idx += 1
                last_char = chars[char_index]
            else:
                if idx != -1:
                    probs[idx].append(max_prob)
        else:
            last_char = None

    for i, item in enumerate(probs):
        probs[i] = sum(probs[i]) / len(probs[i])

    return text, probs
