def find_optimal(logit, positions, idx):
    maximum = -100
    highest = -1
    for i, item in enumerate(positions):
        if maximum < logit[item][idx]:
            maximum = logit[item][idx]
            highest = item

    return highest


def narrow_label(label, logit, idx_of_last, on_one_liberal):
    last_char = None
    repeating = []
    for i, item in enumerate(label):
        if last_char == item and last_char != idx_of_last:
            repeating.extend([i])
        else:
            if repeating != []:
                high = find_optimal(logit, repeating, last_char)
                for e, elem in enumerate(repeating):
                    if on_one_liberal:
                        label[elem] = idx_of_last - 1
                    else:
                        label[elem] = idx_of_last
                label[high] = last_char
        if last_char != item:
            repeating = []
            if item != idx_of_last:
                repeating.append(i)
        last_char = item
    if repeating != []:
        high = find_optimal(logit, repeating, last_char)
        for i, item in enumerate(repeating):
            if on_one_liberal:
                label[item] = idx_of_last - 1
            else:
                label[item] = idx_of_last
        label[high] = last_char

    return label
