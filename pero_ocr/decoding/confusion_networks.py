from pero_ocr.sequence_alignment import levenshtein_alignment_path
import math


def get_pivot(cn):
    pivot = []
    for sausage in cn:
        pivot.append(sorted(sausage, key=lambda k: sausage[k], reverse=True)[0])

    return pivot


def add_hypothese(cn, transcript, score):
    if cn == []:
        cn = []
        for symbol in transcript:
            cn.append({symbol: score})

        return cn

    pivot = get_pivot(cn)
    alignment = levenshtein_alignment_path(list(transcript), pivot)
    cn_total_weight = sum(sum(position.values()) for position in cn) / len(cn)

    cn_pointer = 0
    tr_pointer = 0
    for direction in alignment:
        if direction == -1:  # move in the confusion network
            if None in cn[cn_pointer]:
                cn[cn_pointer][None] += score
            else:
                cn[cn_pointer][None] = score
            cn_pointer += 1
        elif direction == 0:  # move in both
            tr_sym = transcript[tr_pointer]
            if tr_sym in cn[cn_pointer]:
                cn[cn_pointer][tr_sym] += score
            else:
                cn[cn_pointer][tr_sym] = score
            tr_pointer += 1
            cn_pointer += 1
        elif direction == 1:  # move in the confusion network
            tr_sym = transcript[tr_pointer]
            if cn_pointer == len(cn):
                cn.append({None: cn_total_weight, tr_sym: score})
            else:
                cn = cn[:cn_pointer] + [{None: cn_total_weight, tr_sym: score}] + cn[cn_pointer:]
                cn_pointer += 1
            tr_pointer += 1
        else:
            raise RuntimeError("Got unexpected direction {}".format(direction))

    return cn


def normalize_cn(cn):
    for i in range(len(cn)):
        sausage_normalizer = sum(cn[i].values())
        for symbol in cn[i]:
            cn[i][symbol] /= sausage_normalizer

    return cn


def produce_cn_from_boh(boh, visual_weight=1.0, lm_weight=1.0, normalize=True):
    cn = []
    for hyp in boh:
        log_prob = visual_weight*hyp.vis_sc + (lm_weight*hyp.lm_sc if hyp.lm_sc is not None else 0.0)
        cn = add_hypothese(cn, hyp.transcript, math.exp(log_prob))

    if normalize:
        cn = normalize_cn(cn)

    return cn


def best_cn_path(cn):
    best_symbols = [sorted(position.keys(), key=lambda symbol: position[symbol], reverse=True)[0] for position in cn]
    return ''.join([s for s in best_symbols if s is not None])


def sorted_cn_paths(cn):
    if not cn:
        return []

    def path_from_arcs(arcs):
        string = ''
        prob = 1.0
        for (c, p) in arcs:
            if c is not None:
                string += c
            prob *= p

        return (string, prob)

    iters = [None] * len(cn)
    current = [None] * len(cn)

    def reset_iter(i):
        iters[i] = iter(sorted(cn[i].items(), key=lambda x: x[1], reverse=True))
        current[i] = next(iters[i])

    for i in range(len(cn)):
        reset_iter(i)

    paths = []

    while True:
        paths.append(path_from_arcs(current))

        finished = False
        for rotor_index in reversed(range(len(iters))):
            try:
                current[rotor_index] = next(iters[rotor_index])
                break
            except StopIteration:
                reset_iter(rotor_index)
        else:
            finished = True

        if finished:
            break

    return sorted(paths, key=lambda x: x[1], reverse=True)
