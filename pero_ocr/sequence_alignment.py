import numpy as np


def levenshtein_distance(source, target, sub_cost=1, ins_cost=1, del_cost=1):
    target = np.array(target)
    dist = np.arange(len(target) + 1) * ins_cost
    for s in source:
        dist[1:] = np.minimum(dist[1:] + del_cost, dist[:-1] + (target != s) * sub_cost)
        dist[0] += del_cost
        for ii in range(len(dist) - 1):
            if dist[ii + 1] > dist[ii] + ins_cost:
                dist[ii + 1] = dist[ii] + ins_cost
    return dist[-1]


def levenshtein_alignment(source, target, sub_cost=1, ins_cost=1, del_cost=1, empty_symbol=None):
    target = np.array(target)
    backtrack = np.ones((len(source) + 1, len(target) + 1))
    backtrack[0] = -1
    dist = np.arange(len(target) + 1) * ins_cost
    for ii, s in enumerate(source):
        cost4sub = dist[:-1] + (target != s) * sub_cost
        dist += del_cost
        where_sub = cost4sub < dist[1:]
        dist[1:][where_sub] = cost4sub[where_sub]
        backtrack[ii + 1, 1:][where_sub] = 0
        for jj in range(len(dist) - 1):
            if dist[jj + 1] > dist[jj] + ins_cost:
                dist[jj + 1] = dist[jj] + ins_cost
                backtrack[ii + 1, jj + 1] = -1
    src_pos = len(source)
    tar_pos = len(target)
    alig = []
    while tar_pos > 0 or src_pos > 0:
        where = backtrack[src_pos, tar_pos]
        if where >= 0:
            src_pos -= 1
        if where <= 0:
            tar_pos -= 1
        alig.insert(0, (empty_symbol if where < 0 else source[src_pos],
                        empty_symbol if where > 0 else target[tar_pos]))
    return alig


def levenshtein_alignment_path(source, target, sub_cost=1, ins_cost=1, del_cost=1, empty_symbol=None):
    target = np.array(target)
    backtrack = np.ones((len(source) + 1, len(target) + 1))
    backtrack[0] = -1
    dist = np.arange(len(target) + 1) * ins_cost
    for ii, s in enumerate(source):
        cost4sub = dist[:-1] + (target != s) * sub_cost
        dist += del_cost
        where_sub = cost4sub < dist[1:]
        dist[1:][where_sub] = cost4sub[where_sub]
        backtrack[ii + 1, 1:][where_sub] = 0
        for jj in range(len(dist) - 1):
            if dist[jj + 1] > dist[jj] + ins_cost:
                dist[jj + 1] = dist[jj] + ins_cost
                backtrack[ii + 1, jj + 1] = -1
    src_pos = len(source)
    tar_pos = len(target)

    align = []
    while tar_pos > 0 or src_pos > 0:
        where = backtrack[src_pos, tar_pos]
        if where >= 0:
            src_pos -= 1
        if where <= 0:
            tar_pos -= 1
        align.append(where)
    return list(reversed(align))


def edit_stats_for_alignment(alig, empty_symbol=None):
    if len(alig) == 0:
        return 0, 0, 0, 0, 0

    alig = np.array(alig)
    ncor = np.sum(alig[:, 0] == alig[:, 1])
    ndel = np.sum(alig[:, 0] == np.array(empty_symbol))
    nphn = np.sum(alig[:, 1] != np.array(empty_symbol))
    nins = len(alig) - nphn
    nsub = nphn - ncor - ndel
    return nphn, ncor, nins, ndel, nsub


def levenshtein_distance_substring(source, target, sub_cost=1, ins_cost=1, del_cost=1):
    if len(target) > len(source):
        target, source = source, target

    target = np.array(target)
    dist = np.ones((1 + len(target) + 1)) * float('inf')
    dist[0] = 0
    for s in source:
        dist[1:-1] = np.minimum(dist[1:-1] + del_cost, dist[:-2] + (target != s) * sub_cost)

        for ii in range(len(dist) - 2):
            if dist[ii + 1] > dist[ii] + ins_cost:
                dist[ii + 1] = dist[ii] + ins_cost

        dist[-1] = np.minimum(dist[-1], dist[-2])

    return dist[-1]

def levenshtein_alignment_substring(source, target, sub_cost=1, ins_cost=1, del_cost=1, empty_symbol=None):
    swapped = False
    if len(target) > len(source):
        target, source = source, target
        swapped = True

    target = np.array(target)
    backtrack = np.ones((len(source) + 1, 1 + len(target) + 1))
    backtrack[0] = -1
    dist = np.ones((1 + len(target) + 1)) * float('inf')
    dist[0] = 0

    for ii, s in enumerate(source):
        cost4sub = dist[:-2] + (target != s) * sub_cost
        dist[1:-1] += del_cost
        where_sub = cost4sub < dist[1:-1]
        dist[1:-1][where_sub] = cost4sub[where_sub]
        backtrack[ii + 1, 1:-1][where_sub] = 0
        for jj in range(len(dist) - 2):
            if dist[jj + 1] > dist[jj] + ins_cost:
                dist[jj + 1] = dist[jj] + ins_cost
                backtrack[ii + 1, jj + 1] = -1

        if dist[-1] == dist[-2]:
            backtrack[ii + 1, -1] = 0
        elif dist[-1] > dist[-2]:
            dist[-1] = dist[-2]
            backtrack[ii + 1, -1] = -1
        else:
            pass

    suffix_beginning = backtrack.shape[0]
    if np.any(backtrack[:, -1] > 0):
        suffix_beginning = np.where(backtrack[:, -1] < 1)[0][-1] + 1

    backtrack = backtrack[:suffix_beginning, :-1]

    src_pos = backtrack.shape[0] - 1
    tar_pos = len(target)
    alig = []

    for char in source[suffix_beginning - 1:]:
        alig.append((char, empty_symbol))

    while tar_pos > 0 or src_pos > 0:
        where = backtrack[src_pos, tar_pos]
        if where >= 0:
            src_pos -= 1
        if where <= 0:
            tar_pos -= 1
        alig.insert(0, (empty_symbol if where < 0 else source[src_pos],
                        empty_symbol if where > 0 else target[tar_pos]))

    if swapped:
        alig = [(pair[1], pair[0]) for pair in alig]

    return alig
