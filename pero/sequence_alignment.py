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
        if where >= 0: src_pos -= 1
        if where <= 0: tar_pos -= 1
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
        if where >= 0: src_pos -= 1
        if where <= 0: tar_pos -= 1
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
