from collections import namedtuple
import math
try:
    from scipy.misc import logsumexp
except:
    from scipy.special import logsumexp

Hypothese = namedtuple('Hypothese', 'transcript vis_sc lm_sc')


class BagOfHypotheses:
    def __init__(self):
        self._hyps = []
        self._posteriors = []

    def add(self, transcript, visual_sc, lm_sc=None):
        self._hyps.append(Hypothese(transcript, visual_sc, lm_sc))

    def sort(self):
        self._hyps.sort(key=lambda hyp: hyp.vis_sc, reverse=True)

    def __str__(self):
        self.recompute_posteriors()

        longest_len = max(len(hyp.transcript) for hyp in self)

        string = ""
        str_fmt = "{:" + str(longest_len) + "}"
        for i, hyp in enumerate(self):
            total_fmt = "{} " + str_fmt + " {:5.1f} {:5.1f} \n"
            string += total_fmt.format(i, "'{}'".format(hyp.transcript), hyp.vis_sc, hyp.lm_sc)

        return string

    def get_lm_scores(self, lm):
        for i, hyp in enumerate(self._hyps):
            if len(hyp.transcript) == 0:
                lm_score = 1.0
            else:
                lm_score = lm.single_sentence_nll(hyp.transcript, '</s>')
            self._hyps[i] = Hypothese(hyp.transcript, hyp.vis_sc, -lm_score)

    def __iter__(self):
        for hyp in self._hyps:
            yield hyp

    def __len__(self):
        return len(self._hyps)

    def recompute_posteriors(self):
        total_prob = logsumexp([hyp.vis_sc for hyp in self._hyps])
        self._posteriors = [hyp.vis_sc - total_prob for hyp in self._hyps]

    def confidence(self):
        self.recompute_posteriors()
        return math.exp(max(self._posteriors))

    def best_hyp(self):
        return max(self._hyps, key=lambda hyp: hyp.vis_sc + (hyp.lm_sc if hyp.lm_sc is not None else 0)).transcript
