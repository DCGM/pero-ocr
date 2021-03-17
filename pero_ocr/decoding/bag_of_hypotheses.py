from collections import namedtuple
import math
try:
    from scipy.misc import logsumexp
except:
    from scipy.special import logsumexp

Hypothese = namedtuple('Hypothese', 'transcript vis_sc lm_sc')


class BagOfHypotheses:
    def __init__(self, lm_weight=1.0):
        self._hyps = []
        self.lm_weight = lm_weight

    def add(self, transcript, visual_sc, lm_sc=None):
        self._hyps.append(Hypothese(transcript, visual_sc, lm_sc))

    def sort(self):
        self._hyps.sort(key=lambda hyp: hyp.vis_sc, reverse=True)

    def __str__(self):
        longest_len = max(len(hyp.transcript) for hyp in self)

        string = ""
        str_fmt = "{:" + str(longest_len) + "}"
        for i, hyp in enumerate(self):
            total_fmt = "{} " + str_fmt + " {:5.1f} {:5.1f} \n"
            string += total_fmt.format(i, "'{}'".format(hyp.transcript), hyp.vis_sc, hyp.lm_sc)

        return string

    def __iter__(self):
        for hyp in self._hyps:
            yield hyp

    def __len__(self):
        return len(self._hyps)

    def total_scores(self):
        try:
            return [hyp.vis_sc + self.lm_weight * hyp.lm_sc for hyp in self._hyps]
        except TypeError:
            return [hyp.vis_sc for hyp in self._hyps]

    def posteriors(self):
        total_scores = self.total_scores()
        total_prob = logsumexp(total_scores)
        return [s - total_prob for s in total_scores]

    def confidence(self):
        posteriors = self.posteriors()
        return math.exp(max(posteriors))

    def transcript_confidence(self, transcript):
        posteriors = self.posteriors()

        for i, hyp in enumerate(self._hyps):
            if hyp.transcript == transcript:
                return math.exp(posteriors[i])

        return 0.0  # Transcript not found in the bag of hypotheses

    def best_hyp(self):
        return max(self._hyps, key=lambda hyp: hyp.vis_sc + (hyp.lm_sc if hyp.lm_sc is not None else 0)).transcript
