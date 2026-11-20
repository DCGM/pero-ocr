import pytest

from pero_ocr.decoding.bag_of_hypotheses import BagOfHypotheses


def bag(lm_weight):
    boh = BagOfHypotheses(lm_weight=lm_weight)
    boh.add('visual', -1.0, -10.0)   # better image score, worse LM score
    boh.add('lm', -3.0, -2.0)        # worse image score, better LM score
    return boh


def test_total_scores_use_lm_weight():
    assert bag(0.5).total_scores() == [-6.0, -4.0]


def test_best_hyp_with_lm_weight_one():
    assert bag(1.0).best_hyp() == 'lm'


@pytest.mark.xfail(reason='best_hyp() adds lm_sc with weight 1 instead of lm_weight (total_scores() uses lm_weight); '
                          'production decodes with LM_SCALE < 1, so the final pick over-weights the LM', strict=True)
@pytest.mark.parametrize('lm_weight', [0.0, 0.1])
def test_best_hyp_respects_lm_weight(lm_weight):
    assert bag(lm_weight).best_hyp() == 'visual'


def test_best_hyp_without_lm_scores():
    boh = BagOfHypotheses()
    boh.add('a', -2.0)
    boh.add('b', -1.0)
    assert boh.best_hyp() == 'b'
