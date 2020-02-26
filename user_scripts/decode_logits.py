#!/usr/bin/env python3

import argparse
import pickle
import time

from pero_ocr.decoding import confusion_networks
from pero_ocr.decoding.decoding_itf import prepare_dense_logits, construct_lm, get_ocr_charset, BLANK_SYMBOL
import pero_ocr.decoding.decoders as decoders
from pero_ocr.transcription_io import save_transcriptions


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--ocr-json', help='Path to OCR config', required=True)
    parser.add_argument('-k', '--beam-size', type=int, help='Width of the beam')
    parser.add_argument('-l', '--lm', help='File with a language model')
    parser.add_argument('--lm-scale', type=float, default=1.0, help='File with a language model')
    parser.add_argument('-g', '--greedy', action='store_true', help='Decode with a greedy decoder')
    parser.add_argument('--use-gpu', action='store_true', help='Make the decoder utilize a GPU')
    parser.add_argument('--model-eos', action='store_true', help='Make the decoder model end of sentences')
    parser.add_argument('-i', '--input', help='Pickled dictionary with names and sparse logits', required=True)
    parser.add_argument('-b', '--best', help='Where to store 1-best output', required=True)
    parser.add_argument('-p', '--confidence', help='Where to store posterior probability of the 1-best', required=True)
    parser.add_argument('-d', '--cn-best', help='Where to store 1-best from confusion network')

    args = parser.parse_args()

    return(args)


def main():
    args = parse_arguments()
    print(args)

    ocr_engine_chars = get_ocr_charset(args.ocr_json)

    if args.greedy:
        decoder = decoders.GreedyDecoder(ocr_engine_chars + [BLANK_SYMBOL])
    else:
        if args.lm:
            lm = construct_lm(args.lm)
        else:
            lm = None
        decoder = decoders.CTCPrefixLogRawNumpyDecoder(ocr_engine_chars + [BLANK_SYMBOL], k=args.beam_size, lm=lm, lm_scale=args.lm_scale, use_gpu=args.use_gpu)

    with open(args.input, 'rb') as f:
        complete_input = pickle.load(f)
    names = complete_input['names']
    logits = complete_input['logits']

    decodings = {}
    confidences = {}
    if args.cn_best:
        cn_decodings = {}

    t_0 = time.time()
    print('')
    for i, (name, sparse_logits) in enumerate(zip(names, logits)):
        time_per_line = (time.time() - t_0) / (i+1)
        nb_lines_ahead = len(names) - (i+1)
        print('\rProcessing {} [{}/{}, {:.2f}s/line, ETA {:.2f}s]'.format(name, i+1, len(names), time_per_line, time_per_line*nb_lines_ahead), end='')
        dense_logits = prepare_dense_logits(sparse_logits)

        if args.greedy:
            boh = decoder(dense_logits)
        else:
            boh = decoder(dense_logits, args.model_eos)
        one_best = boh.best_hyp()
        decodings[name] = one_best
        confidences[name] = boh.confidence()

        if args.cn_best:
            cn = confusion_networks.produce_cn_from_boh(boh)
            cn_decodings[name] = confusion_networks.best_cn_path(cn)
    print('')

    save_transcriptions(args.best, decodings)

    with open(args.confidence, 'w') as f:
        for name in decodings:
            f.write('{} {:.3f}\n'.format(name, confidences[name]))

    if args.cn_best:
        save_transcriptions(args.cn_best, cn_decodings)


if __name__ == "__main__":
    main()
