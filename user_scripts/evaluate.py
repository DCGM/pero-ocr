#!/usr/bin/env python3

import sys
import argparse
from pero_ocr.transcription_io import load_transcriptions
from pero_ocr.error_summary import ErrorsSummary
from enum import Enum


class LineTranscription:
    def __init__(self, filename, ground_truth, transcription):
        self.filename = filename
        self.ground_truth = ground_truth
        self.transcription = transcription

        self.char_errors = ErrorsSummary.from_lists(list(ground_truth), list(transcription))
        self.word_error = ErrorsSummary.from_lists(ground_truth.split(), transcription.split())

    def __str__(self, human_readable=False):
        if human_readable:
            fmt = "{filename}\n\tREF: {ground_truth}\n\tHYP: {transcription}\n\tCHAR {cer}\n\tWORD {wer}"
        else:
            fmt = "{filename}\t{ground_truth}\t{transcription}\tCHAR {cer}\tWORD {wer}"

        return fmt.format(
            filename=self.filename,
            transcription=self.transcription,
            ground_truth=self.ground_truth,
            cer=self.char_errors,
            wer=self.word_error)


class MasterData(Enum):
    ground_truth = 1
    transcriptions = 2


def match_lines(transcriptions, ground_truths, master_data=MasterData.ground_truth, verbose=True):
    lines = []

    if master_data == MasterData.ground_truth:
        master = ground_truths
        slave = transcriptions
    else:
        master = transcriptions
        slave = ground_truths

    for image_id in master:
        master_item = master[image_id]

        try:
            slave_item = slave[image_id]
        except KeyError:
            if verbose:
                print("Missing transcription for '{id}'.".format(id=image_id))
            slave_item = ""

        if master_data == MasterData.ground_truth:
            line = LineTranscription(image_id, master_item, slave_item)
        else:
            line = LineTranscription(image_id, slave_item, master_item)

        lines.append(line)

    return lines


def sort(lines):
    return sorted(lines, key=lambda line: line.char_errors.error_rate, reverse=True)


def save(lines, path, human_readable):
    with open(path, "w") as f:
        for line in lines:
            f.write(line.__str__(human_readable) + "\n")


def nb_symbol_errors(sym, sym_confusions):
    return sum(count for s, count in sym_confusions.items() if s != sym)


def save_confusions(summary, path):
    with open(path, 'w') as f:
        f.write('There was a total of {} lines. Out of this:\n'.format(summary.nb_lines_summarized))
        f.write('\t{} ({:.1f} %) ended with pure deletion\n'.format(
            summary.ending_errors.pure_deletions,
            100.0 * summary.ending_errors.pure_deletions / summary.nb_lines_summarized
        ))
        f.write('\t{} ({:.1f} %) ended with mix of deletion and substitution\n'.format(
            summary.ending_errors.mixed_deletions,
            100.0 * summary.ending_errors.mixed_deletions / summary.nb_lines_summarized
        ))
        f.write('\t{} ({:.1f} %) ended with pure insertion\n'.format(
            summary.ending_errors.pure_insertions,
            100.0 * summary.ending_errors.pure_insertions / summary.nb_lines_summarized
        ))
        f.write('\t{} ({:.1f} %) ended with mix of insertion and substitution\n'.format(
            summary.ending_errors.mixed_insertions,
            100.0 * summary.ending_errors.mixed_insertions / summary.nb_lines_summarized
        ))
        f.write('\t{} ({:.1f} %) ended with pure substitution\n'.format(
            summary.ending_errors.pure_substitutions,
            100.0 * summary.ending_errors.pure_substitutions / summary.nb_lines_summarized
        ))
        f.write('\t{} ({:.1f} %) ended with a proper match\n'.format(
            summary.ending_errors.correct,
            100.0 * summary.ending_errors.correct / summary.nb_lines_summarized
        ))
        f.write('\n')

        for ref_sym, sym_confusions in sorted(summary.confusions.items(), key=lambda kv: nb_symbol_errors(kv[0], kv[1]), reverse=True):
            nb_occurances = sum(sym_confusions.values())
            nb_errors = nb_symbol_errors(ref_sym, sym_confusions)

            confusion_elements = [(hyp_sym, count) for hyp_sym, count in sym_confusions.items() if hyp_sym != ref_sym]
            confusion_elements = sorted(confusion_elements, key=lambda s_and_c: s_and_c[1], reverse=True)
            confusion_elements = ["{} ({})".format(repr(hyp_sym), count) for hyp_sym, count in confusion_elements]

            f.write('{} {} times wrong ({:.1f} %): {}\n'.format(
                ref_sym,
                nb_errors, 100.0*nb_errors/nb_occurances,
                ', '.join(confusion_elements)
            ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', type=str, required=True)
    parser.add_argument('--transcriptions', type=str, required=True)
    parser.add_argument('--output', type=str, required=False, default=None)
    parser.add_argument('--char-confusion', type=str, required=False, default=None)
    parser.add_argument('--word-confusion', type=str, required=False, default=None)
    parser.add_argument('--human-readable', action='store_true', required=False,
                        help='make per-line output easier to process by humans')
    args = parser.parse_args()

    ground_truths = load_transcriptions(args.ground_truth)
    transcriptions = load_transcriptions(args.transcriptions)

    lines = match_lines(transcriptions, ground_truths)
    char_summary = ErrorsSummary.aggregate([line.char_errors for line in lines])
    word_summary = ErrorsSummary.aggregate([line.word_error for line in lines])

    print("CER:", char_summary)
    print("WER:", word_summary)

    if args.output is not None:
        save(sort(lines), args.output, args.human_readable)

    if args.char_confusion is not None:
        save_confusions(char_summary, args.char_confusion)

    if args.word_confusion is not None:
        save_confusions(word_summary, args.word_confusion)

    return 0


if __name__ == "__main__":
    sys.exit(main())
