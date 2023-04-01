def save_transcriptions(path, transcriptions):
    with open(path, 'w') as f:
        for key in transcriptions:
            f.write('{} {}\n'.format(key, transcriptions[key]))


def load_transcriptions(path, embeddings_in_transcripts):
    transcriptions = {}

    with open(path, "r") as f:
        for line_no, line in enumerate(f):
            if len(line) == 0:
                continue

            try:
                image_id, _, transcription = parse_transcription_line(line, embeddings_in_transcripts)
            except ValueError:
                raise ValueError('Failed to parse line {} of file {}'.format(line_no, path))

            transcriptions[image_id] = transcription

    return transcriptions


def parse_transcription_line(line, embeddings_in_transcripts):
    if embeddings_in_transcripts:
        image_id, embedding, transcription = line.split(" ", maxsplit=2)
    else:
        image_id, transcription = line.split(" ", maxsplit=1)
        embedding = None

    if transcription[-1] == '\n':
        transcription = transcription[:-1]

    return image_id, embedding, transcription
