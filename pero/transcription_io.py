def save_transcriptions(path, transcriptions):
    with open(path, 'w') as f:
        for key in transcriptions:
            f.write('{} {}\n'.format(key, transcriptions[key]))


def load_transcriptions(path):
    transcriptions = {}

    with open(path, "r") as f:
        for line_no, line in enumerate(f):
            if len(line) == 0:
                continue

            try:
                image_id, transcription = parse_transcription_line(line)
            except ValueError:
                raise ValueError('Failed to parse line {} of file {}'.format(line_no, path))

            transcriptions[image_id] = transcription

    return transcriptions


def parse_transcription_line(line):
    image_id, transcription = line.split(" ", maxsplit=1)

    if transcription[-1] == '\n':
        transcription = transcription[:-1]

    return image_id, transcription
