def save_transcriptions(path, transcriptions):
    with open(path, 'w', encoding='utf-8') as f:
        for key in transcriptions:
            f.write('{} {}\n'.format(key, transcriptions[key]))


def load_transcriptions(path):
    transcriptions = {}

    with open(path, "r", encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                image_id, transcription = parse_transcription_line(line)
            except ValueError:
                raise ValueError('Failed to parse line {} of file {}'.format(line_no, path))

            transcriptions[image_id] = transcription

    return transcriptions


def parse_transcription_line(line):
    image_id, transcription = line.split(" ", maxsplit=1)
    transcription = transcription.rstrip()
    return image_id, transcription
