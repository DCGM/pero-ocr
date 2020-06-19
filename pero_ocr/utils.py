from os.path import isabs, join


def compose_path(file_path, reference_path):
    if reference_path and not isabs(file_path):
        file_path = join(reference_path, file_path)
    return file_path
