import os
import shutil

input_root = '../LibriSpeech/train-clean-100/'
output_root = '../LibriSpeech/train-wav-100/'


def fetch_dist(directory: str) -> str:
    d = directory.split('/')
    d.pop(-1)
    return '/'.join(d)


for root, dirs, files in os.walk(input_root):
    for file in files:
        if file.endswith('.flac'):
            if file.startswith('._'):
                continue
            source = os.path.join(root, file)
            destination = (source
                           .replace('.flac', '.wav')
                           .replace('train-clean-100', 'train-wav-100'))
            try:
                shutil.copytree(
                    src=fetch_dist(directory=source),
                    dst=fetch_dist(directory=destination)
                )
            except FileExistsError:
                print('Directory "{}" already exists.'.format(destination))

            shutil.copyfile(
                src=source,
                dst=destination
            )
            os.remove(destination.replace('.wav', '.flac'))
            print(f"Saved: {destination}")
