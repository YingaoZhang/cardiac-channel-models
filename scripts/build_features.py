from pathlib import Path
from cardiac_channels.features import build_feature_file


def main():
    root = Path(__file__).resolve().parents[1]
    for channel in ('herg', 'cav1.2', 'nav1.5'):
        destination = root / 'data' / 'features' / f'{channel}_features.npz'
        build_feature_file(root / 'data' / 'dev' / f'{channel}_dev.csv', destination)
        print(destination, flush=True)


if __name__ == '__main__':
    main()
