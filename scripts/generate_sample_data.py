from f1_podium.config import SETTINGS
from f1_podium.data import write_sample_tables


if __name__ == "__main__":
    for output in write_sample_tables(SETTINGS.raw_dir, SETTINGS.seed):
        print(f"Wrote {output}")

