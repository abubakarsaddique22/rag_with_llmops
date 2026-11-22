from pathlib import Path

from transcript_utils import load_and_clean_transcripts


def main():
    repo_root = Path(__file__).resolve().parent
    load_and_clean_transcripts(repo_root / "data" / "raw",
                               repo_root / "data" / "processed")


if __name__ == "__main__":
    main()
