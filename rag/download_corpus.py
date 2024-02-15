import argparse

import os

import wget

BASE_URL = "https://dl.fbaipublicfiles.com/atlas"


def maybe_download_file(source, target):
    if not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"Downloading {source} to {target}")
        wget.download(source, out=str(target))
        print()


def get_s3_path(path):
    return f"{BASE_URL}/{path}"


def get_download_path(output_dir, path):
    return os.path.join(output_dir, path)


AVAILABLE_CORPORA = {
    "corpora/wiki/enwiki-dec2017": {
        "corpus": "corpora/wiki/enwiki-dec2017",
        "description": "Wikipedia dump from Dec 2017, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-dec2018": {
        "corpus": "corpora/wiki/enwiki-dec2018",
        "description": "Wikipedia dump from Dec 2018, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-aug2019": {
        "corpus": "corpora/wiki/enwiki-aug2019",
        "description": "Wikipedia dump from Aug 2019, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-dec2020": {
        "corpus": "corpora/wiki/enwiki-dec2020",
        "description": "Wikipedia dump from Dec 2020, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
    "corpora/wiki/enwiki-dec2021": {
        "corpus": "corpora/wiki/enwiki-dec2021",
        "description": "Wikipedia dump from Dec 2021, preprocessed into passages",
        "files": ["text-list-100-sec.jsonl", "infobox.jsonl"],
    },
}


def _helpstr():
    helpstr = "The following corpora are available for download: "
    for m in AVAILABLE_CORPORA.values():
        helpstr += f'\nCorpus name: {m["corpus"]:<30} Description: {m["description"]}'
    helpstr += "\ndownload by passing --corpus {corpus name}"
    return helpstr


def main(output_directory, requested_corpus):
    AVAILABLE_CORPORA[requested_corpus]
    for filename in AVAILABLE_CORPORA[requested_corpus]["files"]:
        path = f"{requested_corpus}/{filename}"
        source = get_s3_path(path)
        target = get_download_path(output_directory, path)
        maybe_download_file(source, target)


if __name__ == "__main__":
    help_str = _helpstr()
    choices = list(AVAILABLE_CORPORA.keys())
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./data",
        help="Path to the file to which the dataset is written.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        choices=choices,
        help=help_str,
    )
    args = parser.parse_args()
    main(args.output_directory, args.corpus)