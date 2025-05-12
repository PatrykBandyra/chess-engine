import os
import zipfile

import requests


def main() -> None:
    print('Fetching dataset...')
    kaggle_url = 'https://www.kaggle.com/api/v1/datasets/download/ronakbadhe/chess-evaluations'
    zip_file_name = 'archive.zip'

    # Download
    with requests.get(kaggle_url, stream=True, allow_redirects=True) as r:
        with open(zip_file_name, 'wb') as f:
            chunk_size = 8192  # bytes
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)

    # Unzip
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall()

    # Delete trash files
    os.remove(zip_file_name)
    os.remove('random_evals.csv')
    os.remove('tactic_evals.csv')

    print('Dataset fetched')


if __name__ == "__main__":
    main()
