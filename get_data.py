if __name__ == "__main__":
    import argparse
    import requests
    import os
    from tqdm import tqdm

    from climatenet.utils.data import create_datasets
    from climatenet.utils.metrics import currScore

    # Get arguments to see if they passed in True or False
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true", help="If True, don't download the data")
    args = parser.parse_args()

    # If they passed in True, download the data
    if not args.no_download:
        print("Downloading data...")
        if not os.path.exists('data'):
            os.mkdir('data')
        if not os.path.exists('models'):
            os.mkdir('models')

        if not os.path.exists('data/train'):
            os.mkdir('data/train')
        if not os.path.exists('data/test'):
            os.mkdir('data/test')
        if not os.path.exists('models/original_baseline_model'):
            os.mkdir('models/original_baseline_model')
        if not os.path.exists('models/new_baseline_model'):
            os.mkdir('models/new_baseline_model')

        files = []
        url = 'https://portal.nersc.gov/project/ClimateNet/climatenet_new/'
        r = requests.get(url + "train/", allow_redirects=True)
        for line in tqdm(r.iter_lines(), desc="Getting train file names"):
            if "data-" in str(line):
                files.append((str(line).split('href="')[1]).split('">data')[0])

        for file in tqdm(files, desc="Downloading train files"):
            file_url = url + "train/" + file
            r = requests.get(file_url, allow_redirects=True)
            with open("data/train/" + file, 'wb') as f:
                f.write(r.content)

        files = []
        r = requests.get(url + "test/", allow_redirects=True)
        for line in tqdm(r.iter_lines(), desc="Getting test file names"):
            if "data-" in str(line):
                files.append((str(line).split('href="')[1]).split('">data')[0])

        for file in tqdm(files, desc="Downloading test files"):
            file_url = url + "test/" + file
            r = requests.get(file_url, allow_redirects=True)
            with open("data/test/" + file, 'wb') as f:
                f.write(r.content)

        with open("models/new_baseline_model/weights.pth", "wb") as f:
            r = requests.get(url + "model/weights_new.pth", allow_redirects=True)
            f.write(r.content)

        with open("models/original_baseline_model/weights.pth", "wb") as f:
            r = requests.get(url + "model/weights.pth", allow_redirects=True)
            f.write(r.content)
    print("Change data")
    currScore("data/currScore.json", scoring_method="mean")
    create_datasets(data_path="data/", ood=True)
    # Rename train folder to pretrainSet
    os.rename("data/train", "data/pretrainSet")
