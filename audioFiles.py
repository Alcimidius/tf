import os

DATASET_DIR = "archive"

# Read validation/testing lists
with open(os.path.join(DATASET_DIR, "validation_list.txt")) as f:
    validation_files = set(line.strip() for line in f)

with open(os.path.join(DATASET_DIR, "testing_list.txt")) as f:
    testing_files = set(line.strip() for line in f)

# Every folder is a class
CLASS_NAMES = sorted([
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
    and not folder.startswith("_")
])

label_to_int = {label: i for i, label in enumerate(CLASS_NAMES)}
int_to_label = {i: label for label, i in label_to_int.items()}

def get_wav_files():

    train_wav_files = []
    validation_wav_files = []
    test_wav_files = []

    for folder in CLASS_NAMES:

        folder_path = os.path.join(DATASET_DIR, folder)

        for filename in os.listdir(folder_path):

            if not filename.endswith(".wav"):
                continue

            relative_path = f"{folder}/{filename}"
            full_path = os.path.join(folder_path, filename)

            sample = (full_path, folder)

            if relative_path in validation_files:
                validation_wav_files.append(sample)
            elif relative_path in testing_files:
                test_wav_files.append(sample)
            else:
                train_wav_files.append(sample)

    return train_wav_files, validation_wav_files, test_wav_files


def getFileLabelTouple(train_wav_files, validation_wav_files, test_wav_files):

    train = [(path, label_to_int[label]) for path, label in train_wav_files]
    validation = [(path, label_to_int[label]) for path, label in validation_wav_files]
    test = [(path, label_to_int[label]) for path, label in test_wav_files]

    return train, validation, test