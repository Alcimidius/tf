import os

DATASET_DIR = "archive"
TARGET_WORDS = ["yes", "no"]

label_to_int = {
    "yes": 0,
    "no": 1,
    "unknown": 2
}
# --------------------------------------------------------------------
# Read  validation/testing txts
# --------------------------------------------------------------------
with open(os.path.join(DATASET_DIR, "validation_list.txt")) as f:
    validation_files = set(line.strip() for line in f)

with open(os.path.join(DATASET_DIR, "testing_list.txt")) as f:
    testing_files = set(line.strip() for line in f)


def get_wav_files():

    train_wav_files = []
    validation_wav_files = []
    test_wav_files = []

    for folder in os.listdir(DATASET_DIR):

        folder_path = os.path.join(DATASET_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        # Skip metadata folders/files
        if folder.startswith("_"):
            continue

        for filename in os.listdir(folder_path):

            if not filename.endswith(".wav"):
                continue

            relative_path = f"{folder}/{filename}"
            full_path = os.path.join(folder_path, filename)

            if folder in TARGET_WORDS:
                label = folder
            else:
                label = "unknown"

            sample = (full_path, label)

            if relative_path in validation_files:
                validation_wav_files.append(sample)
            elif relative_path in testing_files:
                test_wav_files.append(sample)
            else:
                train_wav_files.append(sample)


    return train_wav_files, validation_wav_files, test_wav_files

def getTrainWavFiles(train_wav_files):

    yes_train = [x for x in train_wav_files if x[1] == "yes"]
    no_train = [x for x in train_wav_files if x[1] == "no"]
    unknown_train = [x for x in train_wav_files if x[1] == "unknown"]

    return yes_train, no_train, unknown_train

def getFileLabelTouple(train_wav_files,validation_wav_files,test_wav_files):

    train = [(path, label_to_int[label]) for path, label in train_wav_files]
    validation = [(path, label_to_int[label]) for path, label in validation_wav_files]
    test = [(path, label_to_int[label]) for path, label in test_wav_files]

    return train, validation, test