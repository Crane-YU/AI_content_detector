import os

from datasets import load_dataset
from transformers import RobertaTokenizer

from utils import parse_arge


# laod huggingface dataset
def load_huggingface_dataset(data_id="yelp_review_full"):
    dataset_train = load_dataset(data_id, split="train")
    dataset_test = load_dataset(data_id, split="test").shard(num_shards=2, index=0)
    dataset_val = load_dataset(data_id, split="test").shard(num_shards=2, index=1)

    return dataset_train, dataset_test, dataset_val


# create fake label
# if label score is low, we take it as human generated comments
# otherwise machine generated
def make_fake_label(example):
    if example["label"] < 2:
        example["label"] = 1  # human generated
    else:
        example["label"] = 0  # machine generated
    return example


# change the label according to make_fake_label
def dataset_label_change(dataset_train, dataset_test, dataset_val):
    dataset_train = dataset_train.map(make_fake_label)
    dataset_test = dataset_test.map(make_fake_label)
    dataset_val = dataset_val.map(make_fake_label)
    return dataset_train, dataset_test, dataset_val


# create small subsets for faster fine-tune
def create_small_dataset(args, dataset_train, dataset_test, dataset_val):
    dataset_train = dataset_train.shuffle(seed=args.seed).select(range(128))
    dataset_test = dataset_test.shuffle(seed=args.seed).select(range(32))
    dataset_val = dataset_val.shuffle(seed=args.seed).select(range(32))
    return dataset_train, dataset_test, dataset_val


# tokenization
def tokenize_func(args, dataset_example):
    tokenizer = RobertaTokenizer.from_pretrained(args.saved_model_dir)
    return tokenizer(
        dataset_example["text"], padding="max_length", truncation=True, max_length=256
    )


def dataset_tokenization(dataset_train, dataset_test, dataset_val):
    dataset_train = dataset_train.map(tokenize_func, batched=True)
    dataset_test = dataset_test.map(tokenize_func, batched=True)
    dataset_val = dataset_val.map(tokenize_func, batched=True)
    return dataset_train, dataset_test, dataset_val


# save the dataset to disk
def save_dataset(args, dataset_train, dataset_test, dataset_val):
    dataset_train.save_to_disk(os.path.join(args.datafolder, "train"))
    dataset_test.save_to_disk(os.path.join(args.datafolder, "test"))
    dataset_val.save_to_disk(os.path.join(args.datafolder, "val"))


# run the data preprocess steps
def run_data_preprocess(args):
    dataset_train, dataset_test, dataset_val = load_huggingface_dataset()
    dataset_train, dataset_test, dataset_val = dataset_label_change(
        dataset_train, dataset_test, dataset_val
    )
    dataset_train, dataset_test, dataset_val = create_small_dataset(
        args, dataset_train, dataset_test, dataset_val
    )
    dataset_train, dataset_test, dataset_val = dataset_tokenization(
        args, dataset_train, dataset_test, dataset_val
    )
    save_dataset(args, dataset_train, dataset_test, dataset_val)


if __name__ == "__main__":
    args = parse_arge()
    run_data_preprocess(args)
