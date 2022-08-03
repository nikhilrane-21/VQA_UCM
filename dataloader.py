import os
import re

import numpy as np
import torch
import torch.nn.functional as Fun
import matplotlib.pyplot as plt
from skimage import transform
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

dict_path = "../../../media/lscsc/nas/suiyuan/datasets"


ucm_image_file_dir = dict_path + "/ucm_images"
ucm_vqa_dir = dict_path + "/ucm_vqa.txt"  # file path contains the question and answer
ucm_images = dict_path + "/ucm_images.txt"  # file path contains the image path

def read_label_txt(data_dir):
    """
    load the questions and the corresponding answers from the data_dir
    :param data_dir:
    :return: question_list, answer_list
    """
    qst_list = []
    ans_list = []
    with open(data_dir) as f:
        qa_list = [line.strip().split("?") for line in f.readlines()]
    for i in range(len(qa_list)):
        ans = qa_list[i][-1]
        ans_list.append(ans)
        qst = qa_list[i][0].split(":")[-1]
        qst_list.append(qst)
    return qst_list, ans_list

def load_str_list(fname):
    """
    Load the str from the txt file
    :param fname:
    :return:
    """
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


class VocabDict:
    def __init__(self, vocab_file):
        self.word_list = vocab_file
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.word2idx_dict["<unk>"] = -1
        self.word2idx_dict["<pad>"] = 55360
        self.vocab_size = len(self.word_list)
        self.unk2idx = self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict else None

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.unk2idx is not None:
            return self.unk2idx
        else:
            raise ValueError(f'word {w} is not in the dictionary, while <unk> is not contained in the dict')

    def tokenize(sent):
        tokens = SENTENCE_SPLIT_REGEX.split(sent.lower())
        tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
        return tokens

    def tokenize_and_index(self, sent):
        idx_l = [self.word2idx(w) for w in self.tokenize(sent)]
        return idx_l

class UCM_RS(Dataset):
    """
    UCM RS dataset.
    """

    def __init__(self, qa_file, img_dir, ptm="bert-base-uncased", max_qst_length=25, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the ucm_images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.qa_file = qa_file
        self.qst, self.ans = read_label_txt(self.qa_file)
        # self.qst_vocab = VocabDict(self.qst)
        # self.ans_vocab = VocabDict(self.ans)
        self.ans_dict = {w: n_w for n_w, w in enumerate(set(self.ans))} # save the answer dict

        # ltms
        self.tokenizer = AutoTokenizer.from_pretrained(ptm)

        self.max_qst_length = max_qst_length

        with open(self.img_dir) as f:
            self.img = [line.strip() for line in f.readlines()]
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        """
        to support the indexing such that dataset[i] can be used to get i-th sample.
        * leave the reading of ucm_images to __getitem__ to keep the memory efficient because all the ucm_images are not stored in the memory at once but read as required.
        :param idx:
        :return:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = plt.imread(self.img[idx])

        qst = self.qst[idx]
        ans = self.ans[idx]

        qst2idc = self.tokenizer(qst)['input_ids']

        ans2idc = torch.tensor(self.ans_dict[ans]).type(torch.LongTensor)
        # ans2idc = Fun.one_hot(ans2idc, num_classes=len(set(self.ans))-1).float()

        sample = {'image': img, 'question': qst2idc, 'answer': ans2idc}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_ans_dict(self):
        return self.ans_dict

class Rescale(object):
    """
    Rescale the image in a sample to a given size
    output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, qst, ans = sample['image'], sample['question'], sample['answer']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'question': qst, 'answer':ans}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, qst, ans = sample['image'], sample['question'], sample['answer']

        h, w = torch.tensor(image.shape[:2])
        new_h, new_w = torch.tensor(self.output_size)

        top = h.random_(0, h - new_h)
        left = w.random_(0, w - new_w)

        img = image[top: top + new_h,
                      left: left + new_w]

        return {'image': img, 'question': qst, 'answer':ans}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, qst, ans = sample['image'], sample['question'], sample['answer']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'question': torch.tensor(qst), 'answer':torch.tensor(ans)}

def construct_data_loader(batch_size, shuffle=True, num_workers=0):

    transformed_dataset = UCM_RS(
        qa_file=ucm_vqa_dir,
        img_dir=ucm_images,
        transform=transforms.Compose([Rescale(225), RandomCrop(128), ToTensor()])
    )

    ucm_vqa_train_set, ucm_vqa_val_set, ucm_vqa_test_set = torch.utils.data.random_split(
        transformed_dataset,
        [
            int(len(transformed_dataset) * 0.7),
            int(len(transformed_dataset) * 0.2),
            int(len(transformed_dataset) * 0.1),
        ],
        generator=torch.Generator().manual_seed(1234)  # seed 1234
    )

    # Batching the datasets, shuffling the datasets, load the datasets in parallel using multiprocessing workers
    # using DataLoader
    ucm_vqa_train_dataloader = DataLoader(ucm_vqa_train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    ucm_vqa_eval_dataloader = DataLoader(ucm_vqa_val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    ucm_vqa_test_dataloader = DataLoader(ucm_vqa_test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return ucm_vqa_train_dataloader, \
           ucm_vqa_eval_dataloader, \
           ucm_vqa_test_dataloader, \
        transformed_dataset.get_ans_dict()

def test(batch_size=4):
    ucm_vqa_train_dataloader, \
    ucm_vqa_eval_dataloader, \
    ucm_vqa_test_dataloader, \
    ans_dict = construct_data_loader(batch_size=batch_size)  # return the dataloader of {train, eval, test}
    print(next(iter(ucm_vqa_train_dataloader)))

# test()
