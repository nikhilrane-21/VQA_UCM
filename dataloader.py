import os
import torch
import matplotlib.pyplot as plt
from skimage import transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ucm_image_file_dir = "data/images"
ucm_vqa_dir = "data/ucm_vqa.txt"  # file path contains the question and answer
ucm_images = "data/ucm_images.txt"  # file path contains the image path


def read_label_txt(data_dir):
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

class UCM_RS(Dataset):
    """
    UCM RS dataset.
    """

    def __init__(self, qa_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.qa_file = qa_file
        self.qst, self.ans = read_label_txt(self.qa_file)
        with open(self.img_dir) as f:
            self.img = [line.strip() for line in f.readlines()]
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        """
        to support the indexing such that dataset[i] can be used to get i-th sample.
        * leave the reading of images to __getitem__ to keep the memory efficient because all the images are not stored in the memory at once but read as required.
        :param idx:
        :return:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = plt.imread(self.img[idx])
        qst = self.qst[idx]
        ans = self.ans[idx]
        sample = {'image': img, 'question': qst, 'answer': ans}

        if self.transform:
            sample = self.transform(sample)

        return sample

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
                'question': qst, 'answer':ans}

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

    # Batching the data, shuffling the data, load the data in parallel using multiprocessing workers
    # using DataLoader
    ucm_vqa_train_dataloader = DataLoader(ucm_vqa_train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    ucm_vqa_eval_dataloader = DataLoader(ucm_vqa_val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    ucm_vqa_test_dataloader = DataLoader(ucm_vqa_test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return ucm_vqa_train_dataloader, \
           ucm_vqa_eval_dataloader, \
           ucm_vqa_test_dataloader
