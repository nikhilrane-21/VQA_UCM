import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from tqdm import tqdm
from dataloader import *
from model import *
from torchmetrics import Accuracy

from transformers import logging
logging.set_verbosity_warning()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
disk_path = "../../../media/lscsc/nas/suiyuan"


# Arg-parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default= disk_path + '/datasets', help='input directory for visual question answering.')
parser.add_argument('--log_dir', type=str, default='./logs', help='directory for logs.')
parser.add_argument('--model_dir', type=str, default= disk_path +'/checkpoints', help='directory for saved models.')
parser.add_argument('--max_qst_length', type=int, default=30, help='maximum length of question, the length in the VQA dataset = 26.')
parser.add_argument('--max_num_ans', type=int, default=10, help='maximum number of answers.')
parser.add_argument('--embed_size', type=int, default=512, help='embedding size of feature vector for both image and question.')
parser.add_argument('--word_embed_size', type=int, default=300, help='embedding size of word used for the input in the LSTM.')
# parser.add_argument('--num_layers', type=int, default=2, help='number of layers of the RNN(LSTM).')
# parser.add_argument('--hidden_size', type=int, default=512, help='hidden_size in the LSTM.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training.')
parser.add_argument('--step_size', type=int, default=10, help='period of learning rate decay.')
parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay.')
parser.add_argument('--num_epochs', type=int, default=15, help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size.')
# parser.add_argument('--num_workers', type=int, default=8, help='number of processes working on cpu.')
parser.add_argument('--save_step', type=int, default=15, help='save step of model.')
args = parser.parse_args()

# Load the dataset (dataloader)
ucm_vqa_train_dataloader, \
ucm_vqa_eval_dataloader, \
ucm_vqa_test_dataloader, \
ans_dict = construct_data_loader(batch_size=args.batch_size)  # return the dataloader of {train, eval, test}

def train():
    # Load the dataset
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

    # Load the model
    model = VqaModel(embed_size=args.embed_size, num_labels=len(ans_dict)).to(device)

    # Criterion, optimizer(params)
    criterion = nn.CrossEntropyLoss()
    params = list(model.img_encoder.fc.parameters()) + list(model.classifier.parameters())  # indicate what parameters need to be updated
    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in tqdm(range(args.num_epochs)):
        running_loss = 0
        model.train()
        batch_step_size = len(ucm_vqa_train_set) / args.batch_size

        # Initialize metric
        metric = Accuracy().to(device)
        scheduler.step()

        for i_batch, sample_batched in enumerate(ucm_vqa_train_dataloader):
            images = sample_batched['image'].to(device)
            questions = sample_batched['question'].to(device)
            labels = sample_batched['answer'].to(device)
            optimizer.zero_grad()
            outputs = model(images, questions)  # [batch_size, ans_vocab_size]
            _, preds = torch.max(outputs, 1)  # [batch_size]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Update metric
            acc = metric(preds, labels)
            # Print the average loss in a mini-batch.
            if i_batch % 15 == 0:
                print('| Training SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Accuracy on batch {}: {:.4f}'
                      .format(epoch + 1, args.num_epochs, i_batch, int(batch_step_size), loss.item(), i_batch, acc))
            running_loss += loss.item()
        # print the average loss and accuracy in an epoch.
        acc = metric.compute()
        epoch_loss = running_loss / batch_step_size
        print('| Training SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Accuracy on all data: {:.4f} \n'
            .format(epoch + 1, args.num_epochs, epoch_loss, acc))
        with open(os.path.join(args.model_dir, 'epoch-{:02d}-batchsize-{:02d}-evaluation.txt'.format(args.num_epochs, args.batch_size)), "a") as file:
            file.write('| Training SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Accuracy on all data: {:.4f} \n'
            .format(epoch + 1, args.num_epochs, epoch_loss, acc))

        # Save the model check points
        if (epoch + 1) % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'epoch-{:02d}-batchsize-{:02d}.pt'.format(epoch + 1, args.batch_size)))



def eval():
    # load the model (init the model and load states dict)
    model = VqaModel(embed_size=args.embed_size, num_labels=len(ans_dict)).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'epoch-{:02d}-batchsize-{:02d}.pt'.format(args.num_epochs, args.batch_size))))

    with torch.no_grad():
        metric = Accuracy().to(device)
        for i_batch, sample_batched in enumerate(ucm_vqa_eval_dataloader):
            images = sample_batched['image'].to(device)
            questions = sample_batched['question'].to(device)
            labels = sample_batched['answer'].int().to(device)
            outputs = model(images, questions)  # [batch_size, ans_vocab_size]
            _, preds = torch.max(outputs, 1)  # [batch_size]
            # update metric
            metric(preds, labels)
        acc = metric.compute()
        print('| Evaluation SET | Accuracy: {:.4f} \n'.format(acc))
        with open(os.path.join(args.model_dir, 'epoch-{:02d}-batchsize-{:02d}-evaluation.txt'.format(args.num_epochs, args.batch_size)), "a") as file:
            file.write(f"Overall Accuracy on Evaluation Dataset is {acc}" + "\n")


def main():
    with open(os.path.join(args.model_dir, 'epoch-{:02d}-batchsize-{:02d}-evaluation.txt'.format(args.num_epochs, args.batch_size)), "a") as file:
        for k in args.__dict__:
            file.write(k + ": " + str(args.__dict__[k]) + " ")
        file.write("\n")
    train()
    eval()


main()
