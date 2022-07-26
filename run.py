from dataloader import *
from model import VqaModel
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from transformers import logging
logging.set_verbosity_warning()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')
# Arg-parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='./datasets', help='input directory for visual question answering.')
parser.add_argument('--log_dir', type=str, default='./logs', help='directory for logs.')
parser.add_argument('--model_dir', type=str, default='./checkpoints', help='directory for saved models.')
parser.add_argument('--max_qst_length', type=int, default=30, help='maximum length of question, the length in the VQA dataset = 26.')
parser.add_argument('--max_num_ans', type=int, default=10, help='maximum number of answers.')
parser.add_argument('--embed_size', type=int, default=1024, help='embedding size of feature vector for both image and question.')
parser.add_argument('--word_embed_size', type=int, default=300, help='embedding size of word used for the input in the LSTM.')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers of the RNN(LSTM).')
parser.add_argument('--hidden_size', type=int, default=512, help='hidden_size in the LSTM.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training.')
parser.add_argument('--step_size', type=int, default=10, help='period of learning rate decay.')
parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay.')
parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs.')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size.')
parser.add_argument('--num_workers', type=int, default=8, help='number of processes working on cpu.')
parser.add_argument('--save_step', type=int, default=1, help='save step of model.')
args = parser.parse_args()

# Load the dataset (dataloader)
ucm_vqa_train_dataloader, \
ucm_vqa_eval_dataloader, \
ucm_vqa_test_dataloader, \
ans_dict = construct_data_loader(batch_size=args.batch_size)  # return the dataloader of {train, eval, test}

def show_example_batch(num_to_end):
    for i_batch, sample_batched in enumerate(ucm_vqa_train_dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['question'].size(), sample_batched['answer'].size())
        if i_batch == num_to_end:
            break

def train():
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

    model = VqaModel(
        embed_size=128, # same as the Bert(ptm)
        ans_vocab_size=len(ans_dict)-1
    ).to(device)

    # print(model)

    criterion = nn.CrossEntropyLoss()

    # params = list(model.img_encoder.fc.parameters()) \
    #     + list(model.qst_encoder.parameters()) \
    #     + list(model.fc1.parameters()) \
    #     + list(model.fc2.parameters()) # indicate what parameters need to be updated

    params = list(model.img_encoder.fc.parameters()) \
            + list(model.fc.parameters())  # indicate what parameters need to be updated

    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):
        model.train()

        batch_step_size = len(ucm_vqa_train_set) / args.batch_size
        acc = 0.0
        corr = 0.0
        running_loss = 0

        for i_batch, sample_batched in enumerate(ucm_vqa_train_dataloader):
            image = sample_batched['image'].to(device)
            question = sample_batched['question'].to(device)
            label = sample_batched['answer'].to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                output = model(image, question) # [batch_size, ans_vocab_size]
                # _, pred = torch.max(output, 1) # [batch_size] (without onehot)
                # one-hot
                pred = output
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

            # scheduler.step()

            # Print the average loss in a mini-batch.
            if i_batch % 1 == 0:
                print('| Training SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                      .format(epoch + 1, args.num_epochs, i_batch, int(batch_step_size),
                              loss.item()))
            running_loss += loss.item()
            corr += torch.stack([(torch.argmax(label) == torch.argmax(pred).to(device))]).any(dim=0).sum()
        # Print the average loss and accuracy in an epoch.
        epoch_loss = running_loss / batch_step_size
        acc = corr.double() / len(ucm_vqa_eval_dataloader)  # multiple choice

        print('| Training SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Accuracy: {:.4f} \n'
              .format(epoch + 1, args.num_epochs, epoch_loss, acc))

        # Save the model check points
        if (epoch + 1) % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model-epoch-{:02d}.ckpt'.format(epoch + 1)))

def eval():
    model = VqaModel(
        embed_size=2096,  # same as the Bert(ptm)
        ans_vocab_size=len(ans_dict)
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model-epoch-{:02d}.ckpt'.format(args.num_epochs))))

    with torch.no_grad():
        acc = 0.0
        corr = 0.0
        for i_batch, sample_batched in enumerate(ucm_vqa_eval_dataloader):
            image = sample_batched['image'].to(device)
            question = sample_batched['question'].to(device)
            label = sample_batched['answer'].to(device)

            output = model(image, question) # [batch_size, ans_vocab_size]
            _, pred = torch.max(output, 1) # [batch_size]

            corr += torch.stack([(label == pred.to(device))]).any(dim=0).sum()
        acc = corr.double() / len(ucm_vqa_eval_dataloader)  # multiple choice
        print('| Evaluation SET | Accuracy: {:.4f} \n'
              .format(acc))
        with open(os.path.join(args.model_dir, 'model-epoch-{:02d}-evaluation.txt'.format(args.num_epochs)), "w") as file:
            file.write(f"Accuracy is {acc}" + "\n")

def main():
    train()
    eval()

main()