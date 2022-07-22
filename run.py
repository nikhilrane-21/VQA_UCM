from dataloader import *

ucm_vqa_train_dataloader, _, _ = construct_data_loader(batch_size=16)  # return the dataloader of {train, eval, test}

for i_batch, sample_batched in enumerate(ucm_vqa_train_dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['question'], sample_batched['answer'])
    if i_batch == 3:
        break

