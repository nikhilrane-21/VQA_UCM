import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG19_Weights
from transformers import AutoModel, AutoTokenizer

from transformers import logging
logging.set_verbosity_warning()

class ImgEncoder(nn.Module):
    def __init__(self, embed_size):
        """
        Image Encoder for VQA
        :param embed_size:
        """
        super(ImgEncoder, self).__init__()
        ptm = torchvision.models.vgg19(VGG19_Weights.IMAGENET1K_V1)  # load the pretrained model
        in_features = ptm.classifier[-1].in_features  # input size of the feature vector
        ptm.classifier = nn.Sequential(
            *list(ptm.classifier.children())[:-1]
            # remove the last fc layer of the ptm (score values from the ImageNet)
        )
        self.model = ptm
        self.fc = nn.Linear(in_features, embed_size)  # feature vector of image

    def forward(self, img):
        """
        Extract feature vector from image vector
        :param image:
        :return: img_feature
        """
        with torch.no_grad():
            img = img.float()
            img_feature = self.model(img) # load the ptm model
        img_feature = self.fc(img_feature)  # [batch_size, embed_size]
        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)  # l2-normalized feature vector

        return img_feature


class QstEncoder(nn.Module):
    def __init__(self, qst_vocab_size=15, word_embed_size=10, embed_size=10, num_layers=3, hidden_size=5,
                 using_transformers=False):
        """
        Question Encoder for VQA
        :param qst_vocab_size:
        :param word_embed_size:
        :param embed_size:
        :param num_layers:
        :param hidden_size:
        :param using_transformers: if using transformers, will load the huggingface ptms for the pst encoding.
        """
        super(QstEncoder, self).__init__()
        self.using_transformers = using_transformers
        if self.using_transformers:
            ptms = "bert-base-uncased"  # using transformers ptms
            self.qst_tokenizer = AutoTokenizer.from_pretrained(ptms)
            self.qst_encoder = AutoModel.from_pretrained(ptms)
        else:
            self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
            self.tanh = nn.Tanh()
            self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
            self.fc = nn.Linear(2 * num_layers * hidden_size, embed_size)  # 2 for hidden and cell states

    def forward(self, qst):
        if self.using_transformers:
            qst_vec = self.qst_tokenizer(qst, return_tensors='pt')
            qst_feature = self.qst_encoder(**qst_vec)[-1]  # [batch_size, embed_size]
            return qst_feature

        qst_vec = self.word2vec(qst)  # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)  # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)  # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)  # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)  # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)  # [batch_size, embed_size]

        return qst_feature


class QstEncoder_ptm(nn.Module):
    """
    ptm model for question encoder
    """
    def __init__(self, embed_size, ptm="bert-base-uncased"):
        super(QstEncoder_ptm, self).__init__()
        self.qst_encoder = AutoModel.from_pretrained(ptm)
        self.fc = nn.Linear(768, embed_size)  # feature vector of qst

    def forward(self, qst):
        qst_feature = self.qst_encoder(input_ids=qst)[-1]  # [batch_size, embed_size]
        qst_feature = self.fc(qst_feature)

        return qst_feature


class VqaModel(nn.Module):
    def __init__(self, embed_size, num_labels):
        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size=embed_size)
        self.qst_encoder = QstEncoder_ptm(embed_size=embed_size, ptm="bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Tanh(),
            nn.Linear(embed_size, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.Linear(64, num_labels)
        )

    def forward(self, img, qst):
        img_feature = self.img_encoder(img)  # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)  # [batch_size, embed_size]
        combined_feature = self._fusion_element_wise(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.classifier(combined_feature)

        return combined_feature

    def _fusion_element_wise(self, img_feature, qst_feature):
        """
        Fusion strategy - 1: Element-wise
        :param img_feature:
        :param qst_feature:
        :return:
        """
        return torch.mul(img_feature, qst_feature)

    def _fusion_concatenate(self, img_feature, qst_feature):
        """
        Fusion strategy - 2 : Concatenation
        :param img_feature:
        :param qst_feature:
        :return:
        """
        return torch.cat((img_feature, qst_feature),1)

    def _fusion_MCB(self, img_feature, qst_feature):
        """
        Fusion strategy - 3: MCB
        :param img_feature:
        :param qst_feature:
        :return:
        """
        # TODO: add the fusion_MCB function
        pass

    def _fusion_MUTAN(self, img_feature, qst_feature):
        """
        Fusion strategy - 4: MUTAN
        :param img_feature:
        :param qst_feature:
        :return:
        """
        # TODO: add the fusion_MUTAN function, http://dx.doi.org/10.1109/ICCV.2017.285
        pass

    def _fusion_Block(self, img_feature, qst_feature):
        """

        :param img_feature:
        :param qst_feature:
        :return:
        """
        # TODO: https://arxiv.org/pdf/1902.00038


class VqaModel_test_Qst(VqaModel):
    """
    Remove the img_encoder, only test the question embedding performance
    """
    def forward(self, img, qst):
        qst_feature = self.qst_encoder(qst)
        combined_feature = self.tanh(qst_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc(combined_feature)

        return combined_feature

class VqaModel_test_Img(nn.Module):
    """
    Remove the qst_encoder, only test the image embedding performance
    """
    def __init__(self, embed_size, num_labels):
        super(VqaModel_test_Img, self).__init__()
        self.img_encoder = ImgEncoder(embed_size=embed_size)
        self.classifier = nn.Sequential(
            nn.Tanh(),
            nn.Linear(embed_size, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.Linear(64, num_labels)
        )

    def forward(self, img):
        img_feature = self.img_encoder(img)
        img_feature = self.classifier(img_feature)

        return img_feature
