

import dapnn.models
import mdm_models

import dapnn.data as data
import dapnn.experiment as exp
import dapnn.model_helpers as mh
import dapnn.nn

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time

max_num_characters = 6

def data_aug(item):
    t_labels = item[1]
    pad_size = (max_num_characters - t_labels.shape[0])
    t_labels = np.pad(t_labels, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)

    label_mask = np.zeros(max_num_characters)
    label_mask[np.arange(0, item[2])] = 1

    num_char_class_vector = np.zeros(max_num_characters, dtype=np.float32)
    num_char_class_vector[item[2] - 1] = 1

    return item[0], t_labels, label_mask, num_char_class_vector

digit_counter_modules = [
    mh.reshape(size=(-1, 1, 28, 28 * max_num_characters)),
    mh.pad2d(kernel_size=5, dilation=1),
    nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2),
    nn.ReLU(),
    mh.pad2d(kernel_size=5, dilation=1),
    nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, stride=2),
    nn.ReLU(),
    mh.reshape(size=(-1, 8 * 7 * 7 * max_num_characters)),
    nn.Linear(in_features=8 * 7 * 7 * max_num_characters, out_features=6),
    nn.LogSoftmax(1)
]

baseline_feature_extraction = [
    mh.reshape(size=(-1, 1, 28, 28 * max_num_characters)),
    mh.pad2d(kernel_size=5, dilation=1),
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
    nn.ReLU(),
    mh.pad2d(kernel_size=5, dilation=1),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),
    nn.ReLU(),
]

baseline_clas_layers = [nn.Linear(in_features=32 * 7 * 7 * max_num_characters, out_features=10 * i) for i in range(1, max_num_characters+1)]

attn_fe_modules = [
    mh.reshape(size=(-1, 1, 28, 28 * max_num_characters)),
    mh.pad2d(kernel_size=5, dilation=1),
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
    nn.ReLU(),
    mh.pad2d(kernel_size=5, dilation=1),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(7, 3), stride=(7, 1)),
    nn.ReLU(),
]

def train_digit_counter(train_dl, val_dl, test_dl, plot=False):
    digit_counter = dapnn.models.sequential(digit_counter_modules, criterion=nn.KLDivLoss(), out_batch_idx=3, name='digit_counter', weight_decay=0)

    digit_counter.load_state_dict(torch.load('./models/best_' + digit_counter.name + '.ckpt'))
    #dapnn.nn.train_model(digit_counter, train_dl, val_dl, train_sec=60)

    #print('Digit Counter Train Acc: {}'.format(dapnn.nn.classifier_accuracy(digit_counter, train_dl)))
    #print('Digit Counter Val Acc: {}'.format(dapnn.nn.classifier_accuracy(digit_counter, val_dl)))
    #print('Digit Counter Test Acc: {}'.format(dapnn.nn.classifier_accuracy(digit_counter, test_dl)))

    if plot:
        plt.plot(digit_counter.train_batch_loss_x.cpu().numpy(), digit_counter.train_batch_loss_y.cpu().numpy(), label='train_loss')
        plt.plot(digit_counter.checkpoint_x.cpu().numpy(), digit_counter.val_ckpt_loss_y.cpu().numpy(), label='val_loss')
        plt.legend()

        plt.show()

        digit_counter.cpu()

        for i in range(50, 60):
            print('Output: {}'.format(np.argmax(digit_counter(torch.Tensor(test_dl.dataset[i][0].reshape((1, 28, 28 * max_num_characters)))).detach().numpy()[0], axis=-1) + 1))
            print('Label: {}'.format(np.argmax(test_dl.dataset[i][3], axis=-1) + 1))
            plt.imshow(test_dl.dataset[i][0])
            plt.show()

    return digit_counter

def main():
    train_dataloader = data.build_DataLoader(data.load_dataset('./datasets/mdm_train.data', data_aug=data_aug), batch_size=100, shuffle=True)
    val_dataloader = data.build_DataLoader(data.load_dataset('./datasets/mdm_val.data', data_aug=data_aug), batch_size=100, shuffle=False)
    test_dataloader = data.build_DataLoader(data.load_dataset('./datasets/mdm_test.data', data_aug=data_aug), batch_size=100, shuffle=False)

    digit_counter = train_digit_counter(train_dataloader, val_dataloader, test_dataloader)

    baseline = mdm_models.baseline(baseline_feature_extraction, baseline_clas_layers, criterion=nn.KLDivLoss(reduction='sum'), weight_decay=0.00001, name='mdm_baseline')
    attn_model = mdm_models.attn_test(attn_fe_modules, criterion=nn.KLDivLoss(reduction='sum'), weight_decay=0.00001, name='attn_test')

    model = attn_model

    model.load_state_dict(torch.load('./models/best_' + model.name + '.ckpt'))
    #dapnn.nn.train_model(model, train_dataloader, val_dataloader, train_sec=180)

    #print('Train Acc: {}'.format(dapnn.nn.classifier_accuracy(model, train_dataloader)))
    #print('Val Acc: {}'.format(dapnn.nn.classifier_accuracy(model, val_dataloader)))
    #print('Test Acc: {}'.format(dapnn.nn.classifier_accuracy(model, test_dataloader)))

    plt.plot(model.train_batch_loss_x.cpu().numpy(), model.train_batch_loss_y.cpu().numpy(), label='train_loss')
    plt.plot(model.checkpoint_x.cpu().numpy(), model.val_ckpt_loss_y.cpu().numpy(), label='val_loss')
    plt.legend()

    plt.show()

    model.cpu()

    for i in range(10, 20):
        input_image = torch.Tensor(test_dataloader.dataset[i][0].reshape((1, 1, 28, 28 * max_num_characters)))
        num_chars = digit_counter(input_image).argmax(dim=-1) + 1

        index = 0 if model.name == 'attn_test' else num_chars-1

        pred = model(input_image, num_chars)[index].argmax(dim=-1)[0].detach().numpy()

        print('Output: {}'.format(pred))
        print('Label: {}'.format(np.argmax(test_dataloader.dataset[i][1], axis=-1)))
        plt.imshow(test_dataloader.dataset[i][0])
        plt.show()


if __name__ == '__main__':
    main()
