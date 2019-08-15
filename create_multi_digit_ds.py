
import dapnn.data as data
import numpy as np
import matplotlib.pyplot as plt

max_num_characters = 6


def load_datasets():
    train_dataset = data.load_dataset('./datasets/mnist_train.data')
    val_dataset = data.load_dataset('./datasets/mnist_val.data')
    test_dataset = data.load_dataset('./datasets/mnist_test.data')

    return train_dataset, val_dataset, test_dataset

def create_multi_digit_ds(dataset, num_samples):
    images = []
    labels = []
    num_chars = []

    for _ in range(num_samples):
        num_characters = np.random.randint(1, max_num_characters+1)
        choices = np.random.choice(np.arange(0, len(dataset)), num_characters)
        t_images = dataset.data[0][choices].reshape(num_characters, 28, 28)

        string_image = np.concatenate(list(t_images), axis=1)
        pad_size = (max_num_characters - num_characters) * 14
        string_image = np.pad(string_image, ((0, 0), (pad_size, pad_size)), mode='constant')

        images.append(string_image)
        labels.append(dataset.data[1][choices])
        num_chars.append(num_characters)

    return data.dataset([np.array(images), np.array(labels), np.array(num_chars).astype(np.longlong)])

if __name__ == '__main__':
    mnist_train, mnist_val, mnist_test = load_datasets()

    mdm_train = create_multi_digit_ds(mnist_train, 100000)
    mdm_val = create_multi_digit_ds(mnist_val, 10000)
    mdm_test = create_multi_digit_ds(mnist_test, 20000)

    mdm_train.save('./datasets/mdm_train.data')
    mdm_val.save('./datasets/mdm_val.data')
    mdm_test.save('./datasets/mdm_test.data')


