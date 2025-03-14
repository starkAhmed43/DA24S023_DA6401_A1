from keras.datasets import fashion_mnist, mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_data(dataset='fashion_mnist'):
    if dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        raise ValueError(f"Dataset {dataset} not supported. Please choose from 'fashion_mnist' or 'mnist'.")

    # Normalize images
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)  # (60000, 784)
    X_test = X_test.reshape(X_test.shape[0], -1)     # (10000, 784)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test