import sys
import numpy as np
import struct
from array import array
import random
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from GUI import GaussianMixtureGUI

# -------------------------
# MNIST Data Loader
# -------------------------
class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        with open(labels_filepath, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f'Labels magic mismatch: {magic}')
            labels = array("B", f.read())

        with open(images_filepath, 'rb') as f:
            magic, size_img, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f'Images magic mismatch: {magic}')
            image_data = array("B", f.read())

        images = [np.array(image_data[i*rows*cols:(i+1)*rows*cols]).reshape(rows, cols) 
                  for i in range(size_img)]
        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

# -------------------------
# Helper: show images
# -------------------------
def show_images(images, titles):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(15, 10))
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, idx+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------
# Neural Network Classes
# -------------------------
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.w = np.random.randn(n_neurons, n_inputs) * np.sqrt(1 / n_inputs)
        self.b = np.zeros((1, n_neurons))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        self.X = X
        self.s = X @ self.w.T + self.b
        self.y_hat = self.sigmoid(self.s)
        return self.y_hat

    def backward(self, delta_next=None, w_next=None, y=None):
        if y is not None:
            error = y - self.y_hat
            self.delta = error * self.sigmoid_derivative(self.s)
        else:
            self.delta = (delta_next @ w_next) * self.sigmoid_derivative(self.s)
        return self.delta

    def update(self, lr):
        self.w += lr * self.delta.T @ self.X
        self.b += lr * np.sum(self.delta, axis=0, keepdims=True)

class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.1):
        self.lr = lr
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y):
        delta = self.layers[-1].backward(y=y)
        for i in reversed(range(len(self.layers)-1)):
            delta = self.layers[i].backward(delta_next=delta, w_next=self.layers[i+1].w)
        for layer in self.layers:
            layer.update(self.lr)

    def train(self, X, y, epochs=50, batch_size=32, verbose=True):
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_shuff, y_shuff = X[idx], y[idx]
            for i in range(0, n, batch_size):
                xb = X_shuff[i:i+batch_size]
                yb = y_shuff[i:i+batch_size]
                self.forward(xb)
                self.backward(yb)
            if verbose and epoch % 10 == 0:
                pred = self.forward(X)
                loss = np.mean((y - pred)**2)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def predict(self, X):
        out = self.forward(X)
        return np.argmax(self.softmax(out), axis=1)

# -------------------------
# Main
# -------------------------
def main():
    # Choose dataset
    dataset_choice = input("Train on (1) MNIST or (2) Generated classes? Enter 1 or 2: ")

    if dataset_choice == '1':
        # ---------------- MNIST ----------------
        print("Loading MNIST dataset...")
        training_images_filepath = 'MINST_dataset/train-images.idx3-ubyte'
        training_labels_filepath = 'MINST_dataset/train-labels.idx1-ubyte'
        test_images_filepath = 'MINST_dataset/t10k-images.idx3-ubyte'
        test_labels_filepath = 'MINST_dataset/t10k-labels.idx1-ubyte'

        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                           test_images_filepath, test_labels_filepath)
        (X_train, y_train), (X_test, y_test) = mnist_dataloader.load_data()

        # # Show some random training and test images # 
        images_2_show = [] 
        titles_2_show = [] 
        for i in range(0, 10): 
            r = random.randint(1, 60000) 
            images_2_show.append(X_train[r]) 
            titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r])) 
        for i in range(0, 5): 
            r = random.randint(1, 10000) 
            images_2_show.append(X_test[r]) 
            titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r])) 
        show_images(images_2_show, titles_2_show)

        # Flatten and normalize
        X_train = np.array(X_train, dtype=np.float32).reshape(len(X_train), -1)/255.0
        X_test = np.array(X_test, dtype=np.float32).reshape(len(X_test), -1)/255.0

        y_train = np.array(y_train, dtype=int)
        y_test = np.array(y_test, dtype=int)
        num_classes = 10
        y_train_one_hot = np.eye(num_classes)[y_train]
        y_test_one_hot = np.eye(num_classes)[y_test]

        # Create NN
        net = NeuralNetwork([784, 64, 32, num_classes], lr=0.05)
        net.train(X_train, y_train_one_hot, epochs=20, batch_size=64)

        # Evaluate
        probabailities = net.forward(X_test)
        y_pred = net.predict(X_test) 
        accuracy = np.mean(y_pred == y_test.astype(int)) 
        print(f"Test accuracy: {accuracy*100:.2f}%") 
        images_to_show = X_test.reshape(-1,28,28) # reshape if flattened 
        sample_idx = random.sample(range(len(X_test)), 10) 
        images_sample = images_to_show[sample_idx] 
        print(f"Probabilities: {probabailities[sample_idx]}") 
        titles_sample = [f"Pred: {y_pred[i]}, True: {y_test[i]}" for i in sample_idx]
        show_images(images_sample, titles_sample)

    elif dataset_choice == '2':
        # ---------------- Generated classes ----------------
        app = QApplication(sys.argv)
        gui = GaussianMixtureGUI()
        gui.show()

        def train_generated_data():
            if not hasattr(gui, "X") or not hasattr(gui, "y"):
                print("Generate classes first in GUI!")
                return

            X = gui.X.astype(np.float32)
            y = gui.y.astype(int)
            num_classes = len(np.unique(y))

            # ------------------------
            # Manual train/test split
            # ------------------------
            n_samples = X.shape[0]
            indices = np.arange(n_samples)
            np.random.seed(42)
            np.random.shuffle(indices)

            split = int(0.8 * n_samples)  # 80% train, 20% test
            train_idx, test_idx = indices[:split], indices[split:]

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # One-hot encode training labels
            y_train_one_hot = np.eye(num_classes)[y_train]

            # ------------------------
            # Normalize features
            # ------------------------
            X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
            X_train_norm = (X_train - X_mean) / (X_std + 1e-8)
            X_test_norm = (X_test - X_mean) / (X_std + 1e-8)  # normalize using train stats

            # ------------------------
            # Neural network: 2 -> 16 -> 8 -> num_classes
            # ------------------------
            net = NeuralNetwork([2, 16, 8, num_classes], lr=0.05)
            net.train(X_train_norm, y_train_one_hot, epochs=100, batch_size=32)

            # ------------------------
            # Predictions and accuracy
            # ------------------------
            train_prob = net.forward(X_train_norm)
            train_pred = np.argmax(train_prob, axis=1)
            train_acc = np.mean(train_pred == y_train)
            print(f"Training accuracy: {train_acc*100:.2f}%")

            test_prob = net.forward(X_test_norm)
            test_pred = np.argmax(test_prob, axis=1)
            test_acc = np.mean(test_pred == y_test)
            print(f"Test accuracy: {test_acc*100:.2f}%")

            print("Probabilities (first 10 test samples):\n", test_prob[:10])

            # ------------------------
            # Plot decision boundary on grid
            # ------------------------
            grid, xx, yy = gui.create_grid()  # grid = (num_grid_points, 2)
            grid_norm = (grid - X_mean) / (X_std + 1e-8)
            grid_prob = net.forward(grid_norm)
            grid_class = np.argmax(grid_prob, axis=1)

            gui.plot_boundary(grid_class, xx, yy)


        gui.train_btn.clicked.connect(train_generated_data)
        sys.exit(app.exec_())


    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
