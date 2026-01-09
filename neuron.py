import sys
import numpy as np
import plotly.graph_objects as go

# ---------------- Activation functions ----------------

def heaviside_step_func(s):
    return np.where(s >= 0, 1, 0) #(s >= 0).astype(int)

def df_heaviside_step_func(s):
    values = np.ones_like(s, dtype=float)    #assume it is one
    return values

"big weights crash"
def sigmoid_func(s,beta=1):
    return 1/(1+np.exp(-beta*s))

def df_sigmoid_func(s,beta=1):
    return beta*sigmoid_func(s,beta) * (1-sigmoid_func(s,beta))


"big values - several areas"
def sin_func(s):
    return np.sin(s)

def df_sin_func(s):
    return np.cos(s)

"if big weights or x then it crashes"
"mapping"
def tanh_func(s):
    return np.tanh(s)

def df_tanh_func(s):
    return 1/(np.cosh(s)**2)

"treats classes unequally"
"mapping"
def sign_func(s):
    return np.where(s < 0, -1, np.where(s == 0, 0, 1))
    
def df_sign_func(s):
    return np.ones_like(s, dtype=float)

"negative pre-activation values do not update the weights"
def ReLU_func(s):
    return np.maximum(0, s) 

def df_ReLU_func(s):
    return np.where(s >= 0, 1, 0)

def leaky_ReLU_func(s):
    return np.where(s > 0, s, 0.01*s)  # 0.01*s for s <= 0

def df_leaky_ReLU_func(s):
    return np.where(s > 0, 1.0, 0.01)

ACTIVATIONS = {
    "heaviside": (heaviside_step_func, df_heaviside_step_func),
    "sigmoid": (sigmoid_func, df_sigmoid_func),
    "sin": (sin_func, df_sin_func),
    "tanh": (tanh_func, df_tanh_func),
}

EVAL_ACTIVATIONS = {
    "heaviside": (heaviside_step_func, 0.5),
    "sigmoid": (sigmoid_func, 0.5),
    "sin": (sin_func, 0.0),
    "tanh": (tanh_func, 0.0),
    "sign": (sign_func,  0.0),
    "relu": (ReLU_func, 0.0),
    "leaky_relu": (leaky_ReLU_func, 0.0),
}
# ---------------- Neuron implementation ----------------

class Neuron:

    '''
    y is the expected (true) class label
    y_hat is a class label predicted by the neuron
    '''

    def __init__(self, activation="sigmoid", eval_activation = "sign",lr=0.1):
        self.w = np.random.randn(2)
        self.b = 0.01   
        self.lr = lr
        self.f, self.df = ACTIVATIONS[activation]
        self.eval_activation, self.threshold = EVAL_ACTIVATIONS[eval_activation]

    def calculate_lr(self,epoch, max_epochs, lr_min, lr_max):
        return lr_min +  (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / max_epochs))

    def forward(self,X):
        self.X = X
        self.s = X @ self.w + self.b 
        self.y_hat = self.f(self.s)
        return self.y_hat
    
    def backward(self, y):
        # dL/dy_hat (MSE loss)
        dL_dy = y - self.y_hat
        # dy_hat/ds
        dy_ds = self.df(self.s)
        # dL/ds
        self.delta = - (dL_dy * dy_ds)
        # gradients
        self.grad_w = self.X.T @ self.delta / len(y)   # sum over all samples/ n
        self.grad_b = np.mean(self.delta)

    def update(self,epoch, epochs):
        lr = self.calculate_lr(epoch,max_epochs = epochs,lr_min=0.001,lr_max=self.lr)
        print(self.grad_w)
        self.w -= lr * self.grad_w
        self.b -= lr * self.grad_b

    def train(self, X, y, epochs=100, print_every=10):
        
        if self.f == tanh_func or self.f == sign_func or self.f == sin_func:
            y[y==0] = -1
        if self.f == sin_func:
            X = self.normalize(X)
        for epoch in range(epochs):
            self.forward(X)
            self.backward(y)
            self.update(epoch, epochs)

            if epoch % print_every == 0:
                loss = self.compute_loss(y, self.y_hat)
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
        
        if self.f == tanh_func or self.f == sign_func or self.f == sin_func:
            y[y==-1] = 0

    def predict_value(self, X):
        if self.eval_activation == sin_func:
            X = self.normalize(X)
        return self.eval_activation(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_value(X) > self.threshold).astype(int)
    
    def compute_loss(self, y, y_hat):
        loss = 0.5 * np.mean((y - y_hat) ** 2)
        return loss
    
    def normalize(self,X):
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        return (X - mean) / std

        

# ---------------- App ----------------

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox

)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl


class GaussianMixtureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generate samples and train neuron")
        self.resize(1000, 700)

        self.init_ui()
        self.generate_and_plot()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # ---------------- Controls ----------------
        control_box = QGroupBox("Parameters")
        control_layout = QVBoxLayout()

        self.modes_spin = QSpinBox()
        self.modes_spin.setRange(1, 10)
        self.modes_spin.setValue(2)

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(10, 5000)
        self.samples_spin.setValue(200)

        self.mean_min_A = QDoubleSpinBox()
        self.mean_min_A.setRange(-10, 10)
        self.mean_min_A.setValue(1.0)

        self.mean_max_A = QDoubleSpinBox()
        self.mean_max_A.setRange(-10, 10)
        self.mean_max_A.setValue(10.0)

        self.var_min_A = QDoubleSpinBox()
        self.var_min_A.setRange(0.001, 5.0)
        self.var_min_A.setValue(0.05)

        self.var_max_A = QDoubleSpinBox()
        self.var_max_A.setRange(0.001, 5.0)
        self.var_max_A.setValue(0.2)

        self.mean_min_B = QDoubleSpinBox()
        self.mean_min_B.setRange(-10, 10)
        self.mean_min_B.setValue(-1.0)

        self.mean_max_B = QDoubleSpinBox()
        self.mean_max_B.setRange(-10, 10)
        self.mean_max_B.setValue(-5.0)

        self.var_min_B = QDoubleSpinBox()
        self.var_min_B.setRange(0.001, 5.0)
        self.var_min_B.setValue(0.05)

        self.var_max_B = QDoubleSpinBox()
        self.var_max_B.setRange(0.001, 5.0)
        self.var_max_B.setValue(0.2)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(50)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1)
        self.lr_spin.setValue(0.05)

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(list(ACTIVATIONS.keys()))

                
        self.eval_activation_combo = QComboBox()
        self.eval_activation_combo.addItems(EVAL_ACTIVATIONS.keys())
        
        generate_btn = QPushButton("Generate Samples")
        generate_btn.clicked.connect(self.generate_and_plot)
        train_btn = QPushButton("Train Neuron")
        train_btn.clicked.connect(self.train_neuron) 

        control_layout.addWidget(QLabel("Modes per class"))
        control_layout.addWidget(self.modes_spin)

        control_layout.addWidget(QLabel("Samples per mode"))
        control_layout.addWidget(self.samples_spin)

        control_layout.addWidget(QLabel("<b>Class A Parameters</b>"))

        control_layout.addWidget(QLabel("Mean range min"))
        control_layout.addWidget(self.mean_min_A)

        control_layout.addWidget(QLabel("Mean range max"))
        control_layout.addWidget(self.mean_max_A)

        control_layout.addWidget(QLabel("Variance range min"))
        control_layout.addWidget(self.var_min_A)

        control_layout.addWidget(QLabel("Variance range max"))
        control_layout.addWidget(self.var_max_A)

        control_layout.addWidget(QLabel("<b>Class B Parameters</b>"))

        control_layout.addWidget(QLabel("Mean range min"))
        control_layout.addWidget(self.mean_min_B)

        control_layout.addWidget(QLabel("Mean range max"))
        control_layout.addWidget(self.mean_max_B)

        control_layout.addWidget(QLabel("Variance range min"))
        control_layout.addWidget(self.var_min_B)

        control_layout.addWidget(QLabel("Variance range max"))
        control_layout.addWidget(self.var_max_B)

        control_layout.addWidget(generate_btn)

        control_layout.addWidget(QLabel("Activation function"))
        control_layout.addWidget(self.activation_combo)
        
        control_layout.addWidget(QLabel("Evaluation activation"))
        control_layout.addWidget(self.eval_activation_combo)

        control_layout.addWidget(QLabel("Training epochs"))
        control_layout.addWidget(self.epochs_spin)

        control_layout.addWidget(QLabel("Starting learning rate"))
        control_layout.addWidget(self.lr_spin)

        control_layout.addWidget(train_btn) 
        control_layout.addStretch()

        control_box.setLayout(control_layout)

        # ---------------- Plot Area ----------------
        self.browser = QWebEngineView()

        main_layout.addWidget(control_box, 1)
        main_layout.addWidget(self.browser, 3)

    def sample_class(self, n_modes, n_samples, mean_min, mean_max, var_min, var_max):
        data = []
        for _ in range(n_modes):
            mean = np.random.uniform(mean_min, mean_max, size=2)
            var = np.random.uniform(var_min, var_max)
            cov = var * np.eye(2)
            samples = np.random.multivariate_normal(mean, cov, n_samples)
            data.append(samples)
        return np.vstack(data)

    def generate_and_plot(self):
        modes = self.modes_spin.value()
        samples = self.samples_spin.value()

        class_a = self.sample_class(
                modes, samples,
                self.mean_min_A.value(),
                self.mean_max_A.value(),
                self.var_min_A.value(),
                self.var_max_A.value()
            )

        class_b = self.sample_class(
                modes, samples,
                self.mean_min_B.value(),
                self.mean_max_B.value(),
                self.var_min_B.value(),
                self.var_max_B.value()
            )

        self.X = np.vstack([class_a, class_b])
        self.y = np.hstack([
            np.zeros(len(class_a)),
            np.ones(len(class_b))
        ])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=class_a[:, 0],
            y=class_a[:, 1],
            mode="markers",
            name="Class A",
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=class_b[:, 0],
            y=class_b[:, 1],
            mode="markers",
            name="Class B",
            marker=dict(size=6)
        ))

        fig.update_layout(
            title="2D Gaussian Mixture Samples",
            xaxis_title="x",
            yaxis_title="y",
            template="plotly_white",
            legend=dict(x=0.01, y=0.99)
        )

        html = fig.to_html(include_plotlyjs="cdn")
        self.browser.setHtml(html, QUrl(""))


    def train_neuron(self):
        if not hasattr(self, "X") or not hasattr(self, "y"):
            print("Generate samples first!")
            return

        lr = self.lr_spin.value()
        activation = self.activation_combo.currentText()
        eval_act = self.eval_activation_combo.currentText()

        self.neuron = Neuron(activation=activation, eval_activation=eval_act, lr=lr)
        print("Initial weights and bias: ", self.neuron.w,'x + ', self.neuron.b)
        epochs = self.epochs_spin.value()
        self.neuron.train(self.X, self.y, epochs=epochs)

        preds = self.neuron.predict(self.X)
        accuracy = np.mean(preds == self.y)
        print("Training accuracy:", accuracy)
        print("Resulting linear function: ", self.neuron.w,'x + ', self.neuron.b)

    # Create a grid over input space
        xx, yy = np.meshgrid(
            np.linspace(self.X[:,0].min()-1, self.X[:,0].max()+1, 200),
            np.linspace(self.X[:,1].min()-1, self.X[:,1].max()+1, 200)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]

        zz = self.neuron.predict(grid).reshape(xx.shape)

        fig = go.Figure()

        # Scatter points
        fig.add_trace(go.Scatter(
            x=self.X[self.y==0, 0],
            y=self.X[self.y==0, 1],
            mode="markers",
            name="Class A (0)",
            marker=dict(size=6, color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=self.X[self.y==1, 0],
            y=self.X[self.y==1, 1],
            mode="markers",
            name="Class B (1)",
            marker=dict(size=6, color='red')
        ))

        
        # Contour for coloring regions
        fig.add_trace(go.Contour(
            x=xx[0],
            y=yy[:,0],
            z=zz,
            showscale=False,
            colorscale="RdBu_r",
            opacity=0.3
        ))
        """
        x1_line = np.linspace(self.X[:,0].min()-1, self.X[:,0].max()+1, 200)
        x2_line = -(self.neuron.w[0]/self.neuron.w[1])*x1_line - self.neuron.b/self.neuron.w[1]  # linear boundary
        fig.add_trace(go.Scatter(
            x=x1_line,
            y=x2_line,
            mode="lines",
            name="Decision boundary",
            line=dict(color="blue", dash="dash", width=2)
        ))
        """

        fig.update_layout(
            title="Samples with Decision Boundary",
            xaxis_title="x",
            yaxis_title="y",
            template="plotly_white"
        )

        html = fig.to_html(include_plotlyjs="cdn")
        self.browser.setHtml(html, QUrl(""))



# ---------------- Run App ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GaussianMixtureApp()
    window.show()
    sys.exit(app.exec_())
