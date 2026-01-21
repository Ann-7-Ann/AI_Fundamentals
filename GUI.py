from PyQt5.QtWidgets import (
    QWidget, QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import plotly.graph_objects as go
import numpy as np

class GaussianMixtureGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generate samples")
        self.resize(1000, 700)
        self.init_ui()
        self.generate_and_plot()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # ---------------- Controls ----------------
        control_box = QGroupBox("Parameters")
        control_layout = QVBoxLayout()

        # Modes & Samples
        self.modes_spin = QSpinBox()
        self.modes_spin.setRange(1, 10)
        self.modes_spin.setValue(2)

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(10, 5000)
        self.samples_spin.setValue(200)

        # Class A params
        self.mean_min_A = QDoubleSpinBox(); self.mean_min_A.setRange(-10, 10); self.mean_min_A.setValue(1.0)
        self.mean_max_A = QDoubleSpinBox(); self.mean_max_A.setRange(-10, 10); self.mean_max_A.setValue(10.0)
        self.var_min_A = QDoubleSpinBox(); self.var_min_A.setRange(0.001, 5.0); self.var_min_A.setValue(0.05)
        self.var_max_A = QDoubleSpinBox(); self.var_max_A.setRange(0.001, 5.0); self.var_max_A.setValue(0.2)

        # Class B params
        self.mean_min_B = QDoubleSpinBox(); self.mean_min_B.setRange(-10, 10); self.mean_min_B.setValue(-1.0)
        self.mean_max_B = QDoubleSpinBox(); self.mean_max_B.setRange(-10, 10); self.mean_max_B.setValue(-5.0)
        self.var_min_B = QDoubleSpinBox(); self.var_min_B.setRange(0.001, 5.0); self.var_min_B.setValue(0.05)
        self.var_max_B = QDoubleSpinBox(); self.var_max_B.setRange(0.001, 5.0); self.var_max_B.setValue(0.2)

        # Buttons
        self.generate_btn = QPushButton("Generate Samples")
        self.generate_btn.clicked.connect(self.generate_and_plot)
        self.train_btn = QPushButton("Train")

        # Add widgets to layout
        control_layout.addWidget(QLabel("Modes per class"))
        control_layout.addWidget(self.modes_spin)

        control_layout.addWidget(QLabel("Samples per mode"))
        control_layout.addWidget(self.samples_spin)

        control_layout.addWidget(QLabel("<b>Class A Parameters</b>"))
        control_layout.addWidget(QLabel("Mean range min")); control_layout.addWidget(self.mean_min_A)
        control_layout.addWidget(QLabel("Mean range max")); control_layout.addWidget(self.mean_max_A)
        control_layout.addWidget(QLabel("Variance range min")); control_layout.addWidget(self.var_min_A)
        control_layout.addWidget(QLabel("Variance range max")); control_layout.addWidget(self.var_max_A)

        control_layout.addWidget(QLabel("<b>Class B Parameters</b>"))
        control_layout.addWidget(QLabel("Mean range min")); control_layout.addWidget(self.mean_min_B)
        control_layout.addWidget(QLabel("Mean range max")); control_layout.addWidget(self.mean_max_B)
        control_layout.addWidget(QLabel("Variance range min")); control_layout.addWidget(self.var_min_B)
        control_layout.addWidget(QLabel("Variance range max")); control_layout.addWidget(self.var_max_B)

        control_layout.addWidget(self.generate_btn)
        
        control_layout.addWidget(self.train_btn)

        control_box.setLayout(control_layout)

        # ---------------- Plot Area ----------------
        self.browser = QWebEngineView()

        # Add to main layout
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

    def create_grid(self):
        # Create a grid over input space
        xx, yy = np.meshgrid(
            np.linspace(self.X[:,0].min()-1, self.X[:,0].max()+1, 200),
            np.linspace(self.X[:,1].min()-1, self.X[:,1].max()+1, 200)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        return grid, xx, yy
    
    def plot_boundary(self, pred, xx, yy):

        zz = pred.reshape(xx.shape)

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