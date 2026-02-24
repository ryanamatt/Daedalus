# **Daedalus 🕊️**

A C++ powered Machine Learning library with Python bindings, built for the sake of learning how the math actually works under the hood.

**Wait, why should I use this?**

Honestly? You probably shouldn't. If you want speed, reliability, or a job in industry, go use scikit-learn or PyTorch. This is a sandbox for me to crash C++ into Python while figuring out what Machine Learning from scratch. It might leak memory, it might segfault, and it definitely won't beat XGBoost. Use it at your own peril (and for the vibes).

## **🚀 Features (The "It Actually Works" List)**

Despite the disclaimer, Daedalus implements several core ML components in C++17, exposed via pybind11:

* **Linear Regression:** Classic Ordinary Least Squares/Gradient Descent.  
* **Logistic Regression:** For your classification needs.  
* **K-Nearest Neighbors (KNN):** Simple, effective, and written in C++.  
* **Neural Networks:**   
  * DenseLayer implementations.  
  * Flexible NeuralNetwork assembly.  
* **Utilities:**  
  * StandardScaler for preprocessing.  
  * train\_test\_split for model selection.  
  * CSV parsing and Matrix conversions built into the core.

## **🧠 The Learning Journey (C++ Notes)**

I am relatively new to C++ and am using this project to transition from high-level Python logic to lower-level systems programming. This library is as much about learning idiomatic C++17 as it is about Machine Learning.

Inside the source, you'll find experiments with:

* **Memory Management:** Learning how to handle row-major data storage in std::vector.  
* **Optimization:** The Matrix class includes a **blocked (tiled) transpose algorithm** to improve L1/L2 cache hits—my first foray into cache-aware programming.  
* **Bindings:** Bridging the gap between C++ performance and Python ease-of-use via pybind11.

## **📁 Project Structure**


├── daedalus/           \# Python package (includes \_core bindings & stubs)  
├── include/daedalus/   \# C++ Header files  
│   ├── core/           \# Matrix, DataFrame, and Math logic  
│   └── models/         \# Model definitions (LinearRegression, etc.)  
├── src/                \# C++ Implementation files  
│   ├── bindings/       \# pybind11 glue code  
│   └── models/         \# Model logic implementation  
├── tests/              \# Python test suite & datasets  
├── examples/           \# Usage scripts  
├── CMakeLists.txt      \# Build configuration  
└── pyproject.toml      \# Python packaging metadata


## **🛠️ Tech Stack & Environment**

This project is currently developed and tested using:

* **Compiler:** GCC 15.1.0 (C++17)  
* **Build System:** CMake 3.31.7  
* **Runtime:** Python 3.12.10  
* **Bindings:** [pybind11](https://github.com/pybind/pybind11)

## **📦 Installation**

Since this uses scikit-build-core, you can install it in editable mode for development:

### **Clone the repo**

```Bash
git clone https://github.com/ryanamatt/daedalus.git  
cd daedalus

# Create a Virtual Enviornment
python -m venv venv

# Activate Virtual Enviornment
venv/Scripts/activate # Windows
source venv/bin/activate # Linux

# Install in editable mode with test dependencies  
pip install -e ".[test]"
```

## **📈 Quick Start**

```Python
from daedalus import read_csv  
from daedalus.models import LinearRegression

# Load and train  
df = read_csv("tests/boston\_housing.csv")  
X = df.to_matrix(['crim', 'zn', 'indus']) # etc  
y = df.to_matrix(['medv'])

model = LinearRegression(learning_rate=0.01)  
model.fit(X, y, epochs=500)
```

See examples/ for more examples.

## **📖 Documentation**

The C++ source is documented using Doxygen-style comments. To generate the HTML documentation:

1. Ensure doxygen is installed.  
2. Run doxygen Doxyfile (if configured) or run Doxygen on the include/ directory.

## **🤝 Contributing**

Contributions are **more than welcome**\! Since this is a learning project, I'd love to see pull requests that modernize the C++ code or fix edge cases. If you find a better way to do something, please open an issue or a PR because I would love to learn. Feel free to make pull requests that add new Machine Learning Algorithms as well.

## **🏛️ About the Name**

The name **Daedalus** is inspired by the master craftsman of Greek mythology. Like Daedalus building wings of wax and feathers, this project is about building the tools to understand complex systems from the ground up. Just remember the disclaimer: if you fly too close to the sun (or a massive production dataset), things might get messy.

## **📜 License**

This library is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
