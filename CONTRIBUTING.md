# **Contributing to Daedalus 🕊️**

First off, thank you for considering contributing to Daedalus\! This project is a sandbox for learning how C++, Python, and Machine Learning math intersect. Whether you're a C++ veteran helping modernize the code or a fellow learner adding your first algorithm, your help is appreciated.

By participating in this project, you agree to abide by our [Code of Conduct](https://www.google.com/search?q=CODE_OF_CONDUCT.md).

## **🛠️ Technical Setup**

Daedalus uses a hybrid stack. To contribute code, you'll need to handle both C++ and Python environments.

### **Prerequisites**

* **C++ Compiler:** GCC 15+ or Clang equivalent (supporting C++17).  
* **Build System:** CMake 3.20+  
* **Python:** 3.12+  
* **Bindings:** [pybind11](https://github.com/pybind/pybind11) (managed via pyproject.toml)

### **Development Workflow**

1. **Fork and Clone:**

```{Bash}  
   git clone git@github.com:ryanamatt/Daedalus.git
   cd daedalus
```

2. **Environment:**  

We recommend using a virtual environment to manage dependencies:

```{Bash}  
   python -m venv venv  
   # Activate (Windows)  
   venv\Scripts\activate  
   # Activate (Linux/macOS)  
   source venv/bin/activate
```

3. **Editable Install:**  
   Since we use scikit-build-core, you can install the library in editable mode. This allows you to run Python tests that call your compiled C++ code:  

``` {Bash}
   pip install -e ".[test]"
```

Or you may use the clean_build scripts which removes compiled binaries and rebuilds the project in editable mode.

```{Bash}
# Linux/MacOS
tools/clean_build.sh

# Windows
tools/clean_build.ps1
```

   *Note: Re-running this command may be necessary if you make significant changes to the C++ headers or CMake logic.*

## **📁 Project Structure**

* include/daedalus/: C++ Headers (The "What").  
* src/: C++ Implementations (The "How").  
* src/bindings/: The pybind11 glue code connecting C++ to Python.  
* daedalus/: The Python package and type stubs.  
* tests/: Python-based test suite.

## **🧪 Testing**

We use pytest for our testing suite. Before submitting a PR, please ensure all tests pass:

```{Bash}
pytest
```

If you add a new model (e.g., DecisionTree), please add a corresponding test file in tests/ to verify the math works as expected.

## **✍️ Coding Standards**

### **C++ Guidelines**

* **Modern C++:** Aim for idiomatic C++17. Prefer std::vector and smart pointers over raw arrays/pointers.  
* **Documentation:** Use Doxygen-style comments in header files (.h).

```{C++}  
  /** 
   * @brief Performs matrix multiplication.  
   * @param other The matrix to multiply with.  
   * @return A new Matrix result.  
   */  
  Matrix multiply(const Matrix& other) const;
```

* **Performance:** Since this is a learning project, focus on readability first, then optimization (e.g., cache-aware loops).

### **Python Guidelines**

* Follow PEP 8 where possible.  
* Ensure any new C++ features are exposed in the daedalus\_cpp bindings and updated in the .pyi stubs for IDE support.

## **🚀 How to Contribute**

1. **Find an Issue:** Check the [Issues](https://github.com/ryanamatt/Daedalus/issues) tab for bugs or feature requests.  
2. **Open a PR:** Use the provided Pull Request template.  

## **🏛️ Attribution**

This project is inspired by the desire to understand ML from the ground up. Thank you for helping build these wings\!