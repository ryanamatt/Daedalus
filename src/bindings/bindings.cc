// src/bindings/bindings.cc

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "daedalus/core/Matrix.h"
#include "daedalus/core/Metrics.h"
#include "daedalus/core/DataFrame.h"
#include "daedalus/core/IO.h"
#include "daedalus/core/Preprocessing.h"
#include "daedalus/core/Metrics.h"
#include "daedalus/core/Utils.h"
#include "daedalus/models/linearRegression.h"
#include "daedalus/models/logisticRegression.h"
#include "daedalus/models/knn.h"
#include "daedalus/models/DenseLayer.h"
#include "daedalus/models/NeuralNetwork.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Daedalus: A Machine Learning library";

    // --- Matrix Bindings ---
    py::class_<Matrix<double>>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def("__getitem__", [](const Matrix<double> &self, py::object index_obj) -> py::object {
            // Check if the input is actually a tuple
            if (!py::isinstance<py::tuple>(index_obj)) {
                throw py::index_error("Matrix indices must be a 2-tuple");
            }

            py::tuple index = index_obj.cast<py::tuple>();
            if (index.size() != 2) {
                throw py::index_error("Matrix indices must be a 2-tuple");
            }

            auto parse_index = [&](py::object item, size_t max_size) -> std::pair<size_t, size_t> {
                if (py::isinstance<py::slice>(item)) {
                    py::slice s(item);
                    size_t start, stop, step, length;
                    if (!s.compute(max_size, &start, &stop, &step, &length))
                        throw py::error_already_set();
                    return {start, stop};
                } 
                
                if (!py::isinstance<py::int_>(item)) {
                    throw py::type_error("Matrix indices must be integers or slices");
                }

                int idx = item.cast<int>();
                if (idx < 0) py::index_error("Index can't be negative");
                if (idx < 0 || (size_t)idx >= max_size) throw py::index_error("Index out of range");
                return {(size_t)idx, (size_t)idx + 1};
            };

            auto rows = parse_index(index[0], self.rows());
            auto cols = parse_index(index[1], self.cols());

            // If both were integers, return a single float value
            if (!py::isinstance<py::slice>(index[0]) && !py::isinstance<py::slice>(index[1])) {
                return py::cast(self(rows.first, cols.first));
            }

            // Otherwise return a sub-matrix
            return py::cast(self.get_slice(rows.first, rows.second, cols.first, cols.second));
        })
        .def("__setitem__", [](Matrix<double> &self, py::tuple index, double value) {
            if (index.size() != 2) throw py::index_error("Matrix indices must be a 2-tuple");
            
            // We only support single-element assignment for now based on your request
            int r = index[0].cast<int>();
            int c = index[1].cast<int>();

            // Handle negative indexing
            if (r < 0) r += self.rows();
            if (c < 0) c += self.cols();

            if (r < 0 || (size_t)r >= self.rows() || c < 0 || (size_t)c >= self.cols()) {
                throw py::index_error("Index out of range");
            }

            self(r, c) = value;
        })
        .def("__repr__", &Matrix<double>::to_string)
        .def_property_readonly("rows", &Matrix<double>::rows)
        .def_property_readonly("cols", &Matrix<double>::cols)
        .def("get_row", &Matrix<double>::get_row, py::arg("idx"))
        .def("__call__", [](Matrix<double> &self, size_t r, size_t c) { return self(r, c); }, 
             py::arg("r"), py::arg("c"))
        .def("set", [](Matrix<double> &self, size_t r, size_t c, double val) { self(r, c) = val; }, 
             py::arg("r"), py::arg("c"), py::arg("val"))
        .def("transpose", &Matrix<double>::transpose)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self * double())
        .def(double() * py::self);

    // --- DataFrame Bindings ---
    py::class_<DataFrame>(m, "DataFrame")
        .def(py::init<>())
        .def(py::init<const std::string&, const std::vector<std::variant<double, int, std::string>>&>(), 
             py::arg("col_name"), py::arg("col_data"),
             "Constructs a DataFrame with an initial column.")
        .def_property_readonly("rows", &DataFrame::rows)
        .def_property_readonly("cols", &DataFrame::cols)
        .def("get_column_names", &DataFrame::get_column_names)
        .def("at", py::overload_cast<size_t, const std::string&>(&DataFrame::at, py::const_), 
            py::arg("row"), py::arg("col_name"))
        .def("at", py::overload_cast<size_t, size_t>(&DataFrame::at, py::const_), 
            py::arg("row"), py::arg("col"))
        .def("__repr__", &DataFrame::to_string)
        .def("head", &DataFrame::head, py::arg("n") = 5, "Returns the first n rows of the DataFrame.")
        .def("add_column", &DataFrame::add_column, py::arg("name"), py::arg("col_data"))
        .def("drop_column", &DataFrame::drop_column, py::arg("name"))
        .def("filter", [](const DataFrame& self, const std::string& col_name, py::function predicate) {
        return self.filter(col_name, [predicate](const std::variant<double, int, std::string>& val) {
            return predicate(val).cast<bool>();
        });
    }, py::arg("col_name"), py::arg("predicate"), "Filters rows using a Python lambda or function.")
        .def("encode_binary", &DataFrame::encode_binary, 
            py::arg("column_name"), 
            py::arg("val0") = "", 
            py::arg("val1") = "")
        .def("to_matrix", &DataFrame::to_matrix, py::arg("target_columns"));

    // --- IO Bindings ---
    m.def("read_csv", &read_csv, py::arg("filename"), py::arg("has_header") = true);

    // --- Preprocessing Bindings ---
    py::class_<StandardScaler>(m, "StandardScaler")
        .def(py::init<>())
        .def("fit", &StandardScaler::fit, py::arg("X"))
        .def("transform", &StandardScaler::transform, py::arg("X"))
        .def("fit_transform", [](StandardScaler &self, const Matrix<double> &X) {
            self.fit(X);
            return self.transform(X);
        }, py::arg("X"));

    // --- Metric Bindings ---
    m.def("mean_squared_error", &Metrics::mean_squared_error, 
          py::arg("y_true"), py::arg("y_pred"), "Calculates Mean Squared Error");
    m.def("r2_score", &Metrics::r2_score, 
          py::arg("y_true"), py::arg("y_pred"), "Calculates R-Squared Score");
    m.def("accuracy_score", &Metrics::accuracy_score, py::arg("y_true"), py::arg("y_pred"));
    m.def("precision_score", &Metrics::precision_score, py::arg("y_true"), py::arg("y_pred"));
    m.def("recall_score", &Metrics::recall_score, py::arg("y_true"), py::arg("y_pred"));
    m.def("f1_score", &Metrics::f1_score, py::arg("y_true"), py::arg("y_pred"));

    // --- Utils Bindings ---
    m.def("train_test_split", [](const Matrix<double>& X, const Matrix<double>& y, double test_size, int seed) {
        return train_test_split(X, y, test_size, seed);
    }, py::arg("X"), py::arg("y"), py::arg("test_size") = 0.2, py::arg("seed") = 42,
    "Splits features and targets into training and testing sets.");

    // --- Linear Regression Bindings ---
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<double, double, std::string>(), py::arg("learning_rate") = 0.01, py::arg("reg_lambda") = 0.01, 
            py::arg("penalty") = "none")
        .def("fit", py::overload_cast<const Matrix<double>&, const Matrix<double>&>(&LinearRegression::fit),
             py::arg("X"), py::arg("y"))
        .def("fit", py::overload_cast<const Matrix<double>&, const Matrix<double>&, int>(&LinearRegression::fit),
             py::arg("X"), py::arg("y"), py::arg("epochs"))
        .def("predict", &LinearRegression::predict, py::arg("X"))
        .def("save_model", &LinearRegression::saveModel, py::arg("filename"))
        .def("load_model", &LinearRegression::loadModel, py::arg("filename"));

    // --- Logistic Regression Bindings
    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<double, double, std::string>(), 
         py::arg("learning_rate") = 0.01, 
         py::arg("reg_lambda") = 0.01, 
         py::arg("penalty") = "none")
        .def("fit", py::overload_cast<const Matrix<double>&, const Matrix<double>&>(&LogisticRegression::fit),
             py::arg("X"), py::arg("y"))
        .def("fit", py::overload_cast<const Matrix<double>&, const Matrix<double>&, int>(&LogisticRegression::fit),
             py::arg("X"), py::arg("y"), py::arg("epochs"))
        .def("predict", &LogisticRegression::predict, py::arg("X"))
        .def("predict_proba", &LogisticRegression::predict_proba, py::arg("X"))
        .def("save_model", &LogisticRegression::saveModel, py::arg("filename"))
        .def("load_model", &LogisticRegression::loadModel, py::arg("filename"));

    // --- KNN Model Bindings ---
    py::class_<KNN>(m, "KNN")
        .def(py::init<int>(), py::arg("k") = 3)
        .def("fit", &KNN::fit, py::arg("X"), py::arg("y"))
        .def("predict", &KNN::predict, py::arg("X"));

    // --- Neural Network Bindings ---
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<double>(), py::arg("lr") = 0.01)
        .def("add", [](NeuralNetwork &nn, int in, int out) {
            nn.add(std::make_unique<DenseLayer>(in, out));
        })
        .def("fit", py::overload_cast<const Matrix<double>&, const Matrix<double>&>(&NeuralNetwork::fit),
             py::arg("X"), py::arg("y"))
        .def("fit", py::overload_cast<const Matrix<double>&, const Matrix<double>&, int>(&NeuralNetwork::fit),
             py::arg("X"), py::arg("y"), py::arg("epochs"))
        .def("predict", &NeuralNetwork::predict, py::arg("X"));
}