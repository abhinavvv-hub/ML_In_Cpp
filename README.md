# ML In C++
Wrote a Machine Learning Library in C++

Edit the main.cpp file to test your data.

> NOTE : No Support for non-numeric type data..Use preprocess.py to handle it or bind it to numeric values

Make sure all files are at the same hierarchy level.


C++ Version Used: C++23

Compile using : g++ -std=c++23 -Wall main.cpp -o main

### Features

1. "Matrix" class is optimized using flat array and uses batching.
2. "LinearRegression" uses Ordinary Least Squares to compute weights and biases.
3. "LogisticRegression" uses gradient descent with sigmoid.
4. "CSVReader" class reads the csv as the "Matrix" class.
