/*
    GUIDE
    import this as #include "matrix.hpp" (Ensure you have ActivationFunctions.hpp in the same hierarchy level as this file)
    Create a Matrix with the constructor as "Matrix mat(2, 3)" which create a matrix of size 2 x 3.
*/

#pragma once

#ifndef matrix_hpp
#define matrix_hpp

#include <random>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include "ActivationFunctions.hpp"

class Matrix {
    uint16_t rows, cols;
    float *grid;

    static std::mt19937& get_shared_rng() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }

public:
    explicit Matrix(int rows_, int cols_) : rows(rows_), cols(cols_) {
        grid = new float[rows * cols](); 
    }

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        grid = new float[rows * cols];
        for (int i = 0; i < rows * cols; ++i) {
            grid[i] = other.grid[i];
        }
    }

    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), grid(other.grid) {
        other.grid = nullptr;
        other.rows = 0;
        other.cols = 0;
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            delete[] grid;
            rows = other.rows;
            cols = other.cols;
            grid = new float[rows * cols];
            for (int i = 0; i < rows * cols; ++i) {
                grid[i] = other.grid[i];
            }
        }
        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            delete[] grid;
            rows = other.rows;
            cols = other.cols;
            grid = other.grid;
            other.grid = nullptr;
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }

    std::pair<uint16_t, uint16_t> size() const {
        return std::make_pair(rows, cols);
    }

    void fill(float x = 0.0) {
        for(size_t i = 0; i < rows * cols; i++) {
            grid[i] = x;
        }
    }

    void randFill(float lowerLimit = 0.0, float upperLimit = 1.0) {
        std::uniform_real_distribution<float> dist(lowerLimit, upperLimit);
        auto& rng = get_shared_rng();
        
        for(size_t i = 0; i < rows * cols; i++) {
            grid[i] = dist(rng);    
        }
    }

    float peek(int16_t rowIdx, int16_t colIdx) const {
        if (rowIdx < 0 || colIdx < 0 || rowIdx >= rows || colIdx >= cols) {
            throw std::invalid_argument("Index out of Bounds!");
        }
        return grid[cols * rowIdx + colIdx]; 
    }

    void update(int16_t rowIdx, int16_t colIdx, float val) {
        if (rowIdx < 0 || colIdx < 0 || rowIdx >= rows || colIdx >= cols) {
            throw std::invalid_argument("Index out of Bounds!");
        }
        grid[cols * rowIdx + colIdx] = val;
    }

    friend std::ostream& operator<<(std::ostream& output, const Matrix& mat) {
        if (mat.rows == 0 || mat.cols == 0) {
            return output << "[]\n";
        }
        output << "[";
        for (size_t i = 0; i < mat.rows * mat.cols; ++i) {
            size_t col = i % mat.cols;
            if (col == 0) {
                output << "[";
            }
            output << mat.grid[i];
            if (col < mat.cols - 1) {
                output << ", ";
            }
            else {
                output << "]";
                if (i < mat.rows * mat.cols - 1) {
                    output << ",\n ";
                }
            }
        }
        output << "]\n";
        return output;
    }

    Matrix& operator+=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Size Mismatch!");
        }
        for (size_t i = 0; i < rows * cols; ++i) {
            grid[i] += other.grid[i];
        }
        return *this;
    }

    Matrix operator+(Matrix other) const {
        other += *this;
        return other;
    }

    Matrix& operator-=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Size Mismatch!");
        }
        for (size_t i = 0; i < rows * cols; ++i) {
            grid[i] -= other.grid[i];
        }
        return *this;
    }

    Matrix operator-(const Matrix& other) const {
        Matrix result = *this;
        result -= other;
        return result;
    }

    void scale(float x = 1.0) {
        for (size_t i = 0; i < rows * cols; i++) {
            grid[i] *= x;
        }
    }

    float determinant() const {
        if (rows != cols) {
            throw std::invalid_argument("Not a square matrix!");
        }
        if (rows == 0) {
            return 1.0;
        }
        float det = 1.0;
        int swaps = 0;
        float* lu = new float[rows * cols]();
        for (size_t i = 0; i < rows * cols; i++) {
            lu[i] = grid[i];
        }
        for (size_t i = 0; i < rows; i++) {
            size_t pivot = i;
            float mx = std::abs(lu[i * rows + i]);
            for (size_t r = i + 1; r < rows; r++) {
                float val = std::abs(lu[r * rows + i]);
                if (val > mx) {
                    mx = val;
                    pivot = r;
                }
            }
            if (mx < 0.000000001) {
                return 0.0;
            }
            if (pivot != i) {
                for (size_t c = 0; c < rows; c++) {
                    std::swap(lu[i * rows + c], lu[pivot * rows + c]);
                }
                swaps++;
            }
            float currDiag = lu[i * rows + i];
            det *= currDiag;
            for (size_t r = i + 1; r < rows; r++) {
                float k = lu[r * rows + i] / currDiag;
                for (size_t c = i + 1; c < rows; c++) {
                    lu[r * rows + c] -= k * lu[i * rows + c];
                }
            }
        }
        if (swaps & 1) {
            det = -det;
        }
        delete[] lu;
        return det;
    }

    Matrix inverse() const {
        if (rows != cols) {
            throw std::invalid_argument("Not a square matrix!");
        }
        if (rows == 0) {
            return Matrix(0, 0);
        }
        float* lu = new float[rows * cols]();
        uint16_t *perm = new uint16_t[rows]();
        for (size_t i = 0; i < rows * cols; i++) {
            lu[i] = grid[i];
        }
        for (size_t i = 0; i < rows; i++) {
            perm[i] = i;
        }
        for (size_t i = 0; i < rows; i++) {
            size_t pivot = i;
            float mx = std::abs(lu[i * rows + i]);
            for (size_t r = i + 1; r < rows; r++) {
                float val = std::abs(lu[r * rows + i]);
                if (val > mx) {
                    mx = val;
                    pivot = r;
                }
            }
            if (mx < 0.000000001) {
                throw std::runtime_error("Singular Matrix!");
            }
            if (pivot != i) {
                std::swap(perm[i], perm[pivot]);
                for (size_t c = 0; c < rows; c++) {
                    std::swap(lu[i * rows + c], lu[pivot * rows + c]);
                }
            }
            float currDiag = lu[i * rows + i];
            for (size_t r = i + 1; r < rows; r++) {
                float k = lu[r * rows + i] / currDiag;
                lu[r * rows + i] = k;
                for (size_t c = i + 1; c < rows; c++) {
                    lu[r * rows + c] -= k * lu[i * rows + c];
                }
            }
        }
        Matrix inv(rows, rows);
        for (size_t k = 0; k < rows; k++) {
            float* x = new float[rows]();
            for (size_t i = 0; i < rows; i++) {
                x[i] = perm[i] == k ? 1.0 : 0.0;
                for (size_t j = 0; j < i; j++) {
                    x[i] -= lu[i * rows + j] * x[j];
                }
            }
            for (int16_t i = rows - 1; i >= 0; i--) {
                for (size_t j = i + 1; j < rows; j++) {
                    x[i] -= lu[i * rows + j] * x[j];
                }
                x[i] = x[i] / lu[i * rows + i];
            }
            for (size_t i = 0; i < rows; i++) {
                inv.grid[i * rows + k] = x[i];
            }
            delete[] x;
        }
        return inv;
    }

    Matrix transpose() const {
        Matrix trans(cols, rows);
        for (int i = 0; i < rows * cols; i++) {
            int r = i / cols, c = i % cols;
            trans.grid[(i - r * cols) * rows + r] = grid[i];
        }
        return trans;
    }

    Matrix& operator*=(const Matrix& other) {
        if (cols != other.rows) {
            throw std::invalid_argument("Size Mismatch for Matrix Multiplication!");
        }
        int nCols = other.cols;
        float* ngrid = new float[rows * nCols](); 
        const int BLOCK_SIZE = 32; 
        for (int sj = 0; sj < nCols; sj += BLOCK_SIZE) {
            for (int sk = 0; sk < cols; sk += BLOCK_SIZE) {
                for (int i = 0; i < rows; i++) {
                    int k1 = std::min(sk + BLOCK_SIZE, (int)cols);
                    for (int k = sk; k < k1; k++) {
                        int j1 = std::min(sj + BLOCK_SIZE, nCols);
                        for (int j = sj; j < j1; j++) {
                            ngrid[i * nCols + j] += grid[i * cols + k] * other.grid[k * other.cols + j];
                        }
                    }
                }
            }
        }
        delete[] grid;
        grid = ngrid;
        cols = nCols;
        return *this;
    }

    Matrix operator*(const Matrix& other) const {
        Matrix result(*this);
        result *= other;
        return result;
    }

    Matrix ReLU() {
        Matrix ans(rows, cols);
        for (int i = 0; i < rows * cols; i++) {
            ans.grid[i] = actFn::ReLU(grid[i]);
        }
        return ans;
    }

    Matrix Sigmoid() {
        Matrix ans(rows, cols);
        for (int i = 0; i < rows * cols; i++) {
            ans.grid[i] = actFn::Sigmoid(grid[i]);
        }
        return ans;
    }

    friend class CSVReader;


    ~Matrix() {
        delete[] grid;
        rows = 0, cols = 0;
    }
};

#endif
