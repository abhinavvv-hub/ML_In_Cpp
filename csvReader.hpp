#pragma once

#ifndef csvReader_hpp
#define csvReader_hpp

#include "matrix.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

class CSVReader {
public:
    static Matrix read(const std::string& filename, bool skipHeader = false) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file!");
        }
        std::string line;
        int rows = 0;
        int cols = 0;
        if (skipHeader && std::getline(file, line)) {}
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (rows == 0) {
                std::stringstream ss(line);
                std::string cell;
                while (std::getline(ss, cell, ',')) {
                    cols++;
                }
            }
            rows++;
        }
        if (rows == 0 || cols == 0) {
            throw std::runtime_error("No valid data found!");
        }
        Matrix mat(rows, cols);
        file.clear(); 
        file.seekg(0, std::ios::beg);
        if (skipHeader && std::getline(file, line)) {}
        size_t i = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                mat.grid[i++] = std::stof(cell); 
            }
        }
        file.close();
        return mat;
    }
};

#endif
