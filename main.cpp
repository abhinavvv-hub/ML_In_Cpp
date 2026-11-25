/*
    For Testing the library
*/

#include <utility>
#include <iostream>
#include "LogisticRegression.hpp"
#include "matrix.hpp"
#include "csvReader.hpp"
#include "LinearRegression.hpp"
#include "EvaluationMetric.hpp"

int32_t main() {
    try {
        Matrix dataset = CSVReader::read("IPO_Underpricing_Final.csv", true);
        Matrix test_dataset = CSVReader::read("IPO_TEST.csv", true);
        std::pair<uint16_t, uint16_t> dims = dataset.size();
        std::cout << "Dataset loaded successfully! Dimensions: " << dims.first << "x" << dims.second << "\n\n";
        if (dims.second < 2) {
            std::cerr << "Error: Dataset must have at least 2 columns (Features and Target).\n";
            return 1;
        }
        Matrix X_Train(dims.first, dims.second - 3);
        Matrix Y_Train(dims.first, 1);
        for (int i = 0; i < dims.first; ++i) {
            for (int j = 2; j < dims.second - 1; j++)
                X_Train.update(i, j - 2, dataset.peek(i, j));
            Y_Train.update(i, 0, dataset.peek(i, dims.second - 1));
        }
        std::cout << "X_Train:\n" << X_Train << '\n';
        std::cout << "Y_Train:\n" << Y_Train << '\n';
        std::cout << "Training model on data...\n";
        LogisticRegression model(X_Train, Y_Train, 0.05f , 10000);
        
        std::cout << "Model training complete!\n\n";
        dims = test_dataset.size();
        Matrix X_Test(dims.first, dims.second - 3);
        Matrix Y_Test(dims.first, 1);
        for (int i = 0; i < dims.first; ++i) {
            for (int j = 2; j < dims.second - 1; j++)
                X_Test.update(i, j - 2, test_dataset.peek(i, j));
            Y_Test.update(i, 0, test_dataset.peek(i, dims.second - 1));
        }
        Matrix Y_Pred = model.predict(X_Test);
        std::cout << "Confusion Matrix : \n" << evalMetric::GenerateConfusionMatrix(Y_Pred, Y_Test) << '\n';
        // Matrix X_Test(2, 1);
        // X_Test.update(0, 0, 10.0);
        // X_Test.update(1, 0, 7);
        // Matrix Y_Test(2, 1);
        // Y_Test.update(0, 0, 115735);
        // Y_Test.update(1, 0, 97300);
        // Matrix Y_Pred = model.predict(X_Test);
        // std::cout << "Predicted Salary on Training Data: \n" << Y_Pred << "\n";
        // std::cout << "Maximum Absolute error : " << evalMetric::RootMeanSquaredError(Y_Test, Y_Pred) << '\n';
    }
    catch (const std::exception& e) {
        std::cerr << "Runtime Exception caught: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
