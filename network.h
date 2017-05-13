#pragma once

#include <iostream>
using std::cout;
using std::endl;
#include <vector>
using std::vector;
#include <limits>
#include <random>
#include <numeric>

#include <eigen3/Eigen/Eigen>
using namespace Eigen;

#include "Data.h"

namespace NeuralNetwork {
class Network {
  public:
	Network(const VectorXi& layerSizes);
	double SGD(const Data& train_data, const Data& test_data, int epochs, int mini_batch_size,
	           double learning_rate);

	double evaluate(const Data& test_data);
  private:
	int numLayers;
	vector<MatrixXd> W;
	vector<VectorXd> B;

	MatrixXd forward(MatrixXd X);
	void train_for_mini_batch(const Data& mini_batch, double learning_rate);
	void backprop(const MatrixXd& x, const VectorXi Y, vector<MatrixXd>& dW, vector<VectorXd>& dB);
	MatrixXd Cost_derivative(MatrixXd A_L, const VectorXi Y);

	MatrixXd sigmoid(const MatrixXd& z);
	MatrixXd sigmoid_derivative(const MatrixXd& z);
	VectorXi argmax(const MatrixXd&& V);
};
}
