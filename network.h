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

	Network(VectorXi& layerSizes);

	double SGD(const Data& train_data, const Data& test_data, int epochs, int mini_batch_size,
	           double learning_rate, double regularization, double momentum, int threads);

	double evaluate(const Data& test_data);
  private:

	std::default_random_engine rand_engine;
	std::normal_distribution<double> dist;

	int numLayers;
	vector<MatrixXd> W;
	vector<VectorXd> B;

	MatrixXd forward(MatrixXd X);
	void train_for_mini_batch(vector<Data>& mini_batch, int mini_batch_size, double learning_rate, double regularization, double momentum, int threads, vector<vector<MatrixXd>>& prev_dW, vector<vector<VectorXd>>& prev_dB, vector<vector<MatrixXd>>& bprev_dW, vector<vector<VectorXd>>& bprev_dB);
	void backprop(MatrixXd& X, VectorXi& Y, vector<MatrixXd>& dW, vector<VectorXd>& dB);

	MatrixXd quadratic_cost_derivative(MatrixXd A_L, const VectorXi Y);
	MatrixXd cross_entropy_cost_derivative(MatrixXd A_L, const VectorXi Y);
	MatrixXd loglikelihood_cost_derivative(MatrixXd A_L, const VectorXi Y);

	MatrixXd softmax_derivative(VectorXd&& activations);
	MatrixXd softmax(MatrixXd&& z);

	MatrixXd sigmoid(const MatrixXd& z);
	MatrixXd sigmoid_derivative(const MatrixXd& activations);

	MatrixXd relu(const MatrixXd& z);
	MatrixXd relu_derivative(const MatrixXd& activations);

	VectorXi argmax(const MatrixXd&& V);
};
}
