#pragma once

#include <iostream>
#include <limits>
#include <ctime>

#include <eigen3/Eigen/Eigen>
using namespace Eigen;

#include "Data.h"

using std::vector;
using std::cout;
using std::endl;

namespace NeuralNetwork
{
	class Network
	{
		public:
			Network(const VectorXi & layer_sizes);
			double SGD(const Data & train_data, const Data & test_data,
			           int epochs, int mini_batch_size, double learning_rate);

		private:
			int number_of_layers;
			vector<MatrixXd> weights;
			vector<VectorXd> biases;

			VectorXd feed_forward(VectorXd&& x);
			void train_for_mini_batch(const Data & mini_batch, double learning_rate);
			void backprop(const VectorXd & x, int y, vector<MatrixXd>& nabla_w, vector<VectorXd>& nabla_b);
			double evaluate(const Data & test_data);
			VectorXd cost_derivative(VectorXd output_activations, int y);

			VectorXd sigmoid(const VectorXd & z);
			VectorXd Dsigmoid(const VectorXd & z);
			int argmax(const VectorXd & v);
	};
}
