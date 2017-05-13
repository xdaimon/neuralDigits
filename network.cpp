#include "network.h"
#include <thread>
#include <functional>
using namespace NeuralNetwork;

#define THREADS 8

Network::Network(const VectorXi& layerSizes) {
	std::default_random_engine gen(53360);
	std::normal_distribution<double> dist(0., 1.);

	// TODO initialize matricies with vector in order to use nicer c++ routines for initialization
	// than the loops?
	numLayers = layerSizes.size();
	for (int i = 1; i < numLayers; ++i) {
		W.push_back(MatrixXd::Zero(layerSizes(i), layerSizes(i - 1)));
		B.push_back(VectorXd::Zero(layerSizes(i)));
	}
	for (int i = 0; i < numLayers - 1; ++i) {
		for (int j = 0; j < W[i].rows(); ++j)
			for (int k = 0; k < W[i].cols(); ++k)
				W[i](j, k) = dist(gen);
		for (int j = 0; j < B[i].rows(); ++j)
			B[i](j) = dist(gen);
	}
}

MatrixXd Network::forward(MatrixXd X) {
	for (int i = 0; i < numLayers - 1; ++i)
		X = sigmoid((W[i] * X).colwise() + B[i]);
	return X;
}

double Network::SGD(const Data& train_data, const Data& test_data, int epochs, int mini_batch_size,
                    double learning_rate) {
	const int dimension = train_data.examples.rows();
	const int num_examples = train_data.examples.cols();
	const int num_batches = num_examples / mini_batch_size;
	vector<Data> mini_batch(THREADS);
	for (int i = 0; i < THREADS; ++i) {
		mini_batch[i].examples = MatrixXd::Zero(dimension, mini_batch_size);
		mini_batch[i].labels = VectorXi::Zero(mini_batch_size);
	}

	// fill random_indices with increasing numbers starting from 0
	vector<int> random_indices(num_examples);
	int n = 0;
	std::generate(random_indices.begin(), random_indices.end(), [&n] { return n++; });

	for (int j = 0; j < epochs; ++j) {
		std::random_shuffle(random_indices.begin(), random_indices.end());
		int i = 0;
		while (i < num_batches-THREADS) {
			for (int l = 0; l < THREADS; ++l) {
				// Make a minibatch
				for (int k = 0; k < mini_batch_size; ++k) {
					mini_batch[l].examples.col(k) = train_data.examples.col(random_indices[k + mini_batch_size * i]);
					mini_batch[l].labels(k) = train_data.labels(random_indices[k + mini_batch_size * i]);
				}
				++i;
			}
			train_for_mini_batch(mini_batch, learning_rate);
		}
		cout << "Epoch " << j << " : " << evaluate(test_data) << "% accuracy" << endl << endl;
	}
	return evaluate(test_data);
}

void Network::train_for_mini_batch(vector<Data>& mini_batch, double learning_rate) {
	vector<vector<MatrixXd> > dW(THREADS); // mini_batch.size()
	vector<vector<VectorXd> > dB(THREADS);
	for (int j = 0; j < THREADS; ++j) {
		for (int i = 0; i < numLayers - 1; ++i) {
			dW[j].push_back(MatrixXd::Zero(W[i].rows(), W[i].cols()));
			dB[j].push_back(VectorXd::Zero(B[i].rows()));
		}
	}

	vector<std::thread> ts(THREADS);
	for (int i = 0; i < THREADS; ++i) {
		ts[i] = std::thread(&Network::backprop, this, std::ref(mini_batch[i].examples),std::ref(mini_batch[i].labels),std::ref(dW[i]),std::ref(dB[i]));
	}
	for (int i = 0; i < THREADS; ++i)
		ts[i].join();

	const int mini_batch_size = mini_batch[0].labels.rows();
	for (int j = 0; j < THREADS; ++j) {
		for (int i = 0; i < numLayers - 1; ++i) {
			W[i] -= (learning_rate * (dW[j][i] / double(mini_batch_size)));
			B[i] -= (learning_rate * (dB[j][i] / double(mini_batch_size)));
		}
	}
}

void Network::backprop(MatrixXd& X, VectorXi& Y, vector<MatrixXd>& dW, vector<VectorXd>& dB) {
	int l = 0;
	vector<MatrixXd> A = {X};
	while (l < numLayers - 1) {
		A.push_back(sigmoid((W[l] * A[l]).colwise() + B[l]));
		++l;
	}

	MatrixXd d_l = Cost_derivative(A[l], Y).array() * sigmoid_derivative(A[l]).array();
	dW[l-1] += d_l * A[l-1].transpose();
	for (int i = 0; i < d_l.cols(); ++i)
		dB[l-1] += d_l.col(i);
	--l;

	while (l >= 1) {
		d_l = (W[l].transpose() * d_l).eval().array() * sigmoid_derivative(A[l]).array();
		dW[l-1] += d_l * A[l-1].transpose();
		for (int i = 0; i < d_l.cols(); ++i)
			dB[l-1] += d_l.col(i);
		--l;
	}
}

double Network::evaluate(const Data& test_data) {
	VectorXi test_results = argmax(forward(test_data.examples)) - test_data.labels;
	int num_correct = 0;
	for (int i = 0; i < test_results.size(); ++i)
		if (test_results(i) == 0)
			num_correct++;
	return 100.*num_correct/test_results.size();
}

MatrixXd Network::Cost_derivative(MatrixXd A_L, const VectorXi y) {
	for (int i = 0; i < A_L.cols(); ++i)
		A_L.col(i)(y(i)) -= 1.;
	return A_L;
}

VectorXi Network::argmax(const MatrixXd&& V) {
	VectorXi Y(V.cols());
	for (int i = 0; i < V.cols(); ++i)
		(void)V.col(i).maxCoeff(&Y(i));
	return Y;
}

MatrixXd Network::sigmoid(const MatrixXd& z) {
	return 1. / (1. + (-z.array()).exp());
}

// Takes o(z) and outputs (1-o(z))*o(z)
MatrixXd Network::sigmoid_derivative(const MatrixXd& z) {
	return (z.array() * (1.0 - z.array()));
}
