#include "network.h"
#include <thread>
using namespace NeuralNetwork;

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
				W[i](j, k) = dist(gen)/sqrt(W[i].cols());
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
                    double learning_rate, double regularization, int threads) {
	const int dimension = train_data.examples.rows();
	const int num_examples = train_data.examples.cols();
	const int num_batches = num_examples/mini_batch_size;
	vector<Data> mini_batch(threads);
	for (int t = 0; t < threads; ++t) {
		mini_batch[t].examples = MatrixXd::Zero(dimension, mini_batch_size/threads);
		mini_batch[t].labels = VectorXi::Zero(mini_batch_size/threads);
	}

	// fill random_indices with increasing numbers starting from 0
	vector<int> random_indices(num_examples);
	int n = 0;
	std::generate(random_indices.begin(), random_indices.end(), [&n] { return n++; });


	for (int j = 0; j < epochs; ++j) {
		auto prev_time = std::chrono::steady_clock::now();
		std::random_shuffle(random_indices.begin(), random_indices.end());
		for (int i = 0; i < num_batches; ++i) {
			// Make a minibatch, each thread processes part of the minibatch
			for (int t = 0; t < threads; ++t) {
				for (int k = 0; k < mini_batch_size/threads; ++k) {
					mini_batch[t].examples.col(k) = train_data.examples.col(random_indices[k + mini_batch_size * i + mini_batch_size/threads*t]);
					mini_batch[t].labels(k) = train_data.labels(random_indices[k + mini_batch_size * i + mini_batch_size/threads*t]);
				}
			}
			train_for_mini_batch(mini_batch, mini_batch_size, learning_rate, regularization, threads);
		}
		// cout << j<<"\t"<<evaluate(test_data) << "% \t accuracy ";
		cout << (std::chrono::steady_clock::now() - prev_time).count()/10000 << endl;
	}
	return evaluate(test_data);
}

void Network::train_for_mini_batch(vector<Data>& mini_batch, int mini_batch_size, double learning_rate, double regularization, int threads) {
	vector<vector<MatrixXd> > dW(threads); // mini_batch.size()
	vector<vector<VectorXd> > dB(threads);
	for (int t = 0; t < threads; ++t) {
		for (int l = 0; l < numLayers - 1; ++l) {
			dW[t].push_back(MatrixXd::Zero(W[l].rows(), W[l].cols()));
			dB[t].push_back(VectorXd::Zero(B[l].rows()));
		}
	}

	vector<std::thread> ts(threads);
	for (int t = 0; t < threads; ++t) {
		ts[t] = std::thread(&Network::backprop, this, std::ref(mini_batch[t].examples),std::ref(mini_batch[t].labels),std::ref(dW[t]),std::ref(dB[t]));
	}
	for (int t = 0; t < threads; ++t)
		ts[t].join();

	for (int l = 0; l < numLayers - 1; ++l) {
		for (int t = 0; t < threads; ++t) {
			W[l] -= learning_rate/double(mini_batch_size) * dW[t][l];
			B[l] -= learning_rate/double(mini_batch_size) * dB[t][l];
		}
		W[l] -= regularization * W[l];
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
