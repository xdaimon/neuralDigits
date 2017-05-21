#include "network.h"
#include <thread>
using namespace NeuralNetwork;

// #define neuron softmax
// #define neuron relu
#define neuron sigmoid

Network::Network(VectorXi& layerSizes) {
	rand_engine = std::default_random_engine(43360);
	dist = std::normal_distribution<double>(0., 1.);

	numLayers = layerSizes.rows();
	for (int i = 1; i < numLayers; ++i) {
		W.push_back(MatrixXd::Zero(layerSizes(i), layerSizes(i - 1)));
		B.push_back(VectorXd::Zero(layerSizes(i)));
	}

	for (int i = 0; i < numLayers - 1; ++i) {
		for (int j = 0; j < W[i].rows(); ++j)
			for (int k = 0; k < W[i].cols(); ++k)
				W[i](j, k) = dist(rand_engine)/sqrt(W[i].cols()); // initialize so that on average the size of input to neurons is not too large
		for (int j = 0; j < B[i].rows(); ++j)
			B[i](j) = dist(rand_engine);
	}
}

MatrixXd Network::forward(MatrixXd X) {
	for (int i = 0; i < numLayers - 1; ++i)
		X = neuron((W[i] * X).colwise() + B[i]);
	return X;
}

double Network::SGD(const Data& train_data, const Data& test_data, int epochs, int mini_batch_size,
                    double learning_rate, double regularization, double momentum, int threads) {
	const int dimension = train_data.examples.rows();
	const int num_examples = train_data.examples.cols();
	const int num_batches = num_examples/mini_batch_size;
	vector<Data> mini_batch(threads);
	for (int t = 0; t < threads; ++t) {
		mini_batch[t].examples = MatrixXd::Zero(dimension, mini_batch_size/threads);
		mini_batch[t].labels = VectorXi::Zero(mini_batch_size/threads);
	}
	vector<vector<MatrixXd> > dW(threads);
	vector<vector<VectorXd> > dB(threads);
	vector<vector<MatrixXd> > bdW(threads);
	vector<vector<VectorXd> > bdB(threads);
	for (int t = 0; t < threads; ++t) {
		for (int l = 0; l < numLayers - 1; ++l) {
			dW[t].push_back(MatrixXd::Zero(W[l].rows(), W[l].cols()));
			dB[t].push_back(VectorXd::Zero(B[l].rows()));
			bdW[t].push_back(MatrixXd::Zero(W[l].rows(), W[l].cols()));
			bdB[t].push_back(VectorXd::Zero(B[l].rows()));
		}
	}

	// fill random_indices with increasing numbers starting from 0
	vector<int> random_indices(num_examples);
	int n = 0;
	std::generate(random_indices.begin(), random_indices.end(), [&n] { return n++; });

	vector<double> accuracies;

	for (int j = 0; j < epochs; ++j) {
		auto prev_time = std::chrono::steady_clock::now();
		// shuffle to avoid biasing the training examples
		std::shuffle(random_indices.begin(), random_indices.end(), rand_engine);
		for (int i = 0; i < num_batches; ++i) {
			// Make a minibatch, each thread processes part of the minibatch
			for (int t = 0; t < threads; ++t) {
				for (int k = 0; k < mini_batch_size/threads; ++k) {
					mini_batch[t].examples.col(k) = train_data.examples.col(random_indices[k + mini_batch_size * i + mini_batch_size/threads*t]);
					mini_batch[t].labels(k) = train_data.labels(random_indices[k + mini_batch_size * i + mini_batch_size/threads*t]);
				}
			}
			train_for_mini_batch(mini_batch, mini_batch_size, learning_rate, regularization, momentum, threads, dW, dB, bdW, bdB);
		}
		accuracies.push_back(evaluate(test_data));

		// static int last = 0;
		// if (j >last+ 10) {
		// 	double sss = accuracies.back()-accuracies.rbegin()[10];
		// 	if (sss < 1) {
		// 		learning_rate /= 2.;
		// 	last=j;
		// 		cout << sss<<"\t";
		// 	}
		// }
		// cout << learning_rate << "\t" << j <<"\t"<< accuracies.back() << "% \t accuracy ";

		cout << j <<"\t"<< accuracies.back() << "% \t accuracy ";
		cout << (std::chrono::steady_clock::now() - prev_time).count()/10000 << endl;
	}

	return evaluate(test_data);
}

void Network::train_for_mini_batch(vector<Data>& mini_batch, int mini_batch_size, double learning_rate, double regularization, double momentum, int threads, vector<vector<MatrixXd>>& prev_dW, vector<vector<VectorXd>>& prev_dB, vector<vector<MatrixXd>>& bprev_dW, vector<vector<VectorXd>>& bprev_dB) {
	vector<vector<MatrixXd> > dW(threads);
	vector<vector<VectorXd> > dB(threads);
	for (int t = 0; t < threads; ++t) {
		for (int l = 0; l < numLayers - 1; ++l) {
			dW[t].push_back(MatrixXd::Zero(W[l].rows(), W[l].cols()));
			dB[t].push_back(VectorXd::Zero(B[l].rows()));
		}
	}

	vector<std::thread> ts(threads);
	for (int t = 0; t < threads; ++t)
		ts[t] = std::thread(&Network::backprop, this, std::ref(mini_batch[t].examples), std::ref(mini_batch[t].labels), std::ref(dW[t]), std::ref(dB[t]));
	for (int t = 0; t < threads; ++t)
		ts[t].join();

// #define MIX_MACRO(_x,_y,_m)

	const double db = .4;
	for (int l = 0; l < numLayers - 1; ++l) {
		for (int t = 0; t < threads; ++t) {
			MatrixXd tempW = prev_dW[t][l];
			MatrixXd tempB = prev_dB[t][l];
			prev_dW[t][l]=((prev_dW[t][l] + bprev_dW[t][l])*(1-momentum) + learning_rate/double(mini_batch_size)*dW[t][l]*momentum).eval();
			prev_dB[t][l]=((prev_dB[t][l] + bprev_dB[t][l])*(1-momentum) + learning_rate/double(mini_batch_size)*dB[t][l]*momentum).eval();
			bprev_dW[t][l] =( (prev_dW[t][l]-tempW)*(1.-db) + bprev_dW[t][l]*db).eval();
			bprev_dB[t][l] =( (prev_dB[t][l]-tempB)*(1.-db) + bprev_dB[t][l]*db).eval();
			W[l] -= prev_dW[t][l];
			B[l] -= prev_dB[t][l];
		}
		W[l] -= regularization * W[l];
	}

	// for (int l = 0; l < numLayers - 1; ++l) {
	// 	for (int t = 0; t < threads; ++t) {
	// 		prev_dW[t][l]=learning_rate/double(mini_batch_size)*dW[t][l];
	// 		prev_dB[t][l]=learning_rate/double(mini_batch_size)*dB[t][l];
	// 		W[l] -= prev_dW[t][l];
	// 		B[l] -= prev_dB[t][l];
	// 	}
	// 	W[l] -= regularization * W[l];
	// }

}

void Network::backprop(MatrixXd& X, VectorXi& Y, vector<MatrixXd>& dW, vector<VectorXd>& dB) {
	int l = 0;
	vector<MatrixXd> A = {X};
	while (l < numLayers - 1) {
		A.push_back(neuron((W[l] * A[l]).colwise() + B[l]));
		++l;
	}

	// quadratic-sigmoid
	// MatrixXd d_l = quadratic_cost_derivative(A[l], Y).array() * sigmoid_derivative(A[l]).array();

	// crossentr-sigmoid
	MatrixXd d_l = cross_entropy_cost_derivative(A[l], Y);

	// loglikelihood-softmax
	// MatrixXd d_l = loglikelihood_cost_derivative(A[l], Y);

	// quadratic-relu
	// MatrixXd d_l = quadratic_cost_derivative(A[l], Y).array() * relu_derivative(A[l]).array();

	dW[l-1] = d_l * A[l-1].transpose();
	dB[l-1] = d_l.col(0);
	for (int i = 1; i < d_l.cols(); ++i)
		dB[l-1] += d_l.col(i);

	--l;

	while (l >= 1) {
		// sigmoid neurons
		d_l = (W[l].transpose() * d_l).eval().array() * sigmoid_derivative(A[l]).array();

		// // softmax neurons
		// MatrixXd newd_l = MatrixXd::Zero(A[l].rows(), d_l.cols());
		// for (int e = 0; e < A[l].cols(); ++e)
		// 	newd_l.col(e) = ((W[l]*softmax_derivative(A[l].col(e))).transpose() * d_l.col(e)).eval();
		// d_l=newd_l;

		// relu neurons
		// d_l = (W[l].transpose() * d_l).eval().array() * relu_derivative(A[l]).array();

		dW[l-1] = d_l * A[l-1].transpose();
		dB[l-1] = d_l.col(0);
		for (int i = 1; i < d_l.cols(); ++i)
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

MatrixXd Network::quadratic_cost_derivative(MatrixXd A_L, const VectorXi Y) {
	for (int i = 0; i < A_L.cols(); ++i)
		A_L.col(i)(Y(i)) -= 1.;
	return A_L;
}

MatrixXd Network::cross_entropy_cost_derivative(MatrixXd A_L, const VectorXi Y) {
	for (int i = 0; i < A_L.cols(); ++i)
		A_L.col(i)(Y(i)) -= 1.;
	return A_L;
}

MatrixXd Network::loglikelihood_cost_derivative(MatrixXd A_L, const VectorXi Y) {
	for (int i = 0; i < A_L.cols(); ++i)
		A_L.col(i)(Y(i)) -= 1.;
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
MatrixXd Network::sigmoid_derivative(const MatrixXd& activations) {
	return (activations.array() * (1.0 - activations.array()));
}

MatrixXd Network::softmax(MatrixXd&& z) {
	// double D = -z.maxCoeff();
	// z = (z.array()+D).exp();
	z = z.array().exp();
	for (int i = 0; i < z.cols(); ++i)
		z.col(i)/=z.col(i).sum();
	return z;
}

// computes the jacobian of the softmax function
MatrixXd Network::softmax_derivative(VectorXd&& activations) {
	return (activations.asDiagonal()).toDenseMatrix()-(activations*activations.transpose());
}

MatrixXd Network::relu(const MatrixXd& z) {
	return z.array().max(ArrayXXd::Zero(z.rows(),z.cols()));
}

MatrixXd Network::relu_derivative(const MatrixXd& activations) {
	return activations.array()/(activations.array()+0.0000001);
}
