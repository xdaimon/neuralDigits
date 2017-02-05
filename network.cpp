#include "network.h"
using namespace NeuralNetwork;

// This code is almost an exact translation of the python code
// The difference is with random number generation and thus with weight / bias initialization and array shuffling
// 
// I've had issues while working with the eigen library due to aliasing.
// Aliasing is when an eigen variable x appears on both the right and left hand side of an assignment.
// More information on aliasing can be found on the eigen website.
// I would prefer not to have this problem, however, eigen is nice to work with and it seems fast.
//
// I claim that it is easier to learn to work with eigen's one quirk (aliasing) than it is to attempt to learn
// all the non obvious syntax used in python

void Network::log_parameters()
{
	for (int i = 0+1; i < number_of_layers-1; ++i)
		cout << "layer " << i+1 << " biases : " << endl << biases[i] << endl << endl;
	for (int i = 0+1; i < number_of_layers-1; ++i)
		cout << "layer " << i+1 << " weights : " << endl << weights[i] << endl << endl;
}

Network::Network(const VectorXi & layer_sizes)
{
	number_of_layers = layer_sizes.size();
	for (int i = 1; i < number_of_layers; ++i)
	{
		weights.push_back(2.*MatrixXd::Random(layer_sizes(i), layer_sizes(i-1)));
		biases.push_back(2.*MatrixXd::Random(layer_sizes(i), 1));
		// weights.push_back(MatrixXd::Zero(layer_sizes(i),layer_sizes(i-1)));
		// biases.push_back(VectorXd::Zero(layer_sizes(i)));
	}

	// Load from python data
	// for (int i = 0; i < weights[0].rows(); ++i)
	// {
	// 	for (int j = 0; j < weights[0].cols(); ++j)
	// 	{
	// 		weights[0](i,j) = w1[i * weights[0].cols() + j];
	// 	}
	// }
	// for (int i = 0; i < weights[1].rows(); ++i)
	// {
	// 	for (int j = 0; j < weights[1].cols(); ++j)
	// 	{
	// 		weights[1](i,j) = w2[i * weights[1].cols() + j];
	// 	}
	// }
	// for (int i = 0; i < biases[0].rows(); ++i)
	// {
	// 	for (int j = 0; j < biases[0].cols(); ++j)
	// 	{
	// 		biases[0](i,j) = b1[i * biases[0].cols() + j];
	// 	}
	// }
	// for (int i = 0; i < biases[1].rows(); ++i)
	// {
	// 	for (int j = 0; j < biases[1].cols(); ++j)
	// 	{
	// 		biases[1](i,j) = b2[i * biases[1].cols() + j];
	// 	}
	// }
}

VectorXd Network::feed_forward(VectorXd&& x)
{
	for (int i = 0; i < number_of_layers-1; ++i)
		x = sigmoid((weights[i] * x).eval() + biases[i]);
	return x;
}

double Network::SGD(const Data & train_data, const Data & test_data,
                    int epochs, int mini_batch_size, double learning_rate)
{
	const int dimension = train_data.examples.rows();
	const int num_examples = train_data.examples.cols();
	const int num_batches = num_examples/mini_batch_size;
	Data mini_batch;
	mini_batch.examples = MatrixXd::Zero(dimension, mini_batch_size);
	mini_batch.labels = VectorXi::Zero(mini_batch_size);
	vector<int> random_indices(num_examples);
	std::iota(random_indices.begin(), random_indices.end(), 0); // fill random_indices with increasing numbers starting from 0
	for (int j = 0; j < epochs; ++j)
	{
		std::random_shuffle(random_indices.begin(), random_indices.end());
		for (int i = 0; i < num_batches; ++i)
		{
			// Make a minibatch
			for (int k = 0; k < mini_batch_size; ++k)
			{
				mini_batch.examples.col(k) = train_data.examples.col(random_indices[k+mini_batch_size*i]);
				mini_batch.labels(k) = train_data.labels(random_indices[k+mini_batch_size*i]);
			}
			train_for_mini_batch(mini_batch, learning_rate);
		}
		cout << "Epoch " << j << " : " << evaluate(test_data) << "% accuracy" << endl << endl;
	}
	return evaluate(test_data);
}

void Network::train_for_mini_batch(const Data & mini_batch, double learning_rate)
{
	vector<MatrixXd> nabla_w(number_of_layers-1);
	vector<VectorXd> nabla_b(number_of_layers-1);
	for (int i = 0; i < number_of_layers-1; ++i)
	{
		nabla_w[i] = MatrixXd::Zero(weights[i].rows(), weights[i].cols());
		nabla_b[i] = VectorXd::Zero(biases[i].rows());
	}
	const int mini_batch_size = mini_batch.labels.rows();
	for (int i = 0; i < mini_batch_size; ++i)
		backprop(mini_batch.examples.col(i), mini_batch.labels(i), nabla_w, nabla_b);
	for (int i = 0; i < number_of_layers-1; ++i)
	{
		weights[i] -= (learning_rate*(nabla_w[i]/double(mini_batch_size)));
		biases[i] -= (learning_rate*(nabla_b[i]/double(mini_batch_size)));
	}
}

void Network::backprop(const VectorXd & x, int y, vector<MatrixXd>& nabla_w, vector<VectorXd>& nabla_b)
{
	vector<VectorXd> activations = {x};
	for (int i = 1; i < number_of_layers; ++i)
		   activations.push_back(sigmoid(weights[i-1]*activations[i-1] + biases[i-1]));
	VectorXd delta = cost_derivative(activations[activations.size()-1], y).array()*activations[activations.size()-1].array()*(1.0-activations[activations.size()-1].array());
	nabla_w[nabla_w.size() -1] += delta * activations[activations.size()-2].transpose();
	nabla_b[nabla_b.size() -1] += delta;
	for (int l = 2; l < number_of_layers; ++l)
	{
		delta = (weights[weights.size() -l+1].transpose() * delta).eval().array()*activations[activations.size()-l].array()*(1.0-activations[activations.size()-l].array());
		nabla_w[nabla_w.size() -l] += delta * activations[activations.size() -l-1].transpose();
		nabla_b[nabla_b.size() -l] += delta;
	}
	// VectorXd activation = x;
	// vector<VectorXd> activations = {x};
	// VectorXd z;
	// vector<VectorXd> zs;
	// for (int i = 0; i < number_of_layers-1; ++i)
	// {
	// 	z = (weights[i]*activation + biases[i]).eval();
	// 	zs.push_back(z);
	// 	activation = sigmoid(z);
	// 	activations.push_back(activation);
	// }
	// VectorXd delta = cost_derivative(activations[activations.size()-1], y).array() * Dsigmoid(zs[zs.size()-1]).array();
	// // VectorXd delta = cost_derivative(activations[activations.size()-1], y).array() * activations[activations.size()-1].array()*(1.0-activations[activations.size()-1].array());
	// nabla_w[nabla_w.size() -1] += delta * activations[activations.size()-2].transpose();
	// nabla_b[nabla_b.size() -1] += delta;
	// VectorXd sp;
	// for (int l = 2; l < number_of_layers; ++l)
	// {
	// 	// z = zs[zs.size()-l];
	// 	// sp = Dsigmoid(z);
	// 	sp = activations[activations.size()-l].array()*(1.0-activations[activations.size()-l].array());
	// 	delta = ((weights[weights.size() -l+1].transpose() * delta).array()*sp.array()).eval();
	// 	nabla_w[nabla_w.size() -l] += delta * activations[activations.size() -l-1].transpose();
	// 	nabla_b[nabla_b.size() -l] += delta;
	// }
	// // for (int i = 0; i < number_of_layers - 1; ++i)
	// // {
	// // 	cout << "layer " << i + 1 << " grad b" << endl << nabla_b[i] << endl << endl;
	// // 	cout << "layer " << i + 1 << " grad w" << endl << nabla_w[i] << endl << endl;
	// // }
}

double Network::evaluate(const Data & test_data)
{
	const int n = test_data.examples.cols();
	vector<int> test_results(n);
	int num_correctly_labeled = 0;
	// This simplification makes the code slower.
	// if (argmax(feed_forward(test_data.examples.col(i))) == test_data.labels(i))
	// 	num_correctly_labeled++;
	for (int i = 0; i < n; ++i)
		test_results[i] = argmax(feed_forward(test_data.examples.col(i)));
	for (int i = 0 ; i < n; ++i)
		if (test_results[i] == test_data.labels(i))
			num_correctly_labeled++;
	return num_correctly_labeled/double(n)*100.;
}

VectorXd Network::cost_derivative(VectorXd output_activations, int y)
{
	// When y is a vector of zeros except for a 1 at the index of the true label
	// return output_activations - y;

	output_activations(y) -= 1.;
	return output_activations;
}

int Network::argmax(const VectorXd & v)
{
	int x;
	(void)v.maxCoeff(&x);
	return x;
}

VectorXd Network::sigmoid(const VectorXd & z)
{
	return 1./(1.+(-z.array()).exp());
}

VectorXd Network::Dsigmoid(const VectorXd & z)
{
	const VectorXd sz = Network::sigmoid(z);
	return sz.array()*(1.0-sz.array());
}

//--- Code for converting module to work with matricies of train examples instead of just single vectors of examples
//--- search eigen partial reduction
// VectorXd Network::feed_forward(MatrixXd&& X)
// {
// 	for (int i = 0; i < number_of_layers-1; ++i)
// 		X = sigmoid((weights[i] * X).colwise() + biases[i]);
// 	return X;
// }

// void Network::backprop(MatrixXd x, int y, vector<MatrixXd>& nabla_w, vector<VectorXd>& nabla_b)
// Ooo
// //add v to each column of m
// mat.colwise() += v;
