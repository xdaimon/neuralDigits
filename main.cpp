#include "network.h"
#include "mnist_loader.h"
using namespace NeuralNetwork;
#include <limits.h>
#include <iomanip>

int main() {

	// This app computes the same computation graph as presented in the tensorflow mnist begginer's tutorial.
	// Both the tutorial and this app use the same sections of data from the mnist data set.
	// The tensorflow tutorial was modified to use a sigmoid neuron with a cross entropy cost, to have a 400 example minibatch size (with shuffe=True),
	// to compute 125 minibatches, and to compute 60 epochs of training over the train set.
	// tensorflow-gpu: 91.78 after roughly 27 seconds
	// this app: 91.78 after roughly 32 seconds
	// (734 in, 10 out, no hidden layers, sigmoid non-linearity with cross entropy cost function)
	// (1.0 learning rate, 60 epochs, 400 batch size)

	// AMD 8350 benchmark
	// batch size 400
	// threads 16
	// 20 avg time
	// 2. momentum paremeter
	// .05 regulariztion parameter
	// 93.38 accuracy
	//
	// LAPTOP I5-6200u 2.3GHZ
	// batch size 400
	// threads 8
	// 14.7 avg time
	// 2. momenutm parameter
	// .05 regularization parameter
	// 92.08 accuracy

	Data train_data;
	Data validation_data;
	Data test_data;
	load_data(train_data, test_data, validation_data);

	Data& use_set = train_data;

	const int num_epochs = 60;
	const int mini_batch_size = 400;
	const int threads = 8;
	const int num_examples = use_set.examples.cols();
	const int num_batches = num_examples/mini_batch_size;

	const double learning_rate = 1.;
	const double regularization = 1. * learning_rate/double(num_examples);
	const double momentum = .3;

	cout << "num_examples: " << num_examples << endl;
	cout << "num_batches: " << num_batches << endl;

	cout << "mini_size % threads" << mini_batch_size%threads << endl;
	cout << "num_examples % mini_size" << num_examples%mini_batch_size << endl;

	if (mini_batch_size%threads != 0) return 0;
	if (num_examples%mini_batch_size != 0) return 0;
	VectorXi layerSizes(1+1);
	layerSizes << 28*28,10;

	Network net(layerSizes);
	double accuracy = net.SGD(use_set, test_data, num_epochs, mini_batch_size, learning_rate, regularization, momentum, threads);
	cout << std::setprecision(std::numeric_limits< double >::max_digits10) << accuracy << endl;
}
