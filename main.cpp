#include "network.h"
#include "mnist_loader.h"
using namespace NeuralNetwork;

int main() {
	const int dimensions = 28 * 28;
	const int num_classes = 10;
	VectorXi layer_sizes(1 + 2);
	layer_sizes << dimensions, 30, num_classes;

	Data train_data;
	Data validation_data;
	Data test_data;
	load_data(train_data, test_data, validation_data);

	Data& use_set = validation_data;

	// Benches for 60 epochs
	// amd 8350 bench
	// batch size 80
	// thread 8
	// 35 avg time
	// 2 mew
	// .05 lambda
	// 93.9 accuracy

	// batch size 400
	// threads 8
	// 16 avg time
	// 2 mew
	// .05 lambda
	// 92.36 accuracy

	// batch size 1000
	// threads 8
	// 19 avg time
	// 2 mew
	// .05 lambda
	// 90.65 accuracy

	// batch size 200
	// threads 8
	// 20 avg time
	// 2 mew
	// .05 lambda
	// 93.38 accuracy

	// batch size 400
	// threads 16
	// 20 avg time
	// 2 mew
	// .05 lambda
	// 93.38 accuracy

	// batch size 200
	// threads 4
	// 16.5 avg time
	// 2 mew
	// .05 lambda
	// 93.38 accuracy

	const int num_epochs = 60;
	const int mini_batch_size = 400;
	const int threads = 8;
	const int num_examples = use_set.examples.cols();
	const int num_batches = num_examples/mini_batch_size;
	cout << "examples not trained on : " << num_examples - num_batches * mini_batch_size << endl;

	const double learning_rate = 2.;
	const double regularization = 0.05 * learning_rate/double(num_examples);

	cout << "num_examples: " << num_examples << endl;
	cout << "num_batches: " << num_batches << endl;

	cout << "mini_size % threads" << mini_batch_size%threads << endl;
	cout << "num_examples % mini_size" << num_examples%mini_batch_size << endl;

	if (mini_batch_size%threads != 0) return 0;
	if (num_examples%mini_batch_size != 0) return 0;

	Network net(layer_sizes);
	double accuracy = net.SGD(use_set, test_data, num_epochs, mini_batch_size, learning_rate, regularization, threads);
	cout << accuracy << endl;
}
