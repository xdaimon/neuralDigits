#include "network.h"
#include "mnist_loader.h"
using namespace NeuralNetwork;

class Test {
	public:
		static bool SGD();
};

bool Test::SGD() {
	// Train from a single data point with only one value which is non zero
	Data train_data;
	train_data.examples = MatrixXd::Zero(3, 3);
	train_data.examples << 1., 0., 0.,
	                       0., 1., 0.,
	                       0., 0., 1.;
	train_data.labels = VectorXi::Zero(3);
	train_data.labels << 0, 1, 2;
	// Test using the same data point
	Data test_data;
	test_data.examples = MatrixXd::Zero(3, 3);
	test_data.examples << 1., 0., 0.,
	                      0., 1., 0.,
	                      0., 0., 1.;
	test_data.labels = VectorXi::Zero(3);
	test_data.labels << 0, 1, 2;

	VectorXi layer_sizes = VectorXi::Zero(3);
	layer_sizes << 3, 4, 3;
	Network net(layer_sizes);

	const int num_epochs = 10;
	const int mini_batch_size = 1;
	const double learning_rate = 3.;
	double accuracy = net.SGD(train_data, test_data, num_epochs, mini_batch_size, learning_rate);
	cout << "Accuracy: " << accuracy << endl;
	return accuracy > 95.;
}

int main()
{
	if (!Test::SGD())
		cout << "Accuracy Test Failed" << endl;
	else
		cout << "Accuracy Test Passed" << endl;

	// Test code
	// highlevel?
	//     load known image into memory
	//     make train example from it
	//     test net can learn this
	// lowlevel?
	//     conceive of some simple train example
	//     test expected (analytically obtained?) output for backprop
	//
}
