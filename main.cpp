#include "network.h"
#include "mnist_loader.h"
using namespace NeuralNetwork;

int main() {
	const int num_epochs = 60*8;
	const int mini_batch_size = 80;
	const double learning_rate = 1.7;

	const int dimensions = 28 * 28;
	const int num_classes = 10;
	VectorXi layer_sizes(1 + 2);
	layer_sizes << dimensions, 30, num_classes;

	Data train_data;
	Data validation_data;
	Data test_data;
	load_data(train_data, test_data, validation_data);

	Network net(layer_sizes);
	net.SGD(train_data, test_data, num_epochs, mini_batch_size, learning_rate);
}
