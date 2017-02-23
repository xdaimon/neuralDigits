#include "network.h"
#include "mnist_loader.h"
using namespace NeuralNetwork;

int main()
{
	const int num_epochs = 30;
	const int mini_batch_size = 10;
	const double learning_rate = 3.;

	const int dimensions = 28*28;
	const int num_classes = 10;
	VectorXi layer_sizes(1+2);
	layer_sizes << dimensions, 30, num_classes;

	Data train_data;
	Data validation_data;
	Data test_data;
	load_data(train_data, test_data, validation_data);

	Network net(layer_sizes);
	double accuracy = net.SGD(train_data, test_data, num_epochs, mini_batch_size, learning_rate);
	cout << "Accuracy: " << accuracy << endl;
}
