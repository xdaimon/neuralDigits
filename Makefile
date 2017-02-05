CC=g++
# CC=clang++
CFLAGS= -O3 -std=c++11 -I.
# CFLAGS= -O0 -g -std=c++11 -I.

# $(CC) $(CFLAGS) main.cpp network.cpp mnist_loader.cpp -o main -lz -lpng12
all:
	$(CC) $(CFLAGS) main.cpp network.cpp mnist_loader.cpp -o main -lz

test:
	$(CC) $(CFLAGS) main_test.cpp network.cpp mnist_loader.cpp -o main -lz

clean:
	rm main
