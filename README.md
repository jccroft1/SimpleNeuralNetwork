# Simple Neural Network 

A Neural Network written using only the Golang standard library.

## Features 

* Learns by Stochastic Gradient Descent 
* L2 Regularisation 
* Cross Entropy or Quadratic Cost Function 
* Sigmoid neuron activation 

## Installation 

1. Clone this repo 
2. Download the MNIST data into the assets folder 

## Usage 

### Training 

Run `go run main.go -test` from cmd/train directory to load the dataset and begin training a network from scratch. 

The test flag will evaluate the network against a separate set of data. 

## To Do 

* Load and save network parameters 
* Standalone 'deployable' network 

## Citations 

This repo is heavily inspired by the below online book by Michael Nielsen. 

Michael A. Nielsen, "Neural Networks and
Deep Learning", Determination Press, 2015
http://neuralnetworksanddeeplearning.com/index.html

