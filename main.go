package main

import (
	"math/rand"
	"time"

	"neuralnetworksanddeeplearning.com/chap1/mnist"
	"neuralnetworksanddeeplearning.com/chap1/network"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	data, err := mnist.Load("train", "./resources")
	if err != nil {
		panic(err)
	}

	n := network.New([]int{784, 30, 10})
	n.SGD(data[:50000], 30, 10, 3.0, data[50000:])
}
