package main

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	"github.com/jccroft1/SimpleNeuralNetwork/pkg/mnist"
	"github.com/jccroft1/SimpleNeuralNetwork/pkg/network"
)

func main() {
	data, err := mnist.Load("test", "../../assets")
	if err != nil {
		panic(err)
	}

	fmt.Printf("Dataset loaded (%v items)\n", len(data))

	networkReader, err := os.Open("../../configs/network")
	// networkFile, err := ioutil.ReadFile("../../configs/network")
	if err != nil {
		panic(err)
	}

	networkDecoder := gob.NewDecoder(networkReader)

	// Decode (receive) and print the values.
	var n network.Network
	err = networkDecoder.Decode(&n)
	if err != nil {
		log.Fatal("decode error 1:", err)
	}

	correct := 0
	for _, x := range data {
		expected := network.MaxIndex(x.Output)

		y, _ := n.FeedForward(x.Input)
		actual := network.MaxIndex(y)

		if expected == actual {
			correct++
		}
	}

	fmt.Printf("Results: %v/%v\n", correct, len(data))
}
