package network

import (
	"fmt"
	"math"
	"math/rand"
)

type Data struct {
	Input  []float64
	Output []float64
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func sigmoidPrime(z float64) float64 {
	s := sigmoid(z)
	return s * (1 - s)
}

// New creates a Network struct
// Assumes first layer is input
// Inits biases and weights to normal distribution
func New(sizes []int) Network {
	n := Network{
		Layers:  len(sizes),
		Sizes:   sizes,
		Biases:  make([][]float64, len(sizes)-1),
		Weights: make([][][]float64, len(sizes)-1),
	}

	for i := 0; i < len(n.Biases); i++ {
		n.Biases[i] = make([]float64, n.Sizes[i+1])

		for j := 0; j < len(n.Biases[i]); j++ {
			n.Biases[i][j] = rand.NormFloat64()
		}
	}

	for i := 0; i < len(n.Weights); i++ {
		n.Weights[i] = make([][]float64, n.Sizes[i+1])

		for j := 0; j < len(n.Weights[i]); j++ {
			n.Weights[i][j] = make([]float64, n.Sizes[i])

			for k := 0; k < len(n.Weights[i][j]); k++ {
				n.Weights[i][j][k] = rand.NormFloat64()
			}
		}
	}

	return n
}

type Network struct {
	Layers  int
	Sizes   []int
	Biases  [][]float64
	Weights [][][]float64
}

func (n Network) FeedForward(a []float64) ([]float64, error) {
	// Check input vector matches input layer
	if len(a) != n.Sizes[0] {
		return []float64{}, fmt.Errorf("Input vector (len %v), does not match input layer (len %v)", len(a), n.Sizes[0])
	}

	return n.feedLayer(a, 1), nil
}

func (n Network) feedLayer(in []float64, layer int) []float64 {
	output := make([]float64, n.Sizes[layer])

	for i := 0; i < len(output); i++ {
		z := n.Biases[layer-1][i]

		for j := 0; j < len(in); j++ {
			z += n.Weights[layer-1][i][j] * in[j]
		}

		output[i] = sigmoid(z)
	}

	// Reached last layer
	if layer == n.Layers-1 {
		return output
	}

	return n.feedLayer(output, layer+1)
}

func (n *Network) SGD(train []Data, epochs, batchSize int, learningRate float64, test []Data) {

	for i := 0; i < epochs; i++ {
		rand.Shuffle(len(train), func(i, j int) { train[i], train[j] = train[j], train[i] })

		for j := 0; j < len(train); j += batchSize {
			batch := train[j : j+batchSize]
			n.ProcessBatch(batch, learningRate)
		}

		fmt.Printf("Epoch %v complete\n", i)
		if len(test) > 0 {
			// evaluate on test data
			correct := n.Evaluate(test)
			fmt.Printf("Test: %v/%v\n", correct, len(test))
		}
	}
}

func (n *Network) ProcessBatch(train []Data, eta float64) {
	biasNabla := make([][]float64, len(n.Biases))
	for i := 0; i < len(biasNabla); i++ {
		biasNabla[i] = make([]float64, len(n.Biases[i]))
		for j := 0; j < len(biasNabla[i]); j++ {
			biasNabla[i][j] = 0
		}
	}

	weightsNabla := make([][][]float64, len(n.Weights))
	for i := 0; i < len(weightsNabla); i++ {
		weightsNabla[i] = make([][]float64, len(n.Weights[i]))
		for j := 0; j < len(weightsNabla[i]); j++ {
			weightsNabla[i][j] = make([]float64, len(n.Weights[i][j]))
			for k := 0; k < len(weightsNabla[i][j]); k++ {
				weightsNabla[i][j][k] = 0
			}
		}
	}

	for _, x := range train {
		biasNablaDelta, weightsNablaDelta := n.backprop(x)

		for i := range biasNabla {
			for j := range biasNabla[i] {
				biasNabla[i][j] += biasNablaDelta[i][j]
			}
		}

		for i := range weightsNabla {
			for j := range weightsNabla[i] {
				for k := range weightsNabla[i][j] {
					weightsNabla[i][j][k] += weightsNablaDelta[i][j][k]
				}
			}
		}
	}

	c := eta / float64(len(train))
	for i := range n.Biases {
		for j := range n.Biases[i] {
			n.Biases[i][j] -= biasNabla[i][j] * c
		}
	}

	for i := range n.Weights {
		for j := range n.Weights[i] {
			for k := range n.Weights[i][j] {
				n.Weights[i][j][k] -= weightsNabla[i][j][k] * c
			}
		}
	}
}

func (n *Network) backprop(data Data) ([][]float64, [][][]float64) {
	// init nablas
	biasNabla := make([][]float64, len(n.Biases))
	for i := 0; i < len(biasNabla); i++ {
		biasNabla[i] = make([]float64, len(n.Biases[i]))
		for j := 0; j < len(biasNabla[i]); j++ {
			biasNabla[i][j] = 0
		}
	}

	weightsNabla := make([][][]float64, len(n.Weights))
	for i := 0; i < len(weightsNabla); i++ {
		weightsNabla[i] = make([][]float64, len(n.Weights[i]))
		for j := 0; j < len(weightsNabla[i]); j++ {
			weightsNabla[i][j] = make([]float64, len(n.Weights[i][j]))
			for k := 0; k < len(weightsNabla[i][j]); k++ {
				weightsNabla[i][j][k] = 0
			}
		}
	}

	activation := data.Input
	activations := [][]float64{data.Input}
	zs := make([][]float64, n.Layers-1)

	// feedforward
	// for each layer
	for layer := 1; layer < n.Layers; layer++ {
		output := make([]float64, n.Sizes[layer])
		zs[layer-1] = make([]float64, n.Sizes[layer])

		// for each neuron
		for i := 0; i < n.Sizes[layer]; i++ {
			z := n.Biases[layer-1][i]

			// for each input neuron
			for j := 0; j < len(activation); j++ {
				z += n.Weights[layer-1][i][j] * activation[j]
			}

			zs[layer-1][i] = z

			output[i] = sigmoid(z)
		}

		activation = output
		activations = append(activations, activation)
	}

	//backward pass
	// output layer (L)
	L := n.Layers - 1
	delta := make([]float64, len(activations[L]))
	for i := 0; i < len(activations[L]); i++ {
		delta[i] = (activations[L][i] - data.Output[i]) * sigmoidPrime(zs[L-1][i])
	}

	biasNabla[L-1] = delta
	for i, d := range delta {
		for j, a := range activations[L-1] {
			weightsNabla[L-1][i][j] = d * a
		}
	}

	// working backwards (L-1, L-2,...)
	for l := L - 1; l > 0; l-- {
		z := zs[l-1]

		// TODO: Might not need this newDelta, just being careful to not override existing delta value
		newDelta := make([]float64, n.Sizes[l])
		for i := 0; i < len(newDelta); i++ {
			newDelta[i] = 0
			for k := 0; k < n.Sizes[l+1]; k++ {
				newDelta[i] += n.Weights[l][k][i] * delta[k]
			}
			newDelta[i] = newDelta[i] * sigmoidPrime(z[i])
		}
		delta = newDelta

		biasNabla[l-1] = delta
		for i, d := range delta {
			for j, a := range activations[l-1] {
				weightsNabla[l-1][i][j] = d * a
			}
		}
	}

	return biasNabla, weightsNabla
}

func (n *Network) Evaluate(test []Data) int {
	correct := 0
	for _, x := range test {
		expected := MaxIndex(x.Output)

		y, _ := n.FeedForward(x.Input)
		actual := MaxIndex(y)

		if expected == actual {
			correct++
		}
	}

	return correct
}

func MaxIndex(in []float64) int {
	max := -math.MaxFloat64
	index := 0
	for i, x := range in {
		if x > max {
			index = i
			max = x
		}
	}
	return index
}
