package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"

	"github.com/jccroft1/SimpleNeuralNetwork/pkg/mnist"
	"github.com/jccroft1/SimpleNeuralNetwork/pkg/network"
)

var (
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")
	test       = flag.Bool("test", true, "enable testing of data with each epoch")
)

func main() {
	flag.Parse()

	rand.Seed(time.Now().UnixNano())

	data, err := mnist.Load("train", "../../assets")
	if err != nil {
		panic(err)
	}

	fmt.Printf("Dataset loaded (%v items)\n", len(data))

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal("could not create CPU profile: ", err)
		}
		defer f.Close() // error handling omitted for example
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal("could not start CPU profile: ", err)
		}
		defer pprof.StopCPUProfile()
	}

	n := network.New([]int{784, 30, 10})

	trainingData := data[:50000]
	testData := []network.Data{}
	if *test {
		testData = data[50000:]
	}

	// train
	n.SGD(trainingData, testData, network.TrainingParameters{
		Epochs:        30,
		BatchSize:     10,
		LearningRate:  0.5,
		Lambda:        5.0,
		Cost:          network.CrossEntropyCost,
		ImprovementIn: 10,
	})

	// save
	networkWriter, err := os.Create("../../configs/network")
	if err != nil {
		fmt.Println("failed to writer network")
		return
	}

	networkEncoder := gob.NewEncoder(networkWriter)

	err = networkEncoder.Encode(n)
	if err != nil {
		fmt.Println("failed to writer network")
		return
	}
}
