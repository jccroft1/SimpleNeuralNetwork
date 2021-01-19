package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"

	"github.com/jccroft1/SimpleNeuralNetwork/mnist"
	"github.com/jccroft1/SimpleNeuralNetwork/network"
)

var (
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")
	test       = flag.Bool("test", true, "enable testing of data with each epoch")
)

func main() {
	flag.Parse()

	rand.Seed(time.Now().UnixNano())

	data, err := mnist.Load("train", "../../resources")
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

	n := network.New([]int{784, 30, 10}, network.CrossEntropyCost)

	trainingData := data[:50000]
	testData := []network.Data{}
	if *test {
		testData = data[50000:]
	}

	n.SGD(trainingData, testData, 30, 10, 0.5, 5.0)
}
