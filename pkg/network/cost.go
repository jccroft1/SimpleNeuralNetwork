package network

type Cost func(z, a, y float64) float64

var CrossEntropyCost Cost = func(z, a, y float64) float64 {
	return a - y
}

var QuadraticCost Cost = func(z, a, y float64) float64 {
	return (a - y) * sigmoidPrime(z)
}
