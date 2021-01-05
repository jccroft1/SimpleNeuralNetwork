package mnist

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"neuralnetworksanddeeplearning.com/chap1/network"
)

const numLabels = 10
const pixelRange = 255

const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	// Width of the input tensor / picture
	Width = 28
	// Height of the input tensor / picture
	Height = 28
)

// RawImage holds the pixel intensities of an image.
// 255 is foreground (black), 0 is background (white).
type RawImage []byte

// Label is a digit label in 0 to 9
type Label uint8

// Load loads the mnist data into two tensors
//
// typ can be "train", "test"
//
// loc represents where the mnist files are held
func Load(typ, loc string) ([]network.Data, error) {
	const (
		trainLabel = "train-labels-idx1-ubyte"
		trainData  = "train-images-idx3-ubyte"
		testLabel  = "t10k-labels-idx1-ubyte"
		testData   = "t10k-images-idx3-ubyte"
	)

	var labelFile, dataFile string
	switch typ {
	case "train", "dev":
		labelFile = filepath.Join(loc, trainLabel)
		dataFile = filepath.Join(loc, trainData)
	case "test":
		labelFile = filepath.Join(loc, testLabel)
		dataFile = filepath.Join(loc, testData)
	}

	var labelData []Label
	var imageData []RawImage
	var err error

	if labelData, err = readLabelFile(os.Open(labelFile)); err != nil {
		return []network.Data{}, fmt.Errorf("Unable to read Labels %v", err)
	}

	if imageData, err = readImageFile(os.Open(dataFile)); err != nil {
		return []network.Data{}, fmt.Errorf("Unable to read Labels %v", err)
	}

	return prepareData(imageData, labelData)
}

func prepareData(imageData []RawImage, labelData []Label) ([]network.Data, error) {
	if len(imageData) != len(labelData) {
		return []network.Data{}, errors.New("image and label data length does not match ")
	}

	pixelCount := len(imageData[0])

	data := make([]network.Data, len(imageData))
	for i := range imageData {
		// Input
		x := make([]float64, pixelCount)
		for j, pixel := range imageData[i] {
			x[j] = pixelWeight(pixel)
		}

		// Output
		y := make([]float64, 10)
		y[labelData[i]] = 1.0

		data[i] = network.Data{
			Input:  x,
			Output: y,
		}
	}
	return data, nil
}

func pixelWeight(px byte) float64 {
	retVal := float64(px)/pixelRange*0.9 + 0.1
	if retVal == 1.0 {
		return 0.999
	}
	return retVal
}

func reversePixelWeight(px float64) byte {
	return byte((pixelRange*px - pixelRange) / 0.9)
}

func readLabelFile(r io.Reader, e error) (labels []Label, err error) {
	if e != nil {
		return nil, e
	}

	var (
		magic int32
		n     int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {
		return nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels = make([]Label, n)
	for i := 0; i < int(n); i++ {
		var l Label
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = l
	}
	return labels, nil
}

func readImageFile(r io.Reader, e error) (imgs []RawImage, err error) {
	if e != nil {
		return nil, e
	}

	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {
		return nil, err /*os.ErrInvalid*/
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return nil, err
	}
	imgs = make([]RawImage, n)
	m := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgs[i] = make(RawImage, m)
		m_, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return nil, err
		}
		if m_ != int(m) {
			return nil, os.ErrInvalid
		}
	}
	return imgs, nil
}
