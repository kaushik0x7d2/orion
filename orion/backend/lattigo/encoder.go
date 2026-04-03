package main

import (
	"C"
	"sync"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
	"github.com/baahl-nyu/lattigo/v6/schemes/ckks"
)

var encMu sync.Mutex

//export NewEncoder
func NewEncoder() {
	encMu.Lock()
	defer encMu.Unlock()

	scheme.Encoder = ckks.NewEncoder(*scheme.Params)
}

//export Encode
func Encode(
	valuesPtr *C.float,
	lenValues C.int,
	level C.int,
	scale C.ulonglong,
) C.int {
	encMu.Lock()
	defer encMu.Unlock()

	values := CArrayToSlice(valuesPtr, lenValues, convertCFloatToFloat)
	plaintext := ckks.NewPlaintext(*scheme.Params, int(level))
	plaintext.Scale = rlwe.NewScale(uint64(scale))

	scheme.Encoder.Encode(values, plaintext)

	idx := PushPlaintext(plaintext)
	return C.int(idx)
}

//export Decode
func Decode(
	plaintextID C.int,
) (*C.float, C.ulong) {
	encMu.Lock()
	defer encMu.Unlock()

	plaintext := RetrievePlaintext(int(plaintextID))
	result := make([]float64, scheme.Params.MaxSlots())
	scheme.Encoder.Decode(plaintext, result)

	arrPtr, length := SliceToCArray(result, convertFloatToCFloat)
	return arrPtr, length
}
