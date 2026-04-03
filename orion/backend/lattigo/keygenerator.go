package main

import (
	"C"
	"sync"
	"unsafe"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)

var keyMu sync.Mutex

//export NewKeyGenerator
func NewKeyGenerator() {
	keyMu.Lock()
	defer keyMu.Unlock()

	scheme.KeyGen = rlwe.NewKeyGenerator(scheme.Params)
}

//export GenerateSecretKey
func GenerateSecretKey() {
	keyMu.Lock()
	defer keyMu.Unlock()

	scheme.SecretKey = scheme.KeyGen.GenSecretKeyNew()
}

//export GeneratePublicKey
func GeneratePublicKey() {
	keyMu.Lock()
	defer keyMu.Unlock()

	scheme.PublicKey = scheme.KeyGen.GenPublicKeyNew(scheme.SecretKey)
}

//export GenerateRelinearizationKey
func GenerateRelinearizationKey() {
	keyMu.Lock()
	defer keyMu.Unlock()

	scheme.RelinKey = scheme.KeyGen.GenRelinearizationKeyNew(scheme.SecretKey)
}

//export GenerateEvaluationKeys
func GenerateEvaluationKeys() {
	keyMu.Lock()
	defer keyMu.Unlock()

	scheme.EvalKeys = rlwe.NewMemEvaluationKeySet(scheme.RelinKey)
}

//export SerializeSecretKey
func SerializeSecretKey() (*C.char, C.ulong) {
	keyMu.Lock()
	defer keyMu.Unlock()

	data, err := scheme.SecretKey.MarshalBinary()
	if err != nil {
		lastError = err.Error()
		return nil, 0
	}

	arrPtr, length := SliceToCArray(data, convertByteToCChar)
	return arrPtr, length
}

//export LoadSecretKey
func LoadSecretKey(dataPtr *C.char, lenData C.ulong) C.int {
	keyMu.Lock()
	defer keyMu.Unlock()

	skSerial := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))

	sk := &rlwe.SecretKey{}
	if err := sk.UnmarshalBinary(skSerial); err != nil {
		lastError = err.Error()
		return -1
	}

	scheme.SecretKey = sk
	return 0
}
