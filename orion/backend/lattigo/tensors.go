package main

import (
	"C"
	"sync"
	"unsafe"

	"github.com/baahl-nyu/lattigo/v6/core/rlwe"
)

var (
	ptHeap   = NewHeapAllocator()
	ctHeap   = NewHeapAllocator()
	tensorMu sync.Mutex
)

func PushPlaintext(plaintext *rlwe.Plaintext) int {
	return ptHeap.Add(plaintext)
}

func PushCiphertext(ciphertext *rlwe.Ciphertext) int {
	return ctHeap.Add(ciphertext)
}

func RetrievePlaintext(plaintextID int) *rlwe.Plaintext {
	return ptHeap.Retrieve(plaintextID).(*rlwe.Plaintext)
}

func RetrieveCiphertext(ciphertextID int) *rlwe.Ciphertext {
	return ctHeap.Retrieve(ciphertextID).(*rlwe.Ciphertext)
}

// ---------------------------------------- //
//             PYTHON BINDINGS              //
// ---------------------------------------- //

//export DeletePlaintext
func DeletePlaintext(plaintextID C.int) {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ptHeap.Delete(int(plaintextID))
}

//export DeleteCiphertext
func DeleteCiphertext(ciphertextID C.int) {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ctHeap.Delete(int(ciphertextID))
}

//export GetPlaintextScale
func GetPlaintextScale(plaintextID C.int) C.ulonglong {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	plaintext := RetrievePlaintext(int(plaintextID))
	scaleBig := &plaintext.Scale.Value
	scale, _ := scaleBig.Uint64()
	return C.ulonglong(scale)
}

//export GetCiphertextScale
func GetCiphertextScale(ciphertextID C.int) C.ulonglong {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ciphertext := RetrieveCiphertext(int(ciphertextID))
	scaleBig := &ciphertext.Scale.Value
	scale, _ := scaleBig.Uint64()
	return C.ulonglong(scale)
}

//export SetPlaintextScale
func SetPlaintextScale(plaintextID C.int, scale C.ulonglong) {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	plaintext := RetrievePlaintext(int(plaintextID))
	plaintext.Scale = rlwe.NewScale(uint64(scale))
}

//export SetCiphertextScale
func SetCiphertextScale(ciphertextID C.int, scale C.ulonglong) {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ciphertext := RetrieveCiphertext(int(ciphertextID))
	ciphertext.Scale = rlwe.NewScale(uint64(scale))
}

//export GetPlaintextLevel
func GetPlaintextLevel(plaintextID C.int) C.int {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	plaintext := RetrievePlaintext(int(plaintextID))
	return C.int(plaintext.Level())
}

//export GetCiphertextLevel
func GetCiphertextLevel(ciphertextID int) C.int {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ciphertext := RetrieveCiphertext(ciphertextID)
	return C.int(ciphertext.Level())
}

//export GetPlaintextSlots
func GetPlaintextSlots(plaintextID int) C.int {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	plaintext := RetrievePlaintext(plaintextID)
	slots := 1 << plaintext.LogDimensions.Cols
	return C.int(slots)
}

//export GetCiphertextSlots
func GetCiphertextSlots(ciphertextID int) C.int {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ciphertext := RetrieveCiphertext(ciphertextID)
	slots := 1 << ciphertext.LogDimensions.Cols
	return C.int(slots)
}

//export GetCiphertextDegree
func GetCiphertextDegree(ciphertextID int) C.int {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ciphertext := RetrieveCiphertext(ciphertextID)
	return C.int(ciphertext.Degree())
}

//export GetModuliChain
func GetModuliChain() (*C.ulonglong, C.ulonglong) {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	moduli := scheme.Params.Q()
	arrPtr, length := SliceToCArray(moduli, convertUint64ToCULonglong)
	return arrPtr, C.ulonglong(length)
}

//export GetAuxModuliChain
func GetAuxModuliChain() (*C.ulonglong, C.ulonglong) {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	moduli := scheme.Params.P()
	arrPtr, length := SliceToCArray(moduli, convertUint64ToCULonglong)
	return arrPtr, C.ulonglong(length)
}

//export SerializeCiphertext
func SerializeCiphertext(ciphertextID C.int) (*C.char, C.ulong) {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ciphertext := RetrieveCiphertext(int(ciphertextID))
	data, err := ciphertext.MarshalBinary()
	if err != nil {
		lastError = err.Error()
		return nil, 0
	}
	arrPtr, length := SliceToCArray(data, convertByteToCChar)
	return arrPtr, length
}

//export LoadCiphertext
func LoadCiphertext(dataPtr *C.char, lenData C.ulong) C.int {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ctSerial := CArrayToByteSlice(unsafe.Pointer(dataPtr), uint64(lenData))
	ct := &rlwe.Ciphertext{}
	if err := ct.UnmarshalBinary(ctSerial); err != nil {
		lastError = err.Error()
		return -1
	}
	idx := PushCiphertext(ct)
	return C.int(idx)
}

//export GetLivePlaintexts
func GetLivePlaintexts() (*C.int, C.ulong) {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ids := ptHeap.GetLiveKeys()
	arrPtr, length := SliceToCArray(ids, convertIntToCInt)
	return arrPtr, length
}

//export GetLiveCiphertexts
func GetLiveCiphertexts() (*C.int, C.ulong) {
	tensorMu.Lock()
	defer tensorMu.Unlock()

	ids := ctHeap.GetLiveKeys()
	arrPtr, length := SliceToCArray(ids, convertIntToCInt)
	return arrPtr, length
}
