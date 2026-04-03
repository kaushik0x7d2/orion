package main

import (
	"C"
	"fmt"
	"math"
	"sync"

	"github.com/baahl-nyu/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/baahl-nyu/lattigo/v6/utils"
)

// Map to store bootstrapping.Evaluators by their slot count
// Initialize the map at package level
var bootstrapperMap = make(map[int]*bootstrapping.Evaluator)
var btpMu sync.Mutex

//export NewBootstrapper
func NewBootstrapper(
	LogPs *C.int,
	lenLogPs C.int,
	numSlots C.int,
) C.int {
	btpMu.Lock()
	defer btpMu.Unlock()

	slots := int(numSlots)

	if _, exists := bootstrapperMap[slots]; exists {
		return 0
	}

	// If not initialized for this slot count, create a new one
	logP := CArrayToSlice(LogPs, lenLogPs, convertCIntToInt)

	btpParametersLit := bootstrapping.ParametersLiteral{
		LogN:     utils.Pointy(scheme.Params.LogN()),
		LogP:     logP,
		Xs:       scheme.Params.Xs(),
		LogSlots: utils.Pointy(int(math.Log2(float64(slots)))),
	}

	btpParams, err := bootstrapping.NewParametersFromLiteral(
		*scheme.Params, btpParametersLit)
	if err != nil {
		lastError = err.Error()
		return -1
	}

	btpKeys, _, err := btpParams.GenEvaluationKeys(scheme.SecretKey)
	if err != nil {
		lastError = err.Error()
		return -1
	}

	var btpEval *bootstrapping.Evaluator
	if btpEval, err = bootstrapping.NewEvaluator(btpParams, btpKeys); err != nil {
		lastError = err.Error()
		return -1
	}

	// Store the new evaluator in the map
	bootstrapperMap[slots] = btpEval
	return 0
}

//export Bootstrap
func Bootstrap(ciphertextID, numSlots C.int) C.int {
	btpMu.Lock()
	defer btpMu.Unlock()

	ctIn := RetrieveCiphertext(int(ciphertextID))
	bootstrapper, err := GetBootstrapper(int(numSlots))
	if err != nil {
		lastError = err.Error()
		return -1
	}

	ctBtp := ctIn.CopyNew()
	ctBtp.LogDimensions.Cols = bootstrapper.LogMaxSlots()

	ctOut, err := bootstrapper.Bootstrap(ctBtp)
	if err != nil {
		lastError = err.Error()
		return -1
	}

	postscale := int(1 << (scheme.Params.LogMaxSlots() - bootstrapper.LogMaxSlots()))
	scheme.Evaluator.Mul(ctOut, postscale, ctOut)

	ctOut.LogDimensions.Cols = scheme.Params.LogMaxSlots()

	idx := PushCiphertext(ctOut)
	return C.int(idx)
}

func GetBootstrapper(numSlots int) (*bootstrapping.Evaluator, error) {
	bootstrapper, exists := bootstrapperMap[numSlots]
	if !exists {
		return nil, fmt.Errorf("no bootstrapper found for slot count: %d", numSlots)
	}
	return bootstrapper, nil
}

//export DeleteBootstrappers
func DeleteBootstrappers() {
	btpMu.Lock()
	defer btpMu.Unlock()

	bootstrapperMap = make(map[int]*bootstrapping.Evaluator)
}
