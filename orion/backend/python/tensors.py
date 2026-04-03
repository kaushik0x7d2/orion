import sys
import math
import ctypes
import numpy as np

class PlainTensor:
    def __init__(self, scheme, ptxt_ids, shape, on_shape=None):
        if scheme is None:
            raise ValueError("PlainTensor requires a valid scheme.")
        self.scheme = scheme
        self.backend = scheme.backend
        self.encoder = scheme.encoder

        self.ids = [ptxt_ids] if isinstance(ptxt_ids, int) else ptxt_ids
        if not self.ids:
            raise ValueError("PlainTensor requires at least one plaintext ID.")
        self.shape = shape
        self.on_shape = on_shape or shape

    def __del__(self):
        if 'sys' in globals() and sys.modules and self.scheme:
            try:
                for idx in self.ids:
                    self.backend.DeletePlaintext(idx)
            except Exception: 
                pass # avoids errors for GC at program termination

    def __len__(self):
        return len(self.ids)
    
    def __str__(self):
        return str(self.decode())
    
    def mul(self, other, in_place=False):
        if not isinstance(other, CipherTensor):
            raise ValueError(f"Multiplication between PlainTensor and "
                             f"{type(other)} is not supported.")

        mul_ids = []
        for i in range(len(self.ids)):
            mul_id = self.evaluator.mul_ciphertext(
                other.ids[i], self.ids[i], in_place)
            mul_ids.append(mul_id)

        if in_place:
            return other
        return CipherTensor(self.scheme, mul_ids, self.shape, self.on_shape) 

    def __mul__(self, other):
        return self.mul(other, in_place=False)     

    def __imul__(self, other):
        return self.mul(other, in_place=True)
    
    def _check_valid(self, other):
        if isinstance(other, (PlainTensor, CipherTensor)):
            if len(self.ids) != len(other.ids):
                raise ValueError(
                    f"Tensor ID count mismatch: {len(self.ids)} vs {len(other.ids)}")

    def get_ids(self):
        return self.ids
    
    def scale(self):
        return self.backend.GetPlaintextScale(self.ids[0])
    
    def set_scale(self, scale):
        for ptxt in self.ids:
            self.backend.SetPlaintextScale(ptxt, scale)

    def level(self):
        return self.backend.GetPlaintextLevel(self.ids[0])
    
    def slots(self):
        return self.backend.GetPlaintextSlots(self.ids[0])
    
    def min(self):
        return self.decode().min()
    
    def max(self):
        return self.decode().max()
    
    def moduli(self):
        return self.backend.GetModuliChain()
    
    def decode(self):
        return self.encoder.decode(self)
    

class CipherTensor:
    def __init__(self, scheme, ctxt_ids, shape, on_shape=None):
        if scheme is None:
            raise ValueError("CipherTensor requires a valid scheme.")
        self.scheme = scheme
        self.backend = scheme.backend
        self.encryptor = scheme.encryptor
        self.evaluator = scheme.evaluator
        self.bootstrapper = scheme.bootstrapper

        self.ids = [ctxt_ids] if isinstance(ctxt_ids, int) else ctxt_ids
        if not self.ids:
            raise ValueError("CipherTensor requires at least one ciphertext ID.")
        self.shape = shape
        self.on_shape = on_shape or shape

    def __del__(self):
        if 'sys' in globals() and sys.modules and self.scheme:
            try:
                for idx in self.ids:
                    self.backend.DeleteCiphertext(idx)
            except Exception: 
                pass # avoids errors for GC at program termination

    def __len__(self):
        return len(self.ids)
    
    def __str__(self):
        ptxt = self.decrypt()
        return str(ptxt.decode())
    
    #--------------#
    #  Operations  #
    #--------------#
    
    def __neg__(self):
        neg_ids = []
        for ctxt in self.ids:
            neg_id = self.evaluator.negate(ctxt)
            neg_ids.append(neg_id)

        return CipherTensor(self.scheme, neg_ids, self.shape, self.on_shape)
    
    def add(self, other, in_place=False):
        self._check_valid(other)

        add_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                add_id = self.evaluator.add_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                add_id = self.evaluator.add_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                add_id = self.evaluator.add_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Addition between CipherTensor and "
                                 f"{type(other)} is not supported.")

            add_ids.append(add_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, add_ids, self.shape, self.on_shape)
    
    def __add__(self, other):
        return self.add(other, in_place=False)

    def __iadd__(self, other):
        return self.add(other, in_place=True)
    
    def sub(self, other, in_place=False):
        self._check_valid(other)

        sub_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                sub_id = self.evaluator.sub_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                sub_id = self.evaluator.sub_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                sub_id = self.evaluator.sub_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Subtraction between CipherTensor and "
                                 f"{type(other)} is not supported.")

            sub_ids.append(sub_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, sub_ids, self.shape, self.on_shape)
    
    def __sub__(self, other):
        return self.sub(other, in_place=False)

    def __isub__(self, other):
        return self.sub(other, in_place=True)
    
    def mul(self, other, in_place=False):
        self._check_valid(other)

        mul_ids = []
        for i in range(len(self.ids)):
            if isinstance(other, (int, float)):
                mul_id = self.evaluator.mul_scalar(
                    self.ids[i], other, in_place)
            elif isinstance(other, PlainTensor):
                mul_id = self.evaluator.mul_plaintext(
                    self.ids[i], other.ids[i], in_place)
            elif isinstance(other, CipherTensor):
                mul_id = self.evaluator.mul_ciphertext(
                    self.ids[i], other.ids[i], in_place)
            else:
                raise ValueError(f"Multiplication between CipherTensor and "
                                 f"{type(other)} is not supported.")
            
            mul_ids.append(mul_id)

        if in_place:
            return self
        return CipherTensor(self.scheme, mul_ids, self.shape, self.on_shape) 
    
    def __mul__(self, other):
        return self.mul(other, in_place=False)     

    def __imul__(self, other):
        return self.mul(other, in_place=True)
    
    def roll(self, amount, in_place=False):
        rot_ids = []
        for ctxt in self.ids:
            rot_id = self.evaluator.rotate(ctxt, amount, in_place)
            rot_ids.append(rot_id)

        return CipherTensor(self.scheme, rot_ids, self.shape, self.on_shape)
    
    def _check_valid(self, other):
        if isinstance(other, (PlainTensor, CipherTensor)):
            if len(self.ids) != len(other.ids):
                raise ValueError(
                    f"Tensor ID count mismatch: {len(self.ids)} vs {len(other.ids)}")

    #----------------------
    #
    #---------------------
    
    def scale(self):
        return self.backend.GetCiphertextScale(self.ids[0])
    
    def set_scale(self, scale):
        for ctxt in self.ids:
            self.backend.SetCiphertextScale(ctxt, scale)

    def level(self):
        return self.backend.GetCiphertextLevel(self.ids[0])
    
    def slots(self):
        return self.backend.GetCiphertextSlots(self.ids[0])
    
    def degree(self):
        return self.backend.GetCiphertextDegree(self.ids[0])
    
    def min(self):
        return self.decrypt().min()
    
    def max(self):
        return self.decrypt().max()
    
    def moduli(self):
        return self.backend.GetModuliChain()
    
    def bootstrap(self):
        elements = self.on_shape.numel()
        slots = 2 ** math.ceil(math.log2(elements))
        slots = int(min(self.slots(), slots)) # sparse bootstrapping
        
        btp_ids = []
        for ctxt in self.ids:
            btp_id = self.bootstrapper.bootstrap(ctxt, slots)
            btp_ids.append(btp_id)

        return CipherTensor(self.scheme, btp_ids, self.shape, self.on_shape)
        
    def decrypt(self):
        return self.encryptor.decrypt(self)

    def serialize(self):
        """Serialize all ciphertexts to a list of byte arrays."""
        serialized = []
        for ctxt_id in self.ids:
            arr, data_ptr = self.backend.SerializeCiphertext(ctxt_id)
            serialized.append(bytes(arr))
            self.backend.FreeCArray(ctypes.cast(data_ptr, ctypes.c_void_p))
        return {
            "ciphertexts": serialized,
            "shape": list(self.shape) if hasattr(self.shape, '__iter__') else [self.shape],
            "on_shape": list(self.on_shape) if hasattr(self.on_shape, '__iter__') else [self.on_shape],
        }

    @classmethod
    def from_serialized(cls, scheme, data):
        """Deserialize byte arrays back into a CipherTensor."""
        import torch
        ctxt_ids = []
        for ct_bytes in data["ciphertexts"]:
            arr = np.frombuffer(ct_bytes, dtype=np.uint8)
            ctxt_id = scheme.backend.LoadCiphertext(arr)
            ctxt_ids.append(ctxt_id)
        shape = torch.Size(data["shape"])
        on_shape = torch.Size(data["on_shape"])
        return cls(scheme, ctxt_ids, shape, on_shape)