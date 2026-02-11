"""
SHA-256 Hash - Single Message

Computes SHA-256 hash of a message block.
Fundamental cryptographic primitive used in Bitcoin, TLS, etc.

SHA-256 operates on 512-bit (64-byte) blocks, producing 256-bit hash.

Optimization opportunities:
- Unroll compression rounds
- Use registers for working variables
- Vectorized message schedule computation
- Parallel hashing of multiple messages
"""

import torch
import torch.nn as nn
import hashlib


class Model(nn.Module):
    """
    SHA-256 hash computation using PyTorch operations.

    This is a naive implementation - the optimized version should use
    bit manipulation intrinsics and unrolled loops.
    """
    def __init__(self):
        super(Model, self).__init__()

        # SHA-256 constants (first 32 bits of fractional parts of cube roots of first 64 primes)
        K = torch.tensor([
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ], dtype=torch.int64)
        self.register_buffer('K', K)

        # Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
        H0 = torch.tensor([
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ], dtype=torch.int64)
        self.register_buffer('H0', H0)

    def _rotr(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Right rotation."""
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    def _ch(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return (x & y) ^ (~x & z) & 0xFFFFFFFF

    def _maj(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return (x & y) ^ (x & z) ^ (y & z)

    def _sigma0(self, x: torch.Tensor) -> torch.Tensor:
        return self._rotr(x, 2) ^ self._rotr(x, 13) ^ self._rotr(x, 22)

    def _sigma1(self, x: torch.Tensor) -> torch.Tensor:
        return self._rotr(x, 6) ^ self._rotr(x, 11) ^ self._rotr(x, 25)

    def _gamma0(self, x: torch.Tensor) -> torch.Tensor:
        return self._rotr(x, 7) ^ self._rotr(x, 18) ^ (x >> 3)

    def _gamma1(self, x: torch.Tensor) -> torch.Tensor:
        return self._rotr(x, 17) ^ self._rotr(x, 19) ^ (x >> 10)

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Compute SHA-256 hash.

        Args:
            message: (64,) bytes as int64 tensor (one 512-bit block)

        Returns:
            hash: (8,) 32-bit words as int64 tensor (256-bit hash)
        """
        # Parse message into 16 32-bit words
        W = torch.zeros(64, dtype=torch.int64, device=message.device)
        for i in range(16):
            W[i] = (message[i*4].long() << 24) | (message[i*4+1].long() << 16) | \
                   (message[i*4+2].long() << 8) | message[i*4+3].long()

        # Extend to 64 words
        for i in range(16, 64):
            W[i] = (self._gamma1(W[i-2]) + W[i-7] + self._gamma0(W[i-15]) + W[i-16]) & 0xFFFFFFFF

        # Initialize working variables
        a, b, c, d, e, f, g, h = self.H0.clone()

        # Compression function main loop
        for i in range(64):
            T1 = (h + self._sigma1(e) + self._ch(e, f, g) + self.K[i] + W[i]) & 0xFFFFFFFF
            T2 = (self._sigma0(a) + self._maj(a, b, c)) & 0xFFFFFFFF
            h = g
            g = f
            f = e
            e = (d + T1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (T1 + T2) & 0xFFFFFFFF

        # Compute final hash
        H = torch.stack([
            (self.H0[0] + a) & 0xFFFFFFFF,
            (self.H0[1] + b) & 0xFFFFFFFF,
            (self.H0[2] + c) & 0xFFFFFFFF,
            (self.H0[3] + d) & 0xFFFFFFFF,
            (self.H0[4] + e) & 0xFFFFFFFF,
            (self.H0[5] + f) & 0xFFFFFFFF,
            (self.H0[6] + g) & 0xFFFFFFFF,
            (self.H0[7] + h) & 0xFFFFFFFF,
        ])

        return H


# Problem configuration
def get_inputs():
    # One 512-bit block (64 bytes)
    message = torch.randint(0, 256, (64,), dtype=torch.int64)
    return [message]

def get_init_inputs():
    return []
