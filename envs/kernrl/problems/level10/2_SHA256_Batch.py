"""
SHA-256 Hash - Batch Processing

Computes SHA-256 hashes for multiple messages in parallel.
Critical for cryptocurrency mining and batch verification.

Optimization opportunities:
- Parallel hashing across messages
- Coalesced memory access for message words
- Shared memory for constants
- Warp-level parallelism within hash
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Batch SHA-256 computation.

    Processes multiple 512-bit messages in parallel.
    """
    def __init__(self):
        super(Model, self).__init__()

        # SHA-256 constants
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

        H0 = torch.tensor([
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
        ], dtype=torch.int64)
        self.register_buffer('H0', H0)

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """
        Compute SHA-256 hashes for batch of messages.

        Args:
            messages: (B, 64) batch of 512-bit messages (bytes as int64)

        Returns:
            hashes: (B, 8) batch of 256-bit hashes (32-bit words as int64)
        """
        B = messages.shape[0]
        device = messages.device

        # Parse messages into 32-bit words: (B, 16)
        words = torch.zeros(B, 16, dtype=torch.int64, device=device)
        for i in range(16):
            words[:, i] = (
                (messages[:, i*4].long() << 24) |
                (messages[:, i*4+1].long() << 16) |
                (messages[:, i*4+2].long() << 8) |
                messages[:, i*4+3].long()
            )

        # Process each message (could be parallelized better)
        hashes = torch.zeros(B, 8, dtype=torch.int64, device=device)

        for b in range(B):
            W = torch.zeros(64, dtype=torch.int64, device=device)
            W[:16] = words[b]

            # Extend to 64 words
            for i in range(16, 64):
                s0 = (((W[i-15] >> 7) | (W[i-15] << 25)) ^
                      ((W[i-15] >> 18) | (W[i-15] << 14)) ^
                      (W[i-15] >> 3)) & 0xFFFFFFFF
                s1 = (((W[i-2] >> 17) | (W[i-2] << 15)) ^
                      ((W[i-2] >> 19) | (W[i-2] << 13)) ^
                      (W[i-2] >> 10)) & 0xFFFFFFFF
                W[i] = (W[i-16] + s0 + W[i-7] + s1) & 0xFFFFFFFF

            # Working variables
            a, b_, c, d, e, f, g, h = self.H0.clone()

            # 64 rounds
            for i in range(64):
                S1 = (((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7))) & 0xFFFFFFFF
                ch = ((e & f) ^ ((~e) & g)) & 0xFFFFFFFF
                temp1 = (h + S1 + ch + self.K[i] + W[i]) & 0xFFFFFFFF
                S0 = (((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10))) & 0xFFFFFFFF
                maj = ((a & b_) ^ (a & c) ^ (b_ & c)) & 0xFFFFFFFF
                temp2 = (S0 + maj) & 0xFFFFFFFF

                h = g
                g = f
                f = e
                e = (d + temp1) & 0xFFFFFFFF
                d = c
                c = b_
                b_ = a
                a = (temp1 + temp2) & 0xFFFFFFFF

            hashes[b] = torch.stack([
                (self.H0[0] + a) & 0xFFFFFFFF,
                (self.H0[1] + b_) & 0xFFFFFFFF,
                (self.H0[2] + c) & 0xFFFFFFFF,
                (self.H0[3] + d) & 0xFFFFFFFF,
                (self.H0[4] + e) & 0xFFFFFFFF,
                (self.H0[5] + f) & 0xFFFFFFFF,
                (self.H0[6] + g) & 0xFFFFFFFF,
                (self.H0[7] + h) & 0xFFFFFFFF,
            ])

        return hashes


# Problem configuration
batch_size = 1024

def get_inputs():
    messages = torch.randint(0, 256, (batch_size, 64), dtype=torch.int64)
    return [messages]

def get_init_inputs():
    return []
