"""
AES-128 ECB Encryption

Encrypts data using AES-128 in ECB mode (for simplicity).
Note: ECB is insecure for real use; this is for kernel optimization practice.

AES operates on 16-byte blocks through:
1. SubBytes - S-box substitution
2. ShiftRows - row rotation
3. MixColumns - column mixing
4. AddRoundKey - XOR with round key

Optimization opportunities:
- T-table implementation (combined operations)
- Parallel block processing
- Shared memory for S-box/T-tables
- Bitsliced implementation
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    AES-128 ECB encryption.
    """
    def __init__(self):
        super(Model, self).__init__()

        # AES S-box (substitution box)
        SBOX = [
            0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
            0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
            0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
            0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
            0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
            0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
            0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
            0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
            0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
            0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
            0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
            0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
            0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
            0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
            0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
            0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
        ]
        self.register_buffer('sbox', torch.tensor(SBOX, dtype=torch.int64))

        # Round constants
        RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]
        self.register_buffer('rcon', torch.tensor(RCON, dtype=torch.int64))

    def _sub_bytes(self, state: torch.Tensor) -> torch.Tensor:
        """Apply S-box substitution."""
        return self.sbox[state.long()]

    def _shift_rows(self, state: torch.Tensor) -> torch.Tensor:
        """Shift rows of state matrix."""
        # state is (4, 4) - rows are shifted by 0, 1, 2, 3 positions
        result = state.clone()
        result[1] = torch.roll(state[1], -1)
        result[2] = torch.roll(state[2], -2)
        result[3] = torch.roll(state[3], -3)
        return result

    def _xtime(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by x in GF(2^8)."""
        return ((x << 1) ^ (((x >> 7) & 1) * 0x1b)) & 0xFF

    def _mix_column(self, col: torch.Tensor) -> torch.Tensor:
        """Mix one column."""
        t = col[0] ^ col[1] ^ col[2] ^ col[3]
        result = torch.zeros(4, dtype=col.dtype, device=col.device)
        result[0] = (col[0] ^ t ^ self._xtime(col[0] ^ col[1])) & 0xFF
        result[1] = (col[1] ^ t ^ self._xtime(col[1] ^ col[2])) & 0xFF
        result[2] = (col[2] ^ t ^ self._xtime(col[2] ^ col[3])) & 0xFF
        result[3] = (col[3] ^ t ^ self._xtime(col[3] ^ col[0])) & 0xFF
        return result

    def _mix_columns(self, state: torch.Tensor) -> torch.Tensor:
        """Apply MixColumns transformation."""
        result = torch.zeros_like(state)
        for i in range(4):
            result[:, i] = self._mix_column(state[:, i])
        return result

    def _add_round_key(self, state: torch.Tensor, round_key: torch.Tensor) -> torch.Tensor:
        """XOR state with round key."""
        return state ^ round_key

    def forward(self, plaintext: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Encrypt plaintext block with AES-128.

        Args:
            plaintext: (16,) 16-byte block
            key: (16,) 16-byte key

        Returns:
            ciphertext: (16,) encrypted block
        """
        device = plaintext.device

        # Key expansion (simplified - generates 11 round keys)
        round_keys = torch.zeros(11, 4, 4, dtype=torch.int64, device=device)
        round_keys[0] = key.reshape(4, 4).T

        for i in range(1, 11):
            prev = round_keys[i-1]
            temp = prev[:, 3].clone()
            # RotWord
            temp = torch.roll(temp, -1)
            # SubWord
            temp = self.sbox[temp.long()]
            # Add Rcon
            temp[0] = temp[0] ^ self.rcon[i-1]
            # Generate round key
            round_keys[i, :, 0] = prev[:, 0] ^ temp
            for j in range(1, 4):
                round_keys[i, :, j] = round_keys[i, :, j-1] ^ prev[:, j]

        # Initial state
        state = plaintext.reshape(4, 4).T.clone()

        # Initial round
        state = self._add_round_key(state, round_keys[0])

        # Main rounds (1-9)
        for r in range(1, 10):
            state = self._sub_bytes(state)
            state = self._shift_rows(state)
            state = self._mix_columns(state)
            state = self._add_round_key(state, round_keys[r])

        # Final round (no MixColumns)
        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        state = self._add_round_key(state, round_keys[10])

        return state.T.flatten()


# Problem configuration
def get_inputs():
    plaintext = torch.randint(0, 256, (16,), dtype=torch.int64)
    key = torch.randint(0, 256, (16,), dtype=torch.int64)
    return [plaintext, key]

def get_init_inputs():
    return []
