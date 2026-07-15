import torch
import triton
import triton.language as tl

BLOCK_M = 16
BLOCK_N = 16
BLOCK_D = 16
BLOCK_K = BLOCK_D

@triton.jit
def _gemm_kernel(A, B, C, M: int, N: int, K: int):
    # C = A @ B, use tiling
    # so each block kernel should handle 1 submatrix of C with shape [BLOCK_M, BLOCK_N]
    # A shape [M, K], B shape [K, N] -> C shape [M, N]
    pidy, pidx = tl.program_id(0), tl.program_id(1)
    m_num_blocks = tl.cdiv(M, BLOCK_M)
    n_num_blocks = tl.cdiv(N, BLOCK_N)
    k_num_blocks = tl.cdiv(K, BLOCK_K)

    # --- C tile: [BLOCK_M, BLOCK_N], for each tile to sum up
    c_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32, device=tl.float32)

    # A[pidy * BLOCK_M][0] ... A[pidy * BLOCK_M + BLOCK_M][BLOCK_D]
    # B[0][pidx * BLOCK_N] ... B[BLOCK_K][pidx * BLOCK_N + BLOCK_N]
    for k in range(k_num_blocks):
        # loop along the shared dimension K
        # --- A tile: [BLOCK_M, BLOCK_K]
        a_row = pidy * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        a_col = k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        a_offsets = a_row * K + a_col
        a_mask = (a_row < M) & (a_col < K)
        a_tile = tl.load(A + a_offsets, mask=a_mask, other=0.0)

        # --- B tile: [BLOCK_K, BLOCK_N]
        b_row = k * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
        b_col = pidx * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        b_offsets = b_row * N + b_col
        b_mask = (b_row < K) & (b_col < N)
        b_tile = tl.load(B + b_offsets, mask=b_mask, other=0.0)

        # --- Calcualte the local C tile:
        c_tile += tl.dot(a_tile, b_tile)

    # --- Store C tile to the global C tensor
    c_row = pidy * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    c_col = pidx * BLOCK_N + tl.arange(0,BLOCK_N)[None, :]
    c_offsets = c_row * N + c_col
    c_mask = (c_row < M) & (c_col < N)
    tl.store(C + c_offsets, c_tile, mask=c_mask)


@triton.jit
def _softmax_kernel(S, P, M: int, N: int):
    # P = softmax(S), use tiling and flash attention (v1)
    # so each block kernel should handle 1 submatrix of S with shape [BLOCK_M, BLOCK_N]
    # with max trick for the exp function
    pidx, pidy = tl.program_id(0), tl.program_id(1)


@triton.jit
def _flashattention_v1_kernel(A, B, C, M: int, N: int, K: int):
    # C = A @ B, use tiling
    # so each block kernel should handle 1 submatrix of C with shape [BLOCK_M, BLOCK_N]
    # A shape [M, K], B shape [K, N] -> C shape [M, N]
    pidy, pidx = tl.program_id(0), tl.program_id(1)
    m_num_blocks = tl.cdiv(M, BLOCK_M)
    n_num_blocks = tl.cdiv(N, BLOCK_N)
    k_num_blocks = tl.cdiv(K, BLOCK_K)

    # --- C tile: [BLOCK_M, BLOCK_N], for each tile to sum up
    c_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32, device=tl.float32)

    # A[pidy * BLOCK_M][0] ... A[pidy * BLOCK_M + BLOCK_M][BLOCK_D]
    # B[0][pidx * BLOCK_N] ... B[BLOCK_K][pidx * BLOCK_N + BLOCK_N]
    for k in range(k_num_blocks):
        # loop along the shared dimension K
        # --- A tile: [BLOCK_M, BLOCK_K]
        a_row = pidy * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        a_col = k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        a_offsets = a_row * K + a_col
        a_mask = (a_row < M) & (a_col < K)
        a_tile = tl.load(A + a_offsets, mask=a_mask, other=0.0)

        # --- B tile: [BLOCK_K, BLOCK_N]
        b_row = k * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
        b_col = pidx * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        b_offsets = b_row * N + b_col
        b_mask = (b_row < K) & (b_col < N)
        b_tile = tl.load(B + b_offsets, mask=b_mask, other=0.0)

        # --- Calcualte the local C tile:
        c_tile += tl.dot(a_tile, b_tile)

    # # --- Store C tile to the global C tensor
    # c_row = pidy * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    # c_col = pidx * BLOCK_N + tl.arange(0,BLOCK_N)[None, :]
    # c_offsets = c_row * N + c_col
    # c_mask = (c_row < M) & (c_col < N)

    # --- Calculate the local softmax tile:
    # 1. find the local maxima in c_tile in each row
    tl.max(c_tile, )

# Q, K, V, output are tensors on the GPU
def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    m_num_blocks = tl.cdiv(M, BLOCK_M)
    n_num_blocks = tl.cdiv(N, BLOCK_N)
    d_num_blocks = tl.cdiv(d, BLOCK_D)

    # S = Q @ K^T, shape [M, N]
    S = torch.empty([M, N], device=Q.device, dtype=Q.dtype)
    _gemm_kernel[(m_num_blocks, n_num_blocks)](Q, K, S, M, N, d)

    # P = softmax(S), shape [M, N]
    P = torch.empty([M, N], device=Q.device, dtype=Q.dtype)
    while True:
        # use reduce to find the global softmax val

    # output = P @ V, shape [M, d]
    _gemm_kernel[(m_num_blocks, d_num_blocks)](P, V, output, M, d, N)
