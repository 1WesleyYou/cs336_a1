import torch
import triton
import triton.language as tl

BLOCK_M = 16
BLOCK_N = 16
BLOCK_D = 16
BLOCK_K = BLOCK_D

dev = torch.cuda.current_device()
props = triton.runtime.driver.active.utils.get_device_properties(dev)
SM_FP32 = int(triton.cdiv(props["max_shared_mem"], torch.float32.itemsize))


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
    c_tile = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

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
    c_col = pidx * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    c_offsets = c_row * N + c_col
    c_mask = (c_row < M) & (c_col < N)
    tl.store(C + c_offsets, c_tile, mask=c_mask)


@triton.jit
def _flashattention_v1_kernel(Q, K, V, O, MTensor, LTensor, M: int, N: int, d: tl.constexpr, kv_id: int, KV_BLOCK_SIZE: tl.constexpr, Q_BLOCK_SIZE: tl.constexpr, BLOCK_D: tl.constexpr):
    q_id = tl.program_id(0)

    # --- Calculate the sub Q tensor with the given K V
    q_row = tl.arange(0, Q_BLOCK_SIZE)[:, None] + q_id * Q_BLOCK_SIZE
    q_col = tl.arange(0, BLOCK_D)[None, :]
    q_mask = (q_row < M) & (q_col < d)
    q_offsets = q_row * d + q_col

    k_col = tl.arange(0, KV_BLOCK_SIZE)[None, :] + kv_id * KV_BLOCK_SIZE
    k_row = tl.arange(0, BLOCK_D)[:, None]
    k_mask = (k_col < N) & (k_row < d)
    k_offsets = k_row + k_col * d

    v_row = tl.arange(0, KV_BLOCK_SIZE)[:, None] + kv_id * KV_BLOCK_SIZE
    v_col = tl.arange(0, BLOCK_D)[None, :]
    v_mask = (v_row < N) & (v_col < d)
    v_offsets = v_row * d + v_col

    ml_col = tl.arange(0, Q_BLOCK_SIZE)[:, None] + q_id * Q_BLOCK_SIZE
    ml_mask = ml_col < M
    ml_offsets = ml_col

    q = tl.load(Q + q_offsets, mask=q_mask, other=0.0)
    k = tl.load(K + k_offsets, mask=k_mask, other=0.0)
    v = tl.load(V + v_offsets, mask=v_mask, other=0.0)
    old_o = tl.load(O + q_offsets, mask=q_mask, other=0.0)
    old_m = tl.load(MTensor + ml_offsets, mask=ml_mask)
    old_l = tl.load(LTensor + ml_offsets, mask=ml_mask)

    local_s = tl.dot(q, k) / (d ** 0.5)  # shape [Q_BLOCK_SIZE, KV_BLOCK_SIZE]
    key_ok = (kv_id * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)[None, :]) < N
    local_s = tl.where(key_ok, local_s, -float("inf"))

    # --- Softmax in row of S
    local_m = tl.max(local_s, axis=-1, keep_dims=True)
    local_l = tl.sum(tl.exp(local_s - local_m), axis=-1, keep_dims=True)
    local_f = tl.exp(local_s - local_m)

    new_m = tl.maximum(old_m, local_m)

    # note here the old value means previous KV blocks, not the other parallel running Q blocks
    new_f = tl.exp(old_m - new_m) * old_o * old_l + tl.dot(tl.exp(local_m - new_m) * local_f, v)
    new_l = tl.exp(old_m - new_m) * old_l + tl.exp(local_m - new_m) * local_l
    new_o = new_f / new_l

    # --- Store back the l, m, o
    tl.store(O + q_offsets, new_o, mask=q_mask)
    tl.store(MTensor + ml_offsets, new_m, mask=ml_mask)
    tl.store(LTensor + ml_offsets, new_l, mask=ml_mask)


def flashattention_v1(Q, K, V, O, m, l, M: int, N: int, d: int, num_q_blocks, num_kv_blocks, KV_BLOCK_SIZE: int, Q_BLOCK_SIZE: int, BLOCK_D: int):
    # keep K, V in the outer kernel, calculate the Q for each K, V
    for j in range(num_kv_blocks):
        _flashattention_v1_kernel[(num_q_blocks,)](Q, K, V, O, m, l, M, N, d, j, KV_BLOCK_SIZE, Q_BLOCK_SIZE, BLOCK_D)


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):

    KV_BLOCK_SIZE = 16  # triton.cdiv(SM_FP32, (4 * d))
    Q_BLOCK_SIZE = 16   # min(triton.cdiv(SM_FP32, (4 * d)), d)
    BLOCK_D = max(16, triton.next_power_of_2(d))

    num_kv_blocks = triton.cdiv(N, KV_BLOCK_SIZE)
    num_q_blocks = triton.cdiv(M, Q_BLOCK_SIZE)

    # initialize output tensor O, l, m
    output.zero_()
    l = torch.zeros([M, 1], dtype=torch.float32, device=Q.device)
    m = torch.full([M, 1], -float("inf"), dtype=torch.float32, device=Q.device)

    flashattention_v1(Q, K, V, output, m, l, M, N, d, num_q_blocks, num_kv_blocks, KV_BLOCK_SIZE, Q_BLOCK_SIZE, BLOCK_D)
