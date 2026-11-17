"""Kernels CUDA propios (Numba CUDA) para el clasificador ASL.

Mapeo de trabajo a hilos/bloques/grids (ver docs/04_diseno_algoritmo_paralelo.md):
  normalize : 1 hilo por valor de salida (N*3*64*64), grid-stride 1D, bloque 256.
  conv3x3   : 1 hilo por pixel de salida; bloque 16x16; grid (ceil(OW/16), ceil(OH/16), N*Cout);
              los pesos de un filtro (Cin*9) se cargan cooperativamente a memoria compartida.
  maxpool2  : 1 hilo por pixel de salida; bloque 16x16; grid (ceil(OW/16), ceil(OH/16), N*C).
  dense     : GEMM por teselas 16x16 con memoria compartida; grid (ceil(Nout/16), ceil(M/16)).
  argmax    : 1 hilo por muestra (29 clases).
Los tensores son tensores de PyTorch en GPU; Numba los lee por __cuda_array_interface__ (sin copias).
"""
import math
from numba import cuda, float32, int32

T = 16          # lado de bloque/tesela
MAX_W = 64 * 9  # max Cin*9 por filtro (Cin<=64)


@cuda.jit
def k_normalize(src, dst):
    """src uint8 (N,H,W,C) -> dst float32 (N,C,H,W) dividido entre 255."""
    N, C, H, W = dst.shape
    total = N * C * H * W
    i = cuda.grid(1)
    stride = cuda.gridsize(1)
    while i < total:
        x = i % W
        t = i // W
        y = t % H
        t = t // H
        c = t % C
        n = t // C
        dst[n, c, y, x] = float32(src[n, y, x, c]) / float32(255.0)
        i += stride


@cuda.jit
def k_conv3x3_relu(x, w, b, y):
    """Convolucion valida 3x3, stride 1, + sesgo + ReLU. x(N,Cin,H,W) w(Cout,Cin,3,3) y(N,Cout,OH,OW)."""
    N, Cout, OH, OW = y.shape
    Cin = x.shape[1]
    sw = cuda.shared.array(MAX_W, float32)
    tid = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    nthreads = cuda.blockDim.x * cuda.blockDim.y
    z = cuda.blockIdx.z
    co = z % Cout
    n = z // Cout
    # carga cooperativa de los pesos del filtro `co` a memoria compartida
    k = tid
    while k < Cin * 9:
        ci = k // 9
        r = k % 9
        sw[k] = w[co, ci, r // 3, r % 3]
        k += nthreads
    cuda.syncthreads()
    ox = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    oy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if ox < OW and oy < OH:
        acc = b[co]
        for ci in range(Cin):
            for ky in range(3):
                for kx in range(3):
                    acc += x[n, ci, oy + ky, ox + kx] * sw[ci * 9 + ky * 3 + kx]
        y[n, co, oy, ox] = acc if acc > 0.0 else float32(0.0)


@cuda.jit
def k_maxpool2(x, y):
    """MaxPool 2x2 stride 2 (descarta fila/columna impar sobrante, como Keras/PyTorch)."""
    N, C, OH, OW = y.shape
    z = cuda.blockIdx.z
    c = z % C
    n = z // C
    ox = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    oy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if ox < OW and oy < OH:
        a = x[n, c, 2 * oy, 2 * ox]
        a = max(a, x[n, c, 2 * oy, 2 * ox + 1])
        a = max(a, x[n, c, 2 * oy + 1, 2 * ox])
        a = max(a, x[n, c, 2 * oy + 1, 2 * ox + 1])
        y[n, c, oy, ox] = a


@cuda.jit
def k_dense(x, w, b, y, relu):
    """y(M,Nout) = x(M,K) @ w(Nout,K)^T + b, GEMM por teselas 16x16 en memoria compartida."""
    M, K = x.shape
    Nout = w.shape[0]
    sa = cuda.shared.array((T, T), float32)
    sb = cuda.shared.array((T, T), float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * T + ty
    col = cuda.blockIdx.x * T + tx
    acc = float32(0.0)
    for t in range((K + T - 1) // T):
        kk = t * T + tx
        sa[ty, tx] = x[row, kk] if (row < M and kk < K) else float32(0.0)
        kb = t * T + ty
        wr = cuda.blockIdx.x * T + tx
        sb[ty, tx] = w[wr, kb] if (wr < Nout and kb < K) else float32(0.0)
        cuda.syncthreads()
        for j in range(T):
            acc += sa[ty, j] * sb[j, tx]
        cuda.syncthreads()
    if row < M and col < Nout:
        v = acc + b[col]
        if relu and v < 0.0:
            v = float32(0.0)
        y[row, col] = v


@cuda.jit
def k_argmax(x, out):
    i = cuda.grid(1)
    if i < x.shape[0]:
        best = 0
        bv = x[i, 0]
        for j in range(1, x.shape[1]):
            if x[i, j] > bv:
                bv = x[i, j]
                best = j
        out[i] = best


# ---------- lanzadores (definen grid y bloque) ----------
def _cd(a, b):
    return (a + b - 1) // b


def normalize(src_u8, dst_f32):
    total = dst_f32.numel()
    block = 256
    grid = min(_cd(total, block), 4096)            # grid-stride: no hace falta 1 bloque por elemento
    k_normalize[grid, block](src_u8, dst_f32)


def conv3x3_relu(x, w, b, y):
    N, Cout, OH, OW = y.shape
    k_conv3x3_relu[(_cd(OW, T), _cd(OH, T), N * Cout), (T, T, 1)](x, w, b, y)


def maxpool2(x, y):
    N, C, OH, OW = y.shape
    k_maxpool2[(_cd(OW, T), _cd(OH, T), N * C), (T, T, 1)](x, y)


def dense(x, w, b, y, relu):
    M = x.shape[0]
    Nout = w.shape[0]
    k_dense[(_cd(Nout, T), _cd(M, T)), (T, T)](x, w, b, y, relu)


def argmax(x, out):
    k_argmax[_cd(x.shape[0], 128), 128](x, out)
