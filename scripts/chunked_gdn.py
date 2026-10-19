"""Chunk-parallel (WY / UT-transform) gated delta rule in pure MLX ops, differentiable by autodiff.

Recurrence implemented by mlx_lm.models.gated_delta._gated_delta_step_ops (per head, state S: Dv x Dk):
    S' = g_t S;  delta_t = beta_t (v_t - S' k_t);  S_t = S' + delta_t k_t^T;  y_t = S_t q_t
Within a chunk with start state S0 and gamma_t = prod_{i<=t} g_i:
    S_t = gamma_t S0 + sum_{j<=t} (gamma_t/gamma_j) delta_j k_j^T
    (I + A) Delta = beta*(V - gamma*(K S0^T)),  A_tj = beta_t (gamma_t/gamma_j) k_t.k_j  (j<t)
    Delta = U - W S0^T,  U = T (beta V),  W = T (beta gamma K),  T = (I+A)^-1
    Y = (gamma Q) S0^T + (M o Q K^T) Delta,     M_tj = gamma_t/gamma_j (j<=t)
    S_C = gamma_C S0 + (Delta * gamma_C/gamma)^T K
T is formed by recursive 2x2 block inversion of the unit-lower-triangular matrix (stable; no Neumann series).
"""
import mlx.core as mx
import mlx.nn as nn


def tri_inv_unit_lower(A):
    """(I + A)^-1 for strictly-lower-triangular A [..., C, C], C a power of two."""
    C = A.shape[-1]
    lead = A.shape[:-2]
    Bs = mx.ones(lead + (C, 1, 1), dtype=A.dtype)
    s = 1
    while s < C:
        n = C // (2 * s)
        L21 = mx.stack([A[..., (2 * m + 1) * s:(2 * m + 2) * s, 2 * m * s:(2 * m + 1) * s] for m in range(n)], axis=-3)
        I1 = Bs[..., 0::2, :, :]
        I2 = Bs[..., 1::2, :, :]
        X = -(I2 @ L21 @ I1)
        Z = mx.zeros(lead + (n, s, s), dtype=A.dtype)
        Bs = mx.concatenate([mx.concatenate([I1, Z], axis=-1), mx.concatenate([X, I2], axis=-1)], axis=-2)
        s *= 2
    return Bs[..., 0, :, :]


def gated_delta_chunked(q, k, v, lg, beta, state=None, chunk=64):
    """q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv]; lg = log(g): [B,T,Hv]; beta: [B,T,Hv]; state [B,Hv,Dv,Dk] fp32.
    Returns y [B,T,Hv,Dv] (q.dtype) and final state (fp32)."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (r := Hv // Hk) > 1:
        q = mx.repeat(q, r, -2); k = mx.repeat(k, r, -2)
    C = chunk
    N = (T + C - 1) // C
    P = N * C - T
    f32 = mx.float32
    q, k, v = q.astype(f32), k.astype(f32), v.astype(f32)
    lg, beta = lg.astype(f32), beta.astype(f32)
    if P:
        pad4 = [(0, 0), (0, P), (0, 0), (0, 0)]
        q = mx.pad(q, pad4); k = mx.pad(k, pad4); v = mx.pad(v, pad4)
        lg = mx.pad(lg, [(0, 0), (0, P), (0, 0)]); beta = mx.pad(beta, [(0, 0), (0, P), (0, 0)])
    # -> [B, H, N, C, D]
    def chunks4(x):
        return x.reshape(B, N, C, Hv, x.shape[-1]).transpose(0, 3, 1, 2, 4)
    def chunks3(x):
        return x.reshape(B, N, C, Hv).transpose(0, 3, 1, 2)
    q, k, v = chunks4(q), chunks4(k), chunks4(v)
    lg, beta = chunks3(lg), chunks3(beta)
    cl = mx.cumsum(lg, axis=-1)                                   # [B,H,N,C]
    diff = cl[..., :, None] - cl[..., None, :]                     # [B,H,N,C,C] (t, j)
    incl = mx.tril(mx.ones((C, C), dtype=mx.bool_))
    strict = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
    Gam = mx.where(incl, mx.exp(mx.where(incl, diff, 0.0)), 0.0)  # gamma_t/gamma_j, j<=t
    kT = k.swapaxes(-1, -2)
    A = mx.where(strict, beta[..., :, None] * Gam * (k @ kT), 0.0)
    Tinv = tri_inv_unit_lower(A)
    gam = mx.exp(cl)
    U = Tinv @ (beta[..., None] * v)                               # [B,H,N,C,Dv]
    W = Tinv @ ((beta * gam)[..., None] * k)                       # [B,H,N,C,Dk]
    QK = mx.where(incl, (q @ kT) * Gam, 0.0)                       # [B,H,N,C,C]
    Qg = gam[..., None] * q
    to_end = mx.exp(cl[..., -1:] - cl)                             # [B,H,N,C]
    gC = mx.exp(cl[..., -1])                                       # [B,H,N]
    ys = []
    for n in range(N):
        S0T = state.swapaxes(-1, -2)                               # [B,H,Dk,Dv]
        D = U[:, :, n] - W[:, :, n] @ S0T                          # [B,H,C,Dv]
        ys.append(Qg[:, :, n] @ S0T + QK[:, :, n] @ D)
        state = gC[:, :, n, None, None] * state + (D * to_end[:, :, n, :, None]).swapaxes(-1, -2) @ k[:, :, n]
    y = mx.stack(ys, axis=2).reshape(B, Hv, N * C, Dv).transpose(0, 2, 1, 3)[:, :T]
    return y.astype(out_dtype), state


def log_g(A_log, a, dt_bias):
    """log of mlx_lm compute_g: g = exp(-exp(A_log) * softplus(a + dt_bias))."""
    return -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)


def make_patched_update(orig_update, chunk=64):
    def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, *, use_kernel=True,
                           lower_bound=None, allow_neg_eigval=False):
        if use_kernel or mask is not None or lower_bound is not None:
            return orig_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel,
                               lower_bound=lower_bound, allow_neg_eigval=allow_neg_eigval)
        beta = mx.sigmoid(b)
        if allow_neg_eigval:
            beta = beta * 2.0
        return gated_delta_chunked(q, k, v, log_g(A_log, a, dt_bias), beta, state, chunk=chunk)
    return gated_delta_update
