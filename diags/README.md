# GPU Diagnostic Scripts

Reusable diagnostic scripts originally created during Bridges-2 GPU debugging sessions.
These are **not** part of the test suite — they're standalone scripts for targeted
investigation of accuracy bottlenecks.

Run on a GPU node:
```bash
python diags/diag_cl_comprehensive.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `diag_cl_comprehensive.py` | Dense l-sampling + n_k_fine convergence test |
| `diag_cl_fast_v2.py` | Apples-to-apples C_l comparison (massless ncdm + RECFAST) |
| `diag_class_xe_oracle.py` | Inject CLASS-exact x_e to isolate error sources |
| `diag_pert_vars.py` | Compare raw perturbation variables against CLASS at specific k |
| `diag_source_decomp_v2.py` | Decompose TT error into SW+Doppler vs ISW contributions |
| `timing_test.py` | JIT compilation and execution timing benchmarks |
