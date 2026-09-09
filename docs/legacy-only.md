# Optional legacy-only builds

The default build includes both the NVSHMEM-based legacy `Buffer` and the
NCCL GIN-based V2 `ElasticBuffer`. Applications that only need `Buffer` can
exclude the V2 backend at build time:

```bash
EP_LEGACY_ONLY=1 python -m pip wheel --no-build-isolation --no-deps . -w dist
```

Build from a clean checkout or remove artifacts from a previous build before
changing this option. The generated wheel records `EP_LEGACY_ONLY=1` as an
import-time default. An explicit environment variable overrides that default,
but setting it to `0` cannot restore APIs excluded from the compiled extension.
To enable V2, rebuild without `EP_LEGACY_ONLY=1` and remove any runtime override.

In a legacy-only build, `Buffer` remains available while `ElasticBuffer` and
`EPHandle` are `None`. The V2 domain-query helpers are not exported at package
level. The build omits the NCCL backend source and its explicit NCCL link flags;
it retains NVSHMEM and the common NCCL headers. Import still performs the
existing NCCL library consistency check and JIT initialization. This option
does not make DeepEP independent of NCCL or relax PyTorch's own dependencies.

This can be useful when an application's existing NCCL version is older than
the V2 requirement, such as NCCL 2.29.7. It is independent of the CUDA wheel
variant: CUDA 12.9 does not inherently require a legacy-only build. The default
full build continues to require the dependencies listed in the main README.

The build configuration can be checked without CUDA hardware using
`python tests/test_legacy_build.py` in an environment with PyTorch installed.
This checks the default build and both explicit values of the option; it does
not compile the extension.

## Kimi K3 legacy communication

The legacy kernels support the 3584-wide routed activation and up to 16 top-k
experts in low-latency dispatch and combine. The Python bindings also register
the callable conversion needed for receive hooks.

The existing GPU regression suite accepts the corresponding geometry:

```bash
python tests/legacy/test_low_latency.py --num-processes 8 \
    --num-tokens 16 --hidden 3584 --num-topk 16 --num-experts 896
```

Run correctness and receive-hook coverage on both EP8 and cross-node EP32,
alongside the existing top-k8 cases. The test launcher interprets `WORLD_SIZE`
as the number of nodes and `RANK` as the node rank; each node spawns
`--num-processes` workers. CUDA graph replay and application integration require
GPU validation in addition to these communication tests.
