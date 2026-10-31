# AMD Strix Halo (Ryzen AI Max, Radeon 8060S, gfx1151) build toolchain: ROCm 10.1.0 from AMD's signed Ubuntu 26.04
# repository, the recipe of docs/STRIX_HALO.md section 1 in a container.  Every input is pinned: the base image by
# digest, the Ubuntu packages by the archive snapshot of one date, ROCm by its signing key's sha256 and an exact
# package version.
#
#   docker build -f docker/hip-gfx1151.Dockerfile --target toolchain -t unbiased-strata-toolchain:gfx1151 .
#
# Needs BuildKit (docker-buildx) for ADD --checksum.

# Ubuntu 26.04, tag resolute-20261024.1 (the newest, pinned by digest)
FROM docker.io/library/ubuntu@sha256:6a06111c3b0a8600feadb270aec771dc239bf407b25625f93f44ab00e11f4d3c AS base
ENV DEBIAN_FRONTEND=noninteractive
# The archive as it was on this date.  The base image has no CA certificates and the snapshot server is HTTPS
# only, so the first install (ca-certificates alone) skips the TLS check; apt still verifies every list and
# package against the Ubuntu archive key, as it does on the base image's plain-HTTP archive.  Everything after
# it is fetched with the TLS check on.
ARG UBUNTU_SNAPSHOT=20261030T000000Z
RUN sed -i -E "s#http://(archive|security).ubuntu.com/ubuntu/#https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#" \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get -o Acquire::https::Verify-Peer=false update -q \
    && apt-get -o Acquire::https::Verify-Peer=false install -y -q --no-install-recommends ca-certificates \
    && apt-get update -q \
    && apt-get install -y -q --no-install-recommends make g++ cmake ninja-build git python3 curl file \
    && rm -rf /var/lib/apt/lists/*

FROM base AS rocm
# ROCm 10.1.0 for gfx1151 (hipBLASLt 1.4.1 = 100401, the version tools/hip/gfx1151-hipblaslt-100401.txt is measured
# with; AMD clang 24) from AMD's Ubuntu 26.04 repository (rocm.docs.amd.com, install with the package manager).
# The signing key is pinned by sha256 (AMD MLSE DevOps, D0F0 04A0 025A 1145 C780 7FCD 0701 EAC4 D5E0 2107); apt checks
# the repository's signature and every package against it (the key must be readable by apt's own user: chmod 644).
# amdrocm-core-dev10.1-gfx1151 requires each of its 48 parts at exactly the same version, so one pin pins all 49.
ADD --checksum=sha256:090a4bf906822c5eb450ae8024b728b9281f2fccb0304641eb75018e4ec6a6dd \
    https://stable.repo.amd.com/rocm/gpg/packages.gpg /etc/apt/keyrings/amdrocm.asc
RUN chmod 644 /etc/apt/keyrings/amdrocm.asc \
    && printf 'Types: deb\nURIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/\nSuites: stable\nComponents: main\nArchitectures: amd64\nSigned-By: /etc/apt/keyrings/amdrocm.asc\n' \
         > /etc/apt/sources.list.d/amdrocm.sources \
    && apt-get update -q \
    && apt-get install -y -q --no-install-recommends amdrocm-core-dev10.1-gfx1151=10.1.0-3 \
    && test "$(dpkg-query -W 'amdrocm*' | cut -f2 | sort -u)" = 10.1.0-3 \
    && rm -rf /var/lib/apt/lists/*

FROM rocm AS toolchain
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd \
    PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/rocm_sysdeps/lib:/opt/rocm/lib/llvm/lib

# The engine for gfx1151, built with no network: everything it compiles is in the build context (the fork with its
# third_party/llama.cpp submodule checked out).  The cmake flags are docs/STRIX_HALO.md section 2's:
# STRATA_PREFILL_MMQ (the prompt experts' MMQ kernels) is not set by `make build BACKEND=hip` itself.  PORTABLE: the
# CPU code for any AVX2 CPU, not the build machine's, as upstream's release build.
#
#   git submodule update --init
#   docker build -f docker/hip-gfx1151.Dockerfile --target build -t unbiased-strata-build:gfx1151 .
FROM toolchain AS build
COPY . /src
WORKDIR /src
ARG JOBS=16
RUN --network=none test -f third_party/llama.cpp/ggml/CMakeLists.txt \
    && make build BACKEND=hip HIP_ARCHS=gfx1151 ROCM_PATH=/opt/rocm PORTABLE=1 JOBS=${JOBS} \
       CMAKE_EXTRA="-DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
         '-DCMAKE_PREFIX_PATH=/opt/rocm;/opt/rocm/lib/rocm_sysdeps;/opt/rocm/lib/llvm' \
         '-DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode'"

# The checks to run first on a Strix Halo box (the tests image below): strata-device lists the GPUs and whether this
# build can run them; the HIP parity tests of the prompt path (the hipBLASLt table, the exact HC down kernel, MMQ); and
# tools/hip/tune_hipblaslt, which recalibrates the hipBLASLt table for the ROCm in this image (docs/AMD_HIP.md).
# Same flags as the build stage, plus the tests, in a folder of their own.
FROM build AS test-build
RUN --network=none set -e; \
    make configure BACKEND=hip HIP_ARCHS=gfx1151 ROCM_PATH=/opt/rocm PORTABLE=1 BUILD_DIR=build-tests \
       CMAKE_EXTRA="-DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
         '-DCMAKE_PREFIX_PATH=/opt/rocm;/opt/rocm/lib/rocm_sysdeps;/opt/rocm/lib/llvm' \
         '-DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode'"; \
    T="strata-device hip_prefill_hcd_exact_parity hip_prefill_hipblaslt_gemm hip_prefill_mmq_parity tune_hipblaslt"; \
    cmake --build build-tests -j ${JOBS} --target $T; \
    mkdir -p /tb; for t in $T; do cp build-tests/bin/$t /tb/; done

# What the runtime image takes from the build: the three programs, the hipBLASLt table calibrated for ROCm 10.1.0's
# hipBLASLt (tools/hip/gfx1151-hipblaslt-100401-rocm10.1.0.txt: same version number as 7.14.1's, other solution ids), the ROCm
# libraries ldd names (at their own paths: rocBLAS / hipBLASLt find their kernel files beside them), the rocBLAS
# and hipBLASLt kernel folders (ROCm 10.1's packages keep their GPU code there, gfx1151 only; no .kpack archive).
FROM build AS runtime-files
RUN --network=none set -e; \
    B=/rt/opt/unbiased-strata; mkdir -p $B/bin $B/share /rt/opt/rocm/lib; \
    cp build/bin/strata build/bin/strata-poly build/bin/unbiased-strata $B/bin/; \
    cp tools/hip/gfx1151-hipblaslt-100401-rocm10.1.0.txt $B/share/; \
    for l in $(ldd build/bin/strata build/bin/strata-poly build/bin/unbiased-strata \
               | awk '$3 ~ /^\/opt\/rocm\// {print $3}' | sort -u); do \
      mkdir -p /rt$(dirname $l); cp -L $l /rt$(dirname $l)/; done; \
    cp -a /opt/rocm/lib/rocblas /opt/rocm/lib/hipblaslt /rt/opt/rocm/lib/

# The runtime image: the plain base (no compiler, no apt packages) and the files above.  It needs no network.
# GPU access: --device /dev/kfd --device /dev/dri and the host's render / video groups (--group-add).
#
#   docker build --provenance=false -f docker/hip-gfx1151.Dockerfile -t unbiased-strata:gfx1151 .
#   (--provenance=false: without it BuildKit adds a per-build record and the image ID changes on every build)
#   docker run --network none --device /dev/kfd --device /dev/dri --group-add <render gid> --group-add <video gid> \
#     -v /models:/models:ro unbiased-strata:gfx1151 --engine strata -m <shard 1> ... -- <engine options>
FROM docker.io/library/ubuntu@sha256:6a06111c3b0a8600feadb270aec771dc239bf407b25625f93f44ab00e11f4d3c AS runtime
COPY --from=runtime-files /rt/ /
ENV PATH=/opt/unbiased-strata/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/rocm_sysdeps/lib:/opt/rocm/lib/llvm/lib \
    STRATA_HIPBLASLT_TUNING=/opt/unbiased-strata/share/gfx1151-hipblaslt-100401-rocm10.1.0.txt
RUN --network=none ! ldd /opt/unbiased-strata/bin/strata /opt/unbiased-strata/bin/strata-poly \
      /opt/unbiased-strata/bin/unbiased-strata | grep 'not found'
ENTRYPOINT ["unbiased-strata"]

# The runtime image plus the checks of test-build, in /opt/unbiased-strata/tests (on PATH).  Default command: list the GPUs.
#   docker build --provenance=false -f docker/hip-gfx1151.Dockerfile --target tests -t unbiased-strata-tests:gfx1151 .
#   docker run --rm --network none --device /dev/kfd --device /dev/dri --group-add <render gid> --group-add <video gid> \
#     unbiased-strata-tests:gfx1151 hip_prefill_hcd_exact_parity
FROM runtime AS tests
COPY --from=test-build /tb/ /opt/unbiased-strata/tests/
ENV PATH=/opt/unbiased-strata/tests:/opt/unbiased-strata/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN --network=none ! ldd /opt/unbiased-strata/tests/* | grep 'not found'
ENTRYPOINT []
CMD ["strata-device", "--list-devices"]

# The offline pack tools (README "Preparing a Flash-Next pack"): tools/ with numpy and regex from the same Ubuntu
# snapshot, the submodule's gguf-py where tools/_paths.py looks for it (it imports yaml on load), and the shipped
# expert profile (it ranks all 24,576 pairs; copy it into the pack).  CPU only, no GPU.
#   docker build --provenance=false -f docker/hip-gfx1151.Dockerfile --target pack-tools -t unbiased-strata-pack:gfx1151 .
#   docker run --rm --network none --user 1000:1000 -v /srv/models:/models:ro -v <out>:/out unbiased-strata-pack:gfx1151 \
#     python3 -I /opt/unbiased-strata/tools/iq_pack.py --gguf /models/<shard 1> --out /out/<pack> --compat-bf16
FROM base AS pack-tools
RUN apt-get update -q && apt-get install -y -q --no-install-recommends python3-numpy python3-regex python3-yaml \
    && rm -rf /var/lib/apt/lists/*
COPY tools /opt/unbiased-strata/tools
COPY third_party/llama.cpp/gguf-py /opt/unbiased-strata/third_party/llama.cpp/gguf-py
COPY data/expert-profile.bin /opt/unbiased-strata/data/expert-profile.bin

# The default target (a plain `docker build`) is the runtime image: it has to be the last stage.
FROM runtime
