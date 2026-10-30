# AMD Strix Halo (Ryzen AI Max, Radeon 8060S, gfx1151) build toolchain: ROCm 7.14.1 from AMD's TheRock builds,
# the recipe of docs/STRIX_HALO.md section 1 in a container.  Every input is pinned: the base image by digest, the
# Ubuntu packages by the archive snapshot of one date, ROCm by URL and sha256.
#
#   docker build -f docker/hip-gfx1151.Dockerfile --target toolchain -t unbiased-strata-toolchain:gfx1151 .
#
# Needs BuildKit (docker-buildx) for ADD --checksum.

# Ubuntu 26.04, tag resolute-20260927
FROM docker.io/library/ubuntu@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7 AS base
ENV DEBIAN_FRONTEND=noninteractive
# The archive as it was on this date.  The base image has no CA certificates and the snapshot server is HTTPS
# only, so the first install (ca-certificates alone) skips the TLS check; apt still verifies every list and
# package against the Ubuntu archive key, as it does on the base image's plain-HTTP archive.  Everything after
# it is fetched with the TLS check on.
ARG UBUNTU_SNAPSHOT=20261010T000000Z
RUN sed -i -E "s#http://(archive|security).ubuntu.com/ubuntu/#https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#" \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get -o Acquire::https::Verify-Peer=false update -q \
    && apt-get -o Acquire::https::Verify-Peer=false install -y -q --no-install-recommends ca-certificates \
    && apt-get update -q \
    && apt-get install -y -q --no-install-recommends make g++ cmake ninja-build git python3 curl file \
    && rm -rf /var/lib/apt/lists/*

FROM base AS rocm
# ROCm 7.14.1 for gfx1151 (hipBLASLt 100401, the version tools/hip/gfx1151-hipblaslt-100401.txt is measured with)
ADD --checksum=sha256:c40e8f2bd6630a7d11557c762b99c6fa8afb04c9bd0e51ed1675ee1ca24afb00 \
    https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.1.tar.gz /tmp/rocm.tar.gz
RUN mkdir -p /opt/rocm && tar -xzf /tmp/rocm.tar.gz -C /opt/rocm && rm /tmp/rocm.tar.gz

FROM base AS toolchain
COPY --from=rocm /opt/rocm /opt/rocm
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

# What the runtime image takes from the build: the three programs, the hipBLASLt table for ROCm 7.14.1, the ROCm
# libraries ldd names (at their own paths: rocBLAS / hipBLASLt find their kernel files beside them), the rocBLAS
# and hipBLASLt kernel folders, and blas_lib_gfx1151.kpack (the GPU code of rocBLAS, hipBLASLt and rocSOLVER; the
# other TheRock archives - fft, hiptensor, rand, rccl, rocalution - belong to libraries nothing here links).
FROM build AS runtime-files
RUN --network=none set -e; \
    B=/rt/opt/unbiased-strata; mkdir -p $B/bin $B/share /rt/opt/rocm/.kpack; \
    cp build/bin/strata build/bin/strata-poly build/bin/unbiased-strata $B/bin/; \
    cp tools/hip/gfx1151-hipblaslt-100401.txt $B/share/; \
    for l in $(ldd build/bin/strata build/bin/strata-poly build/bin/unbiased-strata \
               | awk '$3 ~ /^\/opt\/rocm\// {print $3}' | sort -u); do \
      mkdir -p /rt$(dirname $l); cp -L $l /rt$(dirname $l)/; done; \
    cp -a /opt/rocm/lib/rocblas /opt/rocm/lib/hipblaslt /rt/opt/rocm/lib/; \
    cp /opt/rocm/.kpack/blas_lib_gfx1151.kpack /rt/opt/rocm/.kpack/

# The runtime image: the plain base (no compiler, no apt packages) and the files above.  It needs no network.
# GPU access: --device /dev/kfd --device /dev/dri and the host's render / video groups (--group-add).
#
#   docker build --provenance=false -f docker/hip-gfx1151.Dockerfile -t unbiased-strata:gfx1151 .
#   (--provenance=false: without it BuildKit adds a per-build record and the image ID changes on every build)
#   docker run --network none --device /dev/kfd --device /dev/dri --group-add <render gid> --group-add <video gid> \
#     -v /models:/models:ro unbiased-strata:gfx1151 --engine strata -m <shard 1> ... -- <engine options>
FROM docker.io/library/ubuntu@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7 AS runtime
COPY --from=runtime-files /rt/ /
ENV PATH=/opt/unbiased-strata/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/rocm_sysdeps/lib:/opt/rocm/lib/llvm/lib \
    STRATA_HIPBLASLT_TUNING=/opt/unbiased-strata/share/gfx1151-hipblaslt-100401.txt
RUN --network=none ! ldd /opt/unbiased-strata/bin/strata /opt/unbiased-strata/bin/strata-poly \
      /opt/unbiased-strata/bin/unbiased-strata | grep 'not found'
ENTRYPOINT ["unbiased-strata"]
