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
