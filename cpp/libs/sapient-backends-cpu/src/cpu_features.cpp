// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 OpenHorizon Labs Pvt Ltd — SAPIENT: AGPL-3.0-only OR commercial (see LICENSE, NOTICE)
#include "sapient/backends_cpu/cpu_features.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1UL << 13)
#endif
#elif defined(_WIN32)
#include <windows.h>
#ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
#define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif
#endif
#elif defined(__x86_64__) || defined(_M_X64)
// MSVC proper only; clang-cl takes the GNU arm below (verified: clang ships <cpuid.h> and accepts
// GNU inline asm).
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace sapient::backends_cpu::cpu_features {
namespace {

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
bool sysctl_flag(const char* name) {
    int v = 0;
    size_t len = sizeof(v);
    return ::sysctlbyname(name, &v, &len, nullptr, 0) == 0 && v != 0;
}
#endif
bool probe_dotprod() {
#if defined(__APPLE__)
    return sysctl_flag("hw.optional.arm.FEAT_DotProd");
#elif defined(__linux__)
    return (::getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(_WIN32)
    return ::IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) != 0;
#else
    return false;
#endif
}
bool probe_i8mm() {
#if defined(__APPLE__)
    return sysctl_flag("hw.optional.arm.FEAT_I8MM");
#elif defined(__linux__)
    return (::getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
    return false;
#endif
}
bool probe_avx2_fma() {
    return false;
}

#elif defined(__x86_64__) || defined(_M_X64)
bool probe_dotprod() {
    return false;
}
bool probe_i8mm() {
    return false;
}

struct CpuidRegs {
    uint32_t eax{0}, ebx{0}, ecx{0}, edx{0};
};
CpuidRegs cpuid(uint32_t leaf, uint32_t sub) {
    CpuidRegs r;
#if defined(_MSC_VER) && !defined(__clang__)
    int out[4] = {0, 0, 0, 0};
    __cpuidex(out, static_cast<int>(leaf), static_cast<int>(sub));
    r.eax = static_cast<uint32_t>(out[0]);
    r.ebx = static_cast<uint32_t>(out[1]);
    r.ecx = static_cast<uint32_t>(out[2]);
    r.edx = static_cast<uint32_t>(out[3]);
#else
    __cpuid_count(leaf, sub, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}
uint64_t xgetbv0() {
#if defined(_MSC_VER) && !defined(__clang__)
    return _xgetbv(0);
#else
    uint32_t eax = 0;
    uint32_t edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
}
// std_detect (os/x86.rs): FMA (leaf 1 ECX bit 12) and AVX2 (leaf 7 EBX bit 5) are reported only
// when the OS saves the YMM state — OSXSAVE (leaf 1 ECX bit 27) and XCR0 bits 1 and 2 set.
bool probe_avx2_fma() {
    if (cpuid(0, 0).eax < 7) return false;
    const CpuidRegs l1 = cpuid(1, 0);
    if ((l1.ecx & (1u << 27)) == 0) return false;
    if ((xgetbv0() & 0x6u) != 0x6u) return false;
    const bool fma = (l1.ecx & (1u << 12)) != 0;
    const bool avx2 = (cpuid(7, 0).ebx & (1u << 5)) != 0;
    return fma && avx2;
}

#else
bool probe_dotprod() {
    return false;
}
bool probe_i8mm() {
    return false;
}
bool probe_avx2_fma() {
    return false;
}
#endif

} // namespace

bool has_dotprod() {
    static const bool v = probe_dotprod();
    return v;
}
bool has_i8mm() {
    static const bool v = probe_i8mm();
    return v;
}
bool has_avx2_fma() {
    static const bool v = probe_avx2_fma();
    return v;
}

} // namespace sapient::backends_cpu::cpu_features
