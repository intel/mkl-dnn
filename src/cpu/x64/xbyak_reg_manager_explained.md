# Xbyak x86-64 Register Manager

## Summary of Proposal

This document serves as an overview and proposal for a user-level register manager to allow developers working with Xbyak on x86-64 to more easily track, manage and (de)allocate registers.

This implementation is modeled after the [Xbyak_aarch64 Register Manager (PR #4587)](https://github.com/uxlfoundation/oneDNN/pull/4587) but adapted for x86-64 architecture and its specific characteristics.

## Desired Outcomes

1. Provide tracking and standardized, automatic allocate and free methods for registers.
2. Guarantee unique ownership of registers for each of the three register families:
   - General Purpose (GP): Reg8, Reg16, Reg32, Reg64
   - Vector (Vec): Xmm, Ymm, Zmm
   - Opmask: k0-k7 (AVX-512)
3. Implement scoped registers following the RAII technique.
4. Ensure compatibility & minimum friction with current register manager techniques in oneDNN.
5. Support Intel APX extended registers (r16-r31) with automatic detection.
6. Support AVX-512 extended vector registers (zmm16-zmm31) with automatic detection.

## Context & Motivation

The motivation for this project is to reduce the manual user-level register management for projects using Xbyak on x86-64 and to support both Windows x64 and System V AMD64 ABI calling conventions with automatic platform detection.

### Key Differences from AArch64 Implementation

| Aspect | x86-64 | AArch64 |
|--------|--------|---------|
| **GP Registers** | 16 (r0-r15), 32 with APX (r0-r31) | 31 (x0-x30) |
| **Vector Registers** | 16 (zmm0-zmm15) SSE/AVX/AVX2, 32 (zmm0-zmm31) with AVX-512 | 32 (v0-v31) for FP/SIMD |
| **Mask Registers** | 8 opmask (k0-k7) for AVX-512 | 16 predicate (p0-p15) for SVE |
| **Special Registers** | rsp (r4), rbp (r5), k0 | x16/x17 (IPC), x18 (platform), x29 (FP), x30 (LR) |
| **Register Aliasing** | RAX/EAX/AX/AL share same register<br>XMM/YMM/ZMM share same register | Different sizes access same register |
| **Calling Convention** | Windows x64 / System V AMD64 ABI (auto-detected) | AAPCS64 |
| **Extended Support** | Intel APX adds r16-r31, AVX-512 adds zmm16-zmm31 (runtime detection) | SVE scalable vectors |

## Assumptions

- Initially, the majority of kernels will continue to use the old method of register handling.
- Calling convention automatically selected at compile time via `_WIN32` macro (Windows x64 vs System V AMD64).
- Call-clobbered (volatile) registers are allocated first from the free pool.
- There is no specific order that registers should be allocated in their respective sets.
- Intel APX support is detected at runtime and extended GP registers (r16-r31) are automatically made available.
- AVX-512 support is detected at runtime and extended vector registers (zmm16-zmm31) are automatically made available.
- Register aliasing (e.g., RAX/EAX/AX/AL) is handled by tracking at the index level.

## Implementation

### Overview

The register manager maintains three families of registers, each with three sets that hold indices representing the registers within the family. The `*` represents one of the three register families (GP, Vec, Opmask). Each family has these three sets that hold indices which represent the registers within the family. Further, each family has a used set, which simply tracks any register that has been touched by the manager up to this point.

```
+-------------------------------------+
|     RegPoolManager                  |
+-------------------------------------+
| GP Registers (0-15 or 0-31)         |
|  +-------------+                    |
|  | free_regs   | ---alloc()-->      |
|  +-------------+                    |
|  +-------------+                    |
|  | in_use      | <--alloc()---      |
|  +-------------+    free()---->     |
|  +-------------+                    |
|  | preserved   | (callee-saved)     |
|  +-------------+                    |
|  +-------------+                    |
|  | used        | (tracking)         |
|  +-------------+                    |
+-------------------------------------+
| Vec Registers (0-15 or 0-31)        |
|  +-------------+                    |
|  | free_regs   | ---alloc()-->      |
|  +-------------+                    |
|  +-------------+                    |
|  | in_use      | <--alloc()---      |
|  +-------------+    free()---->     |
|  +-------------+                    |
|  | preserved   | (empty)            |
|  +-------------+                    |
|  +-------------+                    |
|  | used        | (tracking)         |
|  +-------------+                    |
+-------------------------------------+
| Opmask Registers (1-7)              |
|  +-------------+                    |
|  | free_regs   | ---alloc()-->      |
|  +-------------+                    |
|  +-------------+                    |
|  | in_use      | <--alloc()---      |
|  +-------------+    free()---->     |
|  +-------------+                    |
|  | preserved   | (empty)            |
|  +-------------+                    |
|  +-------------+                    |
|  | used        | (tracking)         |
|  +-------------+                    |
+-------------------------------------+
```

### Data Structures and Design Choices

Used `std::set` to track register states as shown above. The manager aims to use STL-based data structures and functions to ensure the helper is as understandable and compatible as possible.

### Register Priority and Management (System V AMD64 ABI)

For information on design choices for registers discussed below, see the [System V AMD64 ABI](https://gitlab.com/x86-psABIs/x86-64-ABI).

#### General Purpose Registers (16 or 32 registers)

**Register Mapping:**
- rax (0), rcx (1), rdx (2), rbx (3), rsp (4), rbp (5), rsi (6), rdi (7)
- r8-r15 (8-15)
- r16-r31 (16-31) - Intel APX only, detected at runtime

**Register Name Helpers:**
Use `Xbyak::Operand` to obtain register indices or construct register objects when allocating registers by name, or pass the numeric index directly (for example, `rm.alloc<Reg64>(10)`). Example:

```cpp
// get numeric index from the Xbyak enum
Reg64 rax = rm.alloc<Reg64>(static_cast<int>(Xbyak::Operand::RAX));
```

This keeps allocation examples consistent with the Xbyak API.

**Calling Convention (automatically selected via `_WIN32` macro):**

*Windows x64:*
- **Volatile (Caller-saved):** rax (0), rcx (1), rdx (2), r8-r11 (8-11) — 7 registers
- **Non-volatile (Callee-saved):** rbx (3), rsi (6), rdi (7), r12-r15 (12-15) — 8 registers
- **Parameter registers:** RCX, RDX, R8, R9 (4 integer parameters)

*System V AMD64 ABI (Linux/Unix/macOS):*
- **Call-Clobbered (Caller-saved):** rax (0), rcx (1), rdx (2), rsi (6), rdi (7), r8-r11 (8-11) — 9 registers
- **Call-Preserved (Callee-saved):** rbx (3), r12-r15 (12-15) — 5 registers
- **Parameter registers:** RDI, RSI, RDX, RCX, R8, R9 (6 integer parameters)

*Extended Registers (APX):*
- r16-r31 (16-31) automatically added as caller-saved when APX is detected

**Special Registers (not auto-allocated):**
- rsp (4) - Stack pointer
- rbp (5) - Base/frame pointer

**Key Difference:** Windows treats RSI/RDI as callee-saved (preserved), while System V treats them as caller-saved (volatile)

**x86-64 Specific Note:**
- Register aliasing (RAX/EAX/AX/AL) is handled by tracking at the index level
- Writing to 32-bit registers (EAX) zero-extends to 64-bit (RAX)
- APX detection uses `Xbyak::util::Cpu::tAPX_F` at construction time

#### Vector Registers (16 or 32 registers)

**Register Mapping:**
- SSE/AVX/AVX2: xmm0-xmm15, ymm0-ymm15 (0-15)
- AVX-512: xmm0-xmm31, ymm0-ymm31, zmm0-zmm31 (0-31)

**Calling Convention (automatically selected via `_WIN32` macro):**

*Windows x64:*
- **Volatile (Caller-saved):** xmm0-xmm5 (0-5) — 6 registers
- **Non-volatile (Callee-saved):** xmm6-xmm15 (6-15) — 10 registers
- **Parameter registers:** xmm0-xmm3 (4 FP parameters)

*System V AMD64 ABI (Linux/Unix/macOS):*
- **Call-Clobbered (Caller-saved):** All vector registers (xmm0-xmm15 or zmm0-zmm31)
- **Call-Preserved:** None
- **Parameter registers:** xmm0-xmm7 (8 FP parameters)

*Extended Registers (AVX-512):*
- xmm16-xmm31/ymm16-ymm31/zmm16-zmm31 (16-31) automatically added as caller-saved when AVX-512 is detected

**Key Difference:** Windows preserves xmm6-xmm15 (major impact for SIMD-heavy code), while System V treats all vector registers as volatile

**x86-64 Specific Note:**
- XMM, YMM, and ZMM registers are aliases of the same physical register
- XMM uses lower 128 bits, YMM uses lower 256 bits, ZMM uses all 512 bits
- Aliasing is automatically handled by index-based tracking
- AVX-512 detection uses `Xbyak::util::Cpu::tAVX512F` at construction time
- Extended registers (16-31) are only available when AVX-512 is detected

#### Opmask Registers (8 registers for AVX-512)

**Register Mapping:**
- k0-k7 (0-7)

**Call-Clobbered (Caller-saved):**
- k1-k7 (1-7)

**Special Register (not auto-allocated):**
- k0 (0) - When used as write mask, means "unmasked" (no masking applied)

### Scoped Registers (RAII)

```cpp
// Scoped register handling with RAII
// Usage: Reg64 reg11 = rm.alloc<Reg64>(11);
//        auto scoped = rm.makeScoped(reg11);
// or
//        auto scoped_reg = rm.makeScoped(rm.alloc<Reg64>());
// reg11 & scoped_reg will free at end of scope when guards' dtors called.

template<class Reg>
class Scoped {
public:
    explicit Scoped(RegPoolManager& rm, Reg r)
        : rm_(&rm), reg_(r) {
        validate_scoped_reg(rm_, reg_);
    }
    
    ~Scoped() {
        if (!rm_) return;
        rm_->free(reg_);
    }
    
    // Disable copy - scoped regs are move only
    Scoped(const Scoped&) = delete;
    Scoped& operator=(const Scoped&) = delete;
    
    // Move constructor
    Scoped(Scoped&& other) noexcept
        : rm_(other.rm_), reg_(other.reg_) {
        other.rm_ = nullptr;
    }
    
    // Expose underlying register
    operator const Reg &() const noexcept { return reg_; }
    const Reg &get() const noexcept { return reg_; }

private:
    RegPoolManager* rm_ = nullptr;
    Reg reg_{};
};
```

**Note:** Scoped registers should not be manually freed, as this will lead to double-free errors and crashes. Users should use `std::move` to change the scoped guard of a register, and therefore its lifetime.

### Special Register Helpers

```cpp
// Helper methods to return special registers
// NOTE: These registers are excluded from automatic allocation

// Stack pointer: rsp (index 4)
inline Reg64 _stack_pointer() {
    used_gp.insert(4);
    return Reg64(4);
}

// Base/frame pointer: rbp (index 5)
inline Reg64 _base_pointer() {
    used_gp.insert(5);
    return Reg64(5);
}

// Opmask k0: special mask that means "unmasked"
inline Opmask _opmask_k0() {
    used_opmask.insert(0);
    return Opmask(0);
}
```

### Extended ISA Support (APX and AVX-512)

The register manager automatically detects Intel APX and AVX-512 support at construction time:

```cpp
RegPoolManager() {
    Xbyak::util::Cpu cpu;
    
    // Detect APX for extended GP registers (r16-r31)
    has_apx_ = cpu.has(Xbyak::util::Cpu::tAPX_F);
    max_gp_reg_idx_ = has_apx_ ? 31 : 15;
    
    // Detect AVX-512 for extended vector registers (zmm16-zmm31)
    has_avx512_ = cpu.has(Xbyak::util::Cpu::tAVX512F);
    max_vec_reg_idx_ = has_avx512_ ? 31 : 15;
    
    // If APX is available, add r16-r31 to the free pool
    if (has_apx_) {
        for (int i = 16; i <= 31; ++i) {
            free_gp_regs.insert(i);
        }
    }
    
    // If AVX-512 is available, add zmm16-zmm31 to the free pool
    if (has_avx512_) {
        for (int i = 16; i <= 31; ++i) {
            free_vec_regs.insert(i);
        }
    }
}
```

Query methods:
```cpp
bool has_apx() const;           // Returns true if APX is supported
int max_gp_registers() const;   // Returns 16 or 32

bool has_avx512() const;        // Returns true if AVX-512 is supported
int max_vec_registers() const;  // Returns 16 or 32
```

**Extended ISA Implementation Notes:**
- r16-r31 (APX) and zmm16-zmm31 (AVX-512) are automatically added as call-clobbered registers
- All range checks use `max_gp_reg_idx_` and `max_vec_reg_idx_` instead of hardcoded values
- Zero overhead: detection happens once at construction
- Backward compatible: works seamlessly on systems without APX or AVX-512

### Error Handling & Strictness

The manager assumes that the user knows what they are doing, in so far that it will throw runtime errors if:

- The user attempts to allocate a specific register when it is in use.
- The user attempts to allocate a specific register that is not in the free or preserved sets.
- The user frees a register that is not in the in-use set.
- The user enters a register index outside the valid range:
  - GP: 0-15 (without APX) or 0-31 (with APX)
  - Vec: 0-16 (without AVX-512) or 0-31 (with AVX-512)
  - Opmask: 0-7
- The user attempts to add a register to the free GP pool when the register is already tracked.
- There are no registers in free_regs or preserved when `alloc()` is called for a register type.
- The user calls `makeScoped` on a register that is not in use.
- The user tries to handle a register type whose `reg_family` is not GP/Vec/Opmask.

### Usage Avoidance

Further, it is important to note that the following should be avoided by users:

**Manual freeing of scoped registers** can cause issues (double frees etc.)

**Avoid alias issues** when handling registers:
```cpp
// BAD - creates aliasing issues
Reg64 reg1 = rm.alloc<Reg64>();
Reg64 reg2 = reg1;  // Both point to same register!
rm.free(reg1);
rm.free(reg2);      // Double free!

// GOOD - use scoped or explicit tracking
auto scoped = rm.makeScoped(rm.alloc<Reg64>());
// or
Reg64 reg1 = rm.alloc<Reg64>();
// ... use reg1 ...
rm.free(reg1);
```

## Using the Manager

```cpp
// Usage:
RegPoolManager rm;

// Allocate next available register
Reg64 r1 = rm.alloc<Reg64>();
Xmm xmm1 = rm.alloc<Xmm>();
Opmask k1 = rm.alloc<Opmask>();

// Free registers
rm.free(r1);
rm.free(xmm1);
rm.free(k1);

// Allocate specific register
Reg64 r10 = rm.alloc<Reg64>(10);

// Scoped registers (RAII)
{
    auto scoped = rm.makeScoped(rm.alloc<Reg64>());
    // Use scoped.get() to access the register
} // Automatically freed here

// Special registers
Reg64 rsp = rm._stack_pointer();
Reg64 rbp = rm._base_pointer();
Opmask k0 = rm._opmask_k0();

// APX support
if (rm.has_apx()) {
    std::cout << "APX supported, " << rm.max_gp_registers() 
              << " GP registers available" << std::endl;
}

// AVX-512 support
if (rm.has_avx512()) {
    std::cout << "AVX-512 supported, " << rm.max_vec_registers() 
              << " vector registers available" << std::endl;
}
```

## Helper Methods

The manager includes minimal helpers to keep it lightweight:

- `reg_in_use(reg)`: Returns true if a register object is in the in-use set
- `gp_idx_in_use(idx)`, `vec_idx_in_use(idx)`, `opmask_idx_in_use(idx)`: Check if an index is in use for a given family
- `makeScoped(reg)`: Creates an RAII guard for automatic deallocation
- `add_to_gp_pool(idx)`: Adds a register to the free GP pool (for special use cases)
- Special register helpers: `_stack_pointer()`, `_base_pointer()`, `_opmask_k0()`
- APX query methods: `has_apx()`, `max_gp_registers()`
- AVX-512 query methods: `has_avx512()`, `max_vec_registers()`
- Getters for in-use, free, preserved, and used sets for each register family

## Implementation Notes for x86-64

### Key Architectural Differences

1. **Register Count Variability**
   - Base x86-64: 16 GP registers, 16 vector registers (SSE/AVX/AVX2)
   - With APX: 32 GP registers (runtime detection required)
   - With AVX-512: 32 vector registers (runtime detection required)
   - AArch64 always has 31 GP registers and 32 vector registers

2. **Register Aliasing**
   - x86-64: RAX/EAX/AX/AL all refer to the same physical register
   - x86-64: XMM/YMM/ZMM all refer to the same physical register
   - Tracking by index automatically handles aliasing
   - 32-bit writes (EAX) zero-extend to 64-bit (RAX)

3. **Calling Convention Differences**

   **Windows x64 vs System V AMD64:**

   | Aspect | Windows x64 | System V AMD64 |
   |--------|-------------|----------------|
   | **GP Volatile** | rax, rcx, rdx, r8-r11 (7) | rax, rcx, rdx, rsi, rdi, r8-r11 (9) |
   | **GP Non-volatile** | rbx, rbp, rsi, rdi, r12-r15 (8) | rbx, rbp, r12-r15 (5) |
   | **Vec Volatile** | xmm0-xmm5 (6) | All xmm/zmm (16 or 32) |
   | **Vec Non-volatile** | xmm6-xmm15 (10) | None |
   | **Integer params** | RCX, RDX, R8, R9 (4) | RDI, RSI, RDX, RCX, R8, R9 (6) |
   | **FP params** | xmm0-xmm3 (4) | xmm0-xmm7 (8) |
   | **Shadow space** | 32 bytes required | Not required |

   **Critical differences:**
   - RSI/RDI: Windows treats as callee-saved, System V as caller-saved
   - Vector preservation: Windows requires saving xmm6-xmm15 (major SIMD impact)
   - Parameter passing: Completely different register sequences
   - Implementation automatically selects correct convention via `_WIN32` macro

4. **Special Registers**
   - x86-64: rsp (stack), rbp (frame), k0 (unmasked)
   - AArch64: x16, x17 (IPC), x18 (platform), x29 (frame), x30 (link)

5. **Extended ISA Support**
   - Intel APX adds 16 GP registers (r16-r31)
   - AVX-512 adds 16 vector registers (zmm16-zmm31)
   - Both require runtime detection via CPUID
   - Automatically integrated when available
   - All extended registers are caller-saved

6. **Vector Register Differences**
   - x86-64 SSE/AVX/AVX2: 16 XMM/YMM registers (xmm0-xmm15, ymm0-ymm15)
   - x86-64 AVX-512: 32 ZMM registers (zmm0-zmm31)
   - Runtime detection determines available count (16 or 32)
   - All are caller-saved in System V ABI
   - AArch64 always has 32 vector registers

7. **Mask Registers**
   - x86-64: 8 opmask registers (k0-k7) for AVX-512
   - k0 is special (unmasked)
   - AArch64: 16 predicate registers (p0-p15) for SVE

## References

- [System V AMD64 ABI](https://gitlab.com/x86-psABIs/x86-64-ABI)
- [Microsoft x64 calling convention](https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention)
- [Overview of x64 Calling Conventions](https://learn.microsoft.com/en-us/cpp/build/x64-software-conventions)
- [Intel APX Specification](https://www.intel.com/content/www/us/en/developer/articles/technical/advanced-performance-extensions-apx.html)
- [Xbyak Documentation](https://github.com/herumi/xbyak)
- [AArch64 Register Manager (PR #4587)](https://github.com/uxlfoundation/oneDNN/pull/4587)
