# AIL (AI Implementation Language)

**A purpose-built programming language and compiler toolchain designed for AI agents — not humans.**

AIL is a systems-level language prototype built on a simple premise: AI tools shouldn't be forced to write code in languages designed for human cognition. Just as we don't ask compilers to optimize for readability, we shouldn't constrain AI code generation to syntax shaped by decades of human ergonomic preferences. AIL is what a programming language looks like when the primary author is a machine.

## Why Does This Exist?

Every mainstream programming language — C, Python, Rust, JavaScript — was designed to be written and read by humans. Their syntax reflects human priorities: familiarity, visual clarity, abbreviation of common patterns, and accommodation of how people think about control flow and state.

AI agents have fundamentally different strengths and weaknesses:

- **They don't need syntactic sugar.** A human appreciates `for item in list` over index arithmetic. An AI doesn't care — it benefits more from a single, unambiguous looping construct.
- **They struggle with implicit behavior.** Implicit type conversions, default arguments, operator overloading, and undefined behavior in C/C++ are major sources of AI-generated bugs. Making semantics explicit eliminates entire categories of errors.
- **They can leverage annotations directly.** A human writes idiomatic loop patterns and hopes the compiler auto-vectorizes. An AI can emit `@vectorize @simd_width(8)` and express hardware intent as a first-class part of the program.
- **They need regularity, not flexibility.** Human languages offer multiple ways to express the same thing (readability, style). AI benefits from one canonical representation — fewer choices means fewer mistakes.

AIL strips away the human-facing affordances and replaces them with a grammar optimized for machine generation: regular, explicit, unambiguous, and directly expressive of low-level performance intent.

## What's In This Repo

This repository contains a complete compiler toolchain prototype, implemented in Python:

| Component | File | Description |
|---|---|---|
| **Language Spec** | `ai_lang_grammar.md` | Formal grammar, type system, and production rules |
| **Lexer & Parser** | `ail_parser.py` | Recursive descent parser producing a full AST |
| **Semantic Analyzer** | `ail_semantic_analyzer.py` | Type checking, scope resolution, and validation |
| **LLVM Code Generator** | `ail_llvm_codegen.py` | LLVM IR generation via llvmlite with CPU and GPU targeting |
| **Optimization Passes** | `ail_optimization_passes.py` | Custom optimization passes driven by AIL annotations |
| **Complete System** | `ail_complete_system.py` | Unified pipeline from source to compiled output |
| **Documentation** | `ail_language_documentation.md` | Full language reference |
| **Setup** | `ail_setup_script.sh` | Environment setup and dependency installation |

## Language Design

### Explicit Memory Model

Every allocation declares its storage strategy. No hidden heap allocations, no garbage collector surprises — the AI knows exactly what it's asking for:

```ail
stack f32[16] small_buffer = {0};        // Stack-allocated, automatic cleanup
heap f32* large_data = alloc(null, n);   // Heap-allocated, manual management
arena WorkItem* task = alloc(pool, sz);  // Arena-allocated, bulk cleanup
```

### Performance Intent as Syntax

Instead of relying on compiler heuristics, AIL lets the code author (the AI) express optimization intent directly through annotations:

```ail
@vectorize @simd_width(8) @hot
func matrix_multiply(const f32* a, const f32* b, heap f32* result,
                     i32 rows, i32 cols, i32 inner_dim) -> void {
    @unroll(4)
    for (i32 i in 0..rows) {
        for (i32 j in 0..cols) {
            f32 sum = 0.0f;

            @vectorize
            for (i32 k in 0..inner_dim step 4) {
                vec4 a_vec = {a[i*inner_dim + k], a[i*inner_dim + k+1],
                             a[i*inner_dim + k+2], a[i*inner_dim + k+3]};
                vec4 b_vec = {b[k*cols + j], b[(k+1)*cols + j],
                             b[(k+2)*cols + j], b[(k+3)*cols + j]};
                sum += dot(a_vec, b_vec);
            }

            result[i*cols + j] = sum;
        }
    }
}
```

### First-Class GPU Support

GPU compute is not bolted on as a separate dialect (like CUDA's C++ extensions). Kernels, shared memory, and thread indexing are part of the language:

```ail
@kernel @block_size(16, 16, 1)
func matmul_kernel(global<f32>* a, global<f32>* b, global<f32>* result,
                   i32 rows_a, i32 cols_a, i32 cols_b) -> void {
    @shared @bank_conflict_free
    shared<f32>[16][16] tile_a;

    i32 tx = thread_idx_x();
    i32 ty = thread_idx_y();
    // ...
}
```

### Built-in Numerical Types

Vector and matrix types are primitives, not library additions:

```ail
vec4 position = {1.0f, 2.0f, 3.0f, 1.0f};
mat4 transform = identity();
f32 d = dot(position.xyz, normal);
vec3 reflected = reflect(incoming, surface_normal);
```

## Technical Depth

Building AIL required working across multiple layers of the compilation stack:

- **Language design** — Defining a formal grammar (documented in BNF-style production rules) that is both expressive enough for real workloads and regular enough for reliable machine generation.
- **Lexical analysis & parsing** — A hand-written recursive descent parser with full operator precedence, annotation handling, and error recovery. No parser generators — the grammar-to-code mapping is direct and transparent.
- **Semantic analysis** — Multi-pass type checking with storage qualifier validation, scope management, function signature resolution, and GPU-specific constraint checking (e.g., shared memory can only appear inside kernel functions).
- **LLVM IR generation** — Translating the AST to LLVM IR via llvmlite, including mapping AIL's annotation system to LLVM metadata, intrinsics, and optimization hints. Supports both CPU (with OpenMP-style parallel loop lowering) and GPU (CUDA/OpenCL kernel emission) targets.
- **Custom optimization** — Annotation-driven optimization passes that operate on the AST before LLVM lowering, handling vectorization strategy selection, loop transformation, and memory layout optimization.

## Design Principles

| Principle | What It Means |
|---|---|
| **Regularity over convenience** | One way to do things. No syntactic shortcuts that create ambiguity. |
| **Explicit over implicit** | Storage, alignment, parallelism, and optimization strategy are all visible in the source. |
| **Annotations over idioms** | Performance intent is stated, not inferred from code patterns. |
| **Composable primitives** | Small, orthogonal features that combine predictably. |
| **Zero undefined behavior** | Every construct has defined semantics. No "implementation-defined" escape hatches. |

## Getting Started

```bash
# Install dependencies
./ail_setup_script.sh

# Or manually
pip install llvmlite

# Run the complete system
python ail_complete_system.py
```

## Status

This is a language prototype and research project demonstrating the concept. The compiler toolchain is functional for the core language subset and produces working LLVM IR for both CPU and GPU targets.

## Author

Designed and built by **Sean Southard**.
