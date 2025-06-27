# AIL Language Documentation
**AI Implementation Language - A Language Designed for AI Agents**

Version 1.0 | Date: 2024

---

## Table of Contents

1. [Introduction](#introduction)
2. [Language Philosophy](#language-philosophy)
3. [Lexical Structure](#lexical-structure)
4. [Type System](#type-system)
5. [Variables and Storage](#variables-and-storage)
6. [Functions](#functions)
7. [Control Flow](#control-flow)
8. [Parallel Programming](#parallel-programming)
9. [GPU Programming](#gpu-programming)
10. [Built-in Functions](#built-in-functions)
11. [Annotations](#annotations)
12. [Memory Management](#memory-management)
13. [Examples](#examples)
14. [Best Practices](#best-practices)
15. [Error Handling](#error-handling)
16. [Language Reference](#language-reference)

---

## Introduction

AIL (AI Implementation Language) is a systems programming language specifically designed for AI agents to generate high-performance parallel and GPU code. It combines the explicitness of C with modern parallel programming constructs and AI-friendly syntax patterns.

### Key Features

- **Explicit parallelism** with `@parallel_for` constructs
- **GPU programming** with kernel functions and memory hierarchies
- **Vector and matrix types** as first-class citizens
- **Memory safety** through storage qualifiers
- **AI-optimized syntax** with predictable patterns
- **Performance annotations** for optimization hints

### Design Goals

1. **Predictability**: Every construct has clear, consistent semantics
2. **Explicitness**: No hidden behavior or implicit operations
3. **Composability**: Language features combine in obvious ways
4. **Performance**: Direct mapping to efficient machine code
5. **AI-Friendly**: Easy for AI agents to understand and generate

---

## Language Philosophy

### For AI Agents

AIL is designed around the principle that AI agents excel at pattern recognition and systematic code generation. Every language feature follows consistent patterns:

- **Regular syntax**: Similar constructs use similar syntax
- **Explicit semantics**: All behavior is visible in the code
- **Predictable performance**: Code performance is evident from syntax
- **Composable abstractions**: Complex operations built from simple parts

### Memory Model

AIL uses an explicit memory model where storage location and lifetime are part of the type system:

```ail
stack i32 local_var;     // Stack-allocated, automatic cleanup
heap f32* large_array;   // Heap-allocated, manual management
global<f32>* gpu_data;   // GPU global memory
shared<f32>[32] cache;   // GPU shared memory
```

### Parallelism Model

Parallelism is explicit and structured:

```ail
@parallel_for (i32 i in 0..count) {
    // Parallel iteration - no hidden synchronization
    output[i] = input[i] * 2.0f;
}
```

---

## Lexical Structure

### Comments

```ail
// Single-line comment

/* 
   Multi-line comment
   Can span multiple lines
*/
```

### Identifiers

Identifiers follow C-style naming:
- Start with letter or underscore
- Contain letters, digits, underscores
- Case-sensitive

```ail
valid_identifier
_private_var
camelCase
snake_case
PascalCase
```

### Keywords

#### Type Keywords
```ail
i8 i16 i32 i64        // Signed integers
u8 u16 u32 u64        // Unsigned integers  
f32 f64               // Floating point
bool void             // Boolean and void
vec2 vec3 vec4        // Vector types
mat2 mat3 mat4        // Matrix types
auto                  // Type inference
```

#### Storage Qualifiers
```ail
stack heap arena      // CPU memory
global shared local   // GPU memory
constant texture      // GPU special memory
managed               // Unified memory
const volatile        // Mutability
atomic                // Atomic operations
```

#### Control Flow
```ail
if else while for loop
break continue return
switch case default
```

#### Function Keywords
```ail
func extern inline
kernel device host host_device
```

#### Parallel Keywords
```ail
parallel_for launch sync
sync_threads sync_warp
critical reduce in step
```

### Literals

#### Integer Literals
```ail
42          // i32 (default)
42i64       // Explicit i64
42u32       // Explicit u32
0xFF        // Hexadecimal
0b1010      // Binary
0o777       // Octal
```

#### Floating Point Literals
```ail
3.14f       // f32
3.14        // f64 (default)
1e-6f       // Scientific notation
.5f         // Leading decimal point
2.f         // Trailing decimal point
```

#### Boolean Literals
```ail
true
false
```

#### String Literals
```ail
"hello world"           // Basic string
"line 1\nline 2"       // With escape sequences
"path\\to\\file"       // Escaped backslashes
```

#### Vector Literals
```ail
{1.0f, 2.0f, 3.0f}         // vec3
{1.0f, 0.0f, 0.0f, 1.0f}   // vec4
{1, 2}                     // ivec2 (inferred from context)
```

### Operators

#### Arithmetic Operators
```ail
+  -  *  /  %           // Basic arithmetic
+= -= *= /= %=          // Compound assignment
```

#### Comparison Operators
```ail
== != < <= > >=         // Comparison
```

#### Logical Operators
```ail
&& || !                 // Logical
```

#### Bitwise Operators
```ail
& | ^ ~ << >>           // Bitwise
&= |= ^= <<= >>=        // Compound bitwise
```

#### Other Operators
```ail
= -> . .. [] () {}      // Assignment, access, grouping
@ # $                   // Special (annotations, etc.)
```

---

## Type System

### Primitive Types

#### Integer Types
```ail
i8   // 8-bit signed integer   (-128 to 127)
i16  // 16-bit signed integer  (-32,768 to 32,767)
i32  // 32-bit signed integer  (-2^31 to 2^31-1)
i64  // 64-bit signed integer  (-2^63 to 2^63-1)

u8   // 8-bit unsigned integer  (0 to 255)
u16  // 16-bit unsigned integer (0 to 65,535)
u32  // 32-bit unsigned integer (0 to 2^32-1)
u64  // 64-bit unsigned integer (0 to 2^64-1)
```

#### Floating Point Types
```ail
f32  // 32-bit IEEE 754 float
f64  // 64-bit IEEE 754 double
```

#### Boolean Type
```ail
bool // true or false
```

#### Void Type
```ail
void // No value (function returns only)
```

### Composite Types

#### Vector Types
```ail
vec2  // 2-component f32 vector
vec3  // 3-component f32 vector  
vec4  // 4-component f32 vector

ivec2 ivec3 ivec4  // Integer vectors
uvec2 uvec3 uvec4  // Unsigned integer vectors
```

Vector components can be accessed via swizzling:
```ail
vec3 v = {1.0f, 2.0f, 3.0f};
f32 x = v.x;    // First component
f32 y = v.y;    // Second component  
f32 z = v.z;    // Third component

// Alternative names
f32 r = v.r;    // Red (same as x)
f32 g = v.g;    // Green (same as y)
f32 b = v.b;    // Blue (same as z)
```

#### Matrix Types
```ail
mat2  // 2x2 f32 matrix (4 elements)
mat3  // 3x3 f32 matrix (9 elements)
mat4  // 4x4 f32 matrix (16 elements)
```

Matrices are stored in column-major order (OpenGL style):
```ail
mat2 m = {
    1.0f, 2.0f,  // First column
    3.0f, 4.0f   // Second column
};
```

#### Pointer Types
```ail
i32*        // Pointer to i32
const f32*  // Pointer to const f32
f32* const  // Const pointer to f32
```

#### Array Types
```ail
i32[10]     // Fixed-size array of 10 i32s
f32[]       // Flexible array (last struct member only)
```

### Template Types

Template types are used for GPU memory hierarchies:

```ail
global<f32>*     // Pointer to GPU global memory
shared<i32>[32]  // GPU shared memory array
local<f32>       // GPU thread-local memory
constant<f32>[4] // GPU constant memory
texture<f32>     // GPU texture memory
managed<f32>*    // CPU-GPU unified memory
atomic<i32>      // Atomic integer
```

### Type Inference

The `auto` keyword enables type inference:

```ail
auto x = 42;        // Inferred as i32
auto y = 3.14f;     // Inferred as f32
auto z = {1, 2, 3}; // Inferred as vec3 (from context)
```

---

## Variables and Storage

### Storage Qualifiers

Storage qualifiers specify where and how variables are stored:

#### CPU Storage
```ail
stack i32 local_var;        // Stack storage (automatic)
heap f32* dynamic_array;    // Heap storage (manual management)
arena char* temp_string;    // Arena storage (bulk cleanup)
```

#### GPU Storage
```ail
global<f32>* device_array;  // GPU global memory (large, slow)
shared<f32>[256] cache;     // GPU shared memory (small, fast)  
local<f32> thread_var;      // GPU thread-private memory
constant<f32>[16] params;   // GPU constant memory (read-only)
texture<f32> image_data;    // GPU texture memory (cached)
```

#### Special Qualifiers
```ail
const i32 CONSTANT = 42;    // Immutable after initialization
volatile i32 hardware_reg;  // Prevents compiler optimization
atomic<i32> counter;        // Atomic operations only
```

### Variable Declaration

```ail
// Basic declaration
i32 variable_name;

// With storage qualifier
stack f32 local_data;

// With initialization
heap i32* array = alloc(null, sizeof(i32) * 100);

// Multiple variables
i32 a, b, c;

// Const variables (must be initialized)
const f32 PI = 3.14159f;
```

### Scope Rules

AIL uses block scope similar to C:

```ail
func example() -> void {
    i32 outer = 1;
    
    if (true) {
        i32 inner = 2;  // Only visible in this block
        outer = 3;      // Can access outer scope
    }
    
    // inner is not visible here
    // outer is still visible
}
```

---

## Functions

### Function Declaration

```ail
// Basic function
func function_name(param1_type param1_name, param2_type param2_name) -> return_type {
    // Function body
    return value;
}

// Void function (no return value)
func print_message(const char* message) -> void {
    // Implementation
}

// Function with no parameters
func get_random() -> i32 {
    return 42;
}
```

### Function Modifiers

```ail
// External linkage
extern func external_function(i32 x) -> f32;

// Force inline
inline func small_function(i32 x) -> i32 {
    return x * 2;
}

// GPU kernel function
kernel func gpu_kernel(global<f32>* data) -> void {
    // GPU code
}

// Device function (called from kernels)
device func helper_function(f32 x) -> f32 {
    return x * x;
}

// Host function (CPU only)
host func cpu_only_function() -> void {
    // CPU code
}

// Host-device function (both CPU and GPU)
host_device func math_function(f32 x) -> f32 {
    return x * x + 1.0f;
}
```

### Parameter Passing

Parameters can have storage qualifiers:

```ail
func process_data(
    const f32* input,       // Read-only pointer
    heap f32* output,       // Writable heap pointer
    i32 size               // Value parameter
) -> void {
    for (i32 i = 0; i < size; i++) {
        output[i] = input[i] * 2.0f;
    }
}
```

### Function Overloading

AIL supports function overloading based on parameter types:

```ail
func add(i32 a, i32 b) -> i32 {
    return a + b;
}

func add(f32 a, f32 b) -> f32 {
    return a + b;
}

func add(vec3 a, vec3 b) -> vec3 {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
```

---

## Control Flow

### Conditional Statements

#### If Statement
```ail
if (condition) {
    // Code when condition is true
}

if (condition) {
    // True branch
} else {
    // False branch  
}

if (condition1) {
    // First condition
} else if (condition2) {
    // Second condition
} else {
    // Default case
}
```

#### Switch Statement (Planned)
```ail
switch (value) {
    case 1:
        // Handle 1
        break;
    case 2:
    case 3:
        // Handle 2 or 3
        break;
    default:
        // Default case
        break;
}
```

### Loops

#### While Loop
```ail
while (condition) {
    // Loop body
    // Condition checked before each iteration
}
```

#### For Loop (C-style)
```ail
for (i32 i = 0; i < 10; i++) {
    // Loop body
}

// With declaration
for (i32 i = 0; i < count; i++) {
    array[i] = i * i;
}
```

#### Range-based For Loop
```ail
// Simple range
for (i32 i in 0..10) {
    // i goes from 0 to 9
}

// With step
for (i32 i in 0..100 step 5) {
    // i goes 0, 5, 10, 15, ..., 95
}

// With variables
i32 start = 10;
i32 end = 20;
for (i32 i in start..end) {
    // i goes from 10 to 19
}
```

#### Infinite Loop
```ail
loop {
    // Infinite loop
    if (should_exit) {
        break;
    }
}
```

### Loop Control

```ail
for (i32 i = 0; i < 100; i++) {
    if (i < 10) {
        continue;  // Skip to next iteration
    }
    
    if (i > 50) {
        break;     // Exit loop
    }
    
    // Process i
}
```

### Return Statement

```ail
func early_return(i32 value) -> i32 {
    if (value < 0) {
        return -1;  // Early return
    }
    
    // Normal processing
    return value * 2;
}

func void_return() -> void {
    if (some_condition) {
        return;  // Early return from void function
    }
    
    // More code
}
```

---

## Parallel Programming

### Parallel For Loops

The `@parallel_for` construct enables data parallelism:

```ail
@parallel_for (i32 i in 0..count) {
    // Each iteration can run in parallel
    output[i] = input[i] * 2.0f;
}
```

#### With Step
```ail
@parallel_for (i32 i in 0..count step 4) {
    // Process 4 elements at a time
    vec4 data = load_vec4(&input[i]);
    vec4 result = data * 2.0f;
    store_vec4(&output[i], result);
}
```

### Reductions

Reductions safely combine values from parallel iterations:

```ail
f32 total = 0.0f;

@parallel_for (i32 i in 0..count) 
reduce(total: add) {
    total += input[i];  // Safe parallel accumulation
}

// total now contains the sum of all input[i]
```

#### Multiple Reductions
```ail
f32 sum = 0.0f;
f32 max_val = -1e9f;

@parallel_for (i32 i in 0..count) 
reduce(sum: add, max_val: max) {
    sum += input[i];
    max_val = max(max_val, input[i]);
}
```

#### Reduction Operations
- `add`: Addition (`+`)
- `mul`: Multiplication (`*`)  
- `min`: Minimum value
- `max`: Maximum value
- `and`: Logical AND
- `or`: Logical OR

### Synchronization

#### Thread Synchronization
```ail
// CPU thread barrier
sync;

// GPU block synchronization  
sync_threads;

// GPU warp synchronization
sync_warp;
```

#### Critical Sections
```ail
i32 global_counter = 0;

@parallel_for (i32 i in 0..count) {
    // Parallel work
    process_item(i);
    
    // Atomic update
    critical {
        global_counter++;
    }
}
```

#### Atomic Operations
```ail
atomic<i32> counter;

@parallel_for (i32 i in 0..count) {
    // Atomic increment
    atomic_add(&counter, 1);
    
    // Atomic compare-and-swap
    i32 expected = 10;
    atomic_compare_exchange(&counter, &expected, 20);
}
```

---

## GPU Programming

### Kernel Functions

GPU kernels are functions that run on the GPU:

```ail
@kernel @block_size(32, 32, 1)
func matrix_add(
    global<f32>* a,
    global<f32>* b, 
    global<f32>* result,
    i32 width,
    i32 height
) -> void {
    // Get thread indices
    i32 x = thread_idx_x() + block_idx_x() * 32;
    i32 y = thread_idx_y() + block_idx_y() * 32;
    
    // Check bounds
    if (x < width && y < height) {
        i32 idx = y * width + x;
        result[idx] = a[idx] + b[idx];
    }
}
```

### Thread Indexing

GPU threads are organized in a 3D hierarchy:

```ail
// Thread index within block
i32 tx = thread_idx_x();
i32 ty = thread_idx_y(); 
i32 tz = thread_idx_z();

// Block index within grid
i32 bx = block_idx_x();
i32 by = block_idx_y();
i32 bz = block_idx_z();

// Block dimensions
i32 block_x = block_dim_x();
i32 block_y = block_dim_y();
i32 block_z = block_dim_z();

// Grid dimensions
i32 grid_x = grid_dim_x();
i32 grid_y = grid_dim_y();
i32 grid_z = grid_dim_z();

// Global thread index
i32 global_x = bx * block_x + tx;
i32 global_y = by * block_y + ty;
```

### Memory Hierarchies

#### Global Memory
```ail
// Large, high-latency memory accessible by all threads
global<f32>* device_array;

// Coalesced access for best performance
@coalesced
for (i32 i = 0; i < size; i++) {
    device_array[i] = compute_value(i);
}
```

#### Shared Memory
```ail
// Fast memory shared by threads in a block
@shared @bank_conflict_free
shared<f32>[32][32] tile;

// Load data into shared memory
tile[ty][tx] = global_data[global_idx];

// Synchronize before using shared data
sync_threads;

// Compute using shared data
f32 result = 0.0f;
for (i32 i = 0; i < 32; i++) {
    result += tile[ty][i] * tile[i][tx];
}
```

#### Constant Memory
```ail
// Read-only memory with broadcast cache
constant<f32>[16] filter_coefficients = {
    0.1f, 0.2f, 0.3f, 0.4f,
    0.5f, 0.6f, 0.7f, 0.8f,
    0.9f, 1.0f, 1.1f, 1.2f,
    1.3f, 1.4f, 1.5f, 1.6f
};

@kernel
func apply_filter(global<f32>* data, i32 size) -> void {
    i32 idx = global_thread_idx();
    if (idx < size) {
        f32 result = 0.0f;
        for (i32 i = 0; i < 16; i++) {
            result += data[idx] * filter_coefficients[i];
        }
        data[idx] = result;
    }
}
```

### Warp Operations

Warps are groups of 32 threads that execute in lockstep:

```ail
@kernel
func warp_reduction(global<f32>* input, global<f32>* output, i32 size) -> void {
    i32 idx = global_thread_idx();
    f32 value = (idx < size) ? input[idx] : 0.0f;
    
    // Warp-level reduction
    value = warp_reduce_add(value);
    
    // First thread in warp writes result
    if (lane_id() == 0) {
        output[warp_id()] = value;
    }
}
```

#### Warp Functions
```ail
// Warp information
i32 warp_id();          // Warp ID within block
i32 lane_id();          // Thread ID within warp (0-31)
i32 warp_size();        // Warp size (typically 32)

// Warp reductions
f32 warp_reduce_add(f32 value);
f32 warp_reduce_min(f32 value);
f32 warp_reduce_max(f32 value);

// Warp communication
f32 warp_shuffle(f32 value, i32 src_lane);
f32 warp_shuffle_up(f32 value, i32 delta);
f32 warp_shuffle_down(f32 value, i32 delta);

// Warp voting
bool warp_all(bool predicate);   // All threads true?
bool warp_any(bool predicate);   // Any thread true?
u32 warp_ballot(bool predicate); // Ballot of all threads
```

### Kernel Launch

Kernels are launched from host code:

```ail
func main() -> i32 {
    // Allocate GPU memory
    global<f32>* device_a = gpu_alloc(sizeof(f32) * size);
    global<f32>* device_b = gpu_alloc(sizeof(f32) * size);
    global<f32>* device_result = gpu_alloc(sizeof(f32) * size);
    
    // Copy data to GPU
    memcpy_host_to_device(host_a, device_a, sizeof(f32) * size);
    memcpy_host_to_device(host_b, device_b, sizeof(f32) * size);
    
    // Launch kernel
    i32 threads_per_block = 256;
    i32 blocks = (size + threads_per_block - 1) / threads_per_block;
    
    launch<{blocks, 1, 1}, {threads_per_block, 1, 1}> 
        vector_add(device_a, device_b, device_result, size);
    
    // Copy result back
    memcpy_device_to_host(device_result, host_result, sizeof(f32) * size);
    
    // Free GPU memory
    gpu_free(device_a);
    gpu_free(device_b);
    gpu_free(device_result);
    
    return 0;
}
```

---

## Built-in Functions

### Mathematical Functions

#### Scalar Math
```ail
// Basic math
f32 abs(f32 x);           // Absolute value
f32 min(f32 a, f32 b);    // Minimum
f32 max(f32 a, f32 b);    // Maximum
f32 clamp(f32 x, f32 min, f32 max);  // Clamp to range

// Trigonometric
f32 sin(f32 x);           // Sine
f32 cos(f32 x);           // Cosine  
f32 tan(f32 x);           // Tangent
f32 asin(f32 x);          // Arc sine
f32 acos(f32 x);          // Arc cosine
f32 atan(f32 x);          // Arc tangent
f32 atan2(f32 y, f32 x);  // Two-argument arc tangent

// Exponential and logarithmic
f32 exp(f32 x);           // e^x
f32 exp2(f32 x);          // 2^x
f32 log(f32 x);           // Natural logarithm
f32 log2(f32 x);          // Base-2 logarithm
f32 log10(f32 x);         // Base-10 logarithm
f32 pow(f32 x, f32 y);    // x^y
f32 sqrt(f32 x);          // Square root
f32 rsqrt(f32 x);         // Reciprocal square root

// Rounding
f32 floor(f32 x);         // Round down
f32 ceil(f32 x);          // Round up
f32 round(f32 x);         // Round to nearest
f32 trunc(f32 x);         // Truncate to integer
f32 fract(f32 x);         // Fractional part

// Utility
f32 lerp(f32 a, f32 b, f32 t);        // Linear interpolation
f32 smoothstep(f32 edge0, f32 edge1, f32 x);  // Smooth interpolation
f32 step(f32 edge, f32 x);            // Step function
```

### Vector Functions

```ail
// Vector creation
vec2 make_vec2(f32 x, f32 y);
vec3 make_vec3(f32 x, f32 y, f32 z);
vec4 make_vec4(f32 x, f32 y, f32 z, f32 w);

// Vector operations
f32 dot(vec3 a, vec3 b);              // Dot product
vec3 cross(vec3 a, vec3 b);           // Cross product  
f32 length(vec3 v);                   // Vector magnitude
f32 length_squared(vec3 v);           // Squared magnitude (faster)
vec3 normalize(vec3 v);               // Unit vector
f32 distance(vec3 a, vec3 b);         // Distance between points
vec3 reflect(vec3 incident, vec3 normal);    // Reflection
vec3 refract(vec3 incident, vec3 normal, f32 eta);  // Refraction

// Component-wise operations
vec3 min(vec3 a, vec3 b);             // Component-wise minimum
vec3 max(vec3 a, vec3 b);             // Component-wise maximum
vec3 clamp(vec3 v, vec3 min, vec3 max);  // Component-wise clamp
vec3 lerp(vec3 a, vec3 b, f32 t);     // Component-wise lerp
vec3 smoothstep(vec3 edge0, vec3 edge1, vec3 x);  // Component-wise smoothstep
```

### Matrix Functions

```ail
// Matrix creation
mat4 identity();                      // Identity matrix
mat4 translation(vec3 offset);        // Translation matrix
mat4 rotation_x(f32 angle);          // Rotation around X axis
mat4 rotation_y(f32 angle);          // Rotation around Y axis  
mat4 rotation_z(f32 angle);          // Rotation around Z axis
mat4 rotation(vec3 axis, f32 angle); // Rotation around arbitrary axis
mat4 scale(vec3 factors);            // Scale matrix
mat4 perspective(f32 fov, f32 aspect, f32 near, f32 far);  // Perspective projection
mat4 orthographic(f32 left, f32 right, f32 bottom, f32 top, f32 near, f32 far);  // Orthographic projection

// Matrix operations
mat4 transpose(mat4 m);              // Matrix transpose
mat4 inverse(mat4 m);                // Matrix inverse
f32 determinant(mat4 m);             // Matrix determinant
mat4 multiply(mat4 a, mat4 b);       // Matrix multiplication
vec4 transform(mat4 m, vec4 v);      // Transform vector by matrix
```

### Memory Functions

```ail
// Memory allocation
void* alloc(arena* a, u64 size);                    // Allocate memory
void* alloc_aligned(arena* a, u64 size, u64 align); // Aligned allocation
void free(void* ptr);                                // Free memory
void* realloc(void* ptr, u64 new_size);            // Reallocate memory

// Memory operations
void memcpy(void* dest, const void* src, u64 size); // Copy memory
void memset(void* ptr, i32 value, u64 size);       // Set memory
i32 memcmp(const void* a, const void* b, u64 size); // Compare memory

// GPU memory
global<void>* gpu_alloc(u64 size);                 // Allocate GPU memory
void gpu_free(global<void>* ptr);                  // Free GPU memory
void memcpy_host_to_device(const void* host, global<void>* device, u64 size);
void memcpy_device_to_host(const global<void>* device, void* host, u64 size);
void memcpy_device_to_device(const global<void>* src, global<void>* dest, u64 size);

// Memory queries
u64 sizeof(type);                       // Size of type
u64 alignof(type);                      // Alignment of type
```

### Thread and Synchronization Functions

```ail
// CPU threading
i32 get_thread_id();                    // Current thread ID
i32 get_thread_count();                 // Total thread count
void thread_barrier();                  // Thread barrier

// GPU threading
i32 thread_idx_x();                     // Thread index X
i32 thread_idx_y();                     // Thread index Y
i32 thread_idx_z();                     // Thread index Z
i32 block_idx_x();                      // Block index X
i32 block_idx_y();                      // Block index Y
i32 block_idx_z();                      // Block index Z
i32 block_dim_x();                      // Block dimension X
i32 block_dim_y();                      // Block dimension Y
i32 block_dim_z();                      // Block dimension Z
i32 grid_dim_x();                       // Grid dimension X
i32 grid_dim_y();                       // Grid dimension Y
i32 grid_dim_z();                       // Grid dimension Z

// Convenience functions
i32 global_thread_idx();                // 1D global thread index
(i32, i32) global_thread_idx_2d();      // 2D global thread index
(i32, i32, i32) global_thread_idx_3d(); // 3D global thread index
```

---

## Annotations

Annotations provide optimization hints and control code generation:

### Performance Annotations

```ail
@vectorize                  // Enable SIMD vectorization
@unroll(4)                 // Unroll loop 4 times
@simd_width(8)             // Use 8-wide SIMD instructions
@inline                    // Force function inlining
@no_inline                 // Prevent function inlining
@pure                      // Function has no side effects
@hot                       // Frequently executed code
@cold                      // Rarely executed code
```

### Memory Annotations

```ail
@memory_aligned(64)        // Align to 64-byte boundary
@cache_friendly            // Optimize for cache locality
@no_alias                  // Pointer doesn't alias others
@coalesced                 // GPU memory access is coalesced
@bank_conflict_free        // No shared memory bank conflicts
@prefetch_friendly         // Layout optimized for prefetching
```

### Target Annotations

```ail
@target("cpu")             // CPU-only code
@target("gpu")             // GPU-only code
@target("avx2")            // Requires AVX2 instruction set
@target("cuda")            // CUDA-specific code
```

### GPU Annotations

```ail
@kernel                    // GPU kernel function
@device                    // GPU device function
@host                      // CPU host function
@host_device               // Both host and device

@block_size(32, 32, 1)     // GPU block dimensions
@grid_size(64, 64, 1)      // GPU grid dimensions
@occupancy(75)             // Target occupancy percentage
@warp_size(32)             // Warp size hint
@shared_memory(1024)       // Shared memory size
```

### Parallel Annotations

```ail
@parallel                  // Enable parallelization
@parallel_for              // Parallel for loop
@reduction(add)            // Reduction operation
@atomic                    // Atomic operations required
@thread_safe               // Thread-safe function
@lock_free                 // Lock-free implementation
```

---

## Memory Management

### Manual Memory Management

```ail
func manual_example() -> void {
    // Allocate heap memory
    heap f32* data = alloc(null, sizeof(f32) * 1000);
    
    // Use the memory
    for (i32 i = 0; i < 1000; i++) {
        data[i] = i * 2.0f;
    }
    
    // Free when done
    free(data);
}
```

### Arena Memory Management

```ail
func arena_example() -> void {
    // Create arena
    arena* temp_arena = create_arena(1024 * 1024);  // 1MB
    
    // Allocate from arena (no individual free needed)
    arena f32* temp_data1 = alloc(temp_arena, sizeof(f32) * 100);
    arena i32* temp_data2 = alloc(temp_arena, sizeof(i32) * 200);
    arena char* temp_string = alloc(temp_arena, 256);
    
    // Use the memory...
    
    // Free entire arena at once
    destroy_arena(temp_arena);
}
```

### Stack Memory Management

```ail
func stack_example() -> void {
    // Stack allocation (automatic cleanup)
    stack f32[100] local_array;
    stack i32 local_var = 42;
    
    // Memory automatically freed when function returns
}
```

### GPU Memory Management

```ail
func gpu_memory_example() -> void {
    const i32 size = 1024;
    
    // Allocate GPU memory
    global<f32>* device_data = gpu_alloc(sizeof(f32) * size);
    
    // Allocate CPU memory
    heap f32* host_data = alloc(null, sizeof(f32) * size);
    
    // Initialize CPU data
    for (i32 i = 0; i < size; i++) {
        host_data[i] = i * 0.5f;
    }
    
    // Copy to GPU
    memcpy_host_to_device(host_data, device_data, sizeof(f32) * size);
    
    // Launch kernel to process data
    launch<{(size + 255) / 256, 1, 1}, {256, 1, 1}> 
        process_kernel(device_data, size);
    
    // Copy result back
    memcpy_device_to_host(device_data, host_data, sizeof(f32) * size);
    
    // Clean up
    free(host_data);
    gpu_free(device_data);
}
```

### Memory Safety

AIL provides several mechanisms for memory safety:

#### Const Correctness
```ail
func safe_function(const f32* readonly_data, f32* writable_data, i32 size) -> void {
    // readonly_data cannot be modified
    // writable_data can be modified
    
    for (i32 i = 0; i < size; i++) {
        writable_data[i] = readonly_data[i] * 2.0f;  // OK
        // readonly_data[i] = 0.0f;  // Compiler error
    }
}
```

#### Storage Qualifier Checking
```ail
func storage_safety() -> void {
    stack f32[100] stack_array;
    heap f32* heap_array = alloc(null, sizeof(f32) * 100);
    
    // Compiler tracks storage types
    process_stack_data(stack_array);    // OK
    process_heap_data(heap_array);      // OK
    // process_stack_data(heap_array);  // Type error
    
    free(heap_array);  // Required for heap memory
    // free(stack_array);  // Error - stack memory auto-freed
}
```

---

## Examples

### Vector Addition

```ail
@parallel @vectorize
func vector_add(const f32* a, const f32* b, heap f32* result, i32 count) -> void {
    @parallel_for (i32 i in 0..count) {
        result[i] = a[i] + b[i];
    }
}

func main() -> i32 {
    const i32 size = 1000000;
    
    // Allocate memory
    heap f32* a = alloc(null, sizeof(f32) * size);
    heap f32* b = alloc(null, sizeof(f32) * size);
    heap f32* result = alloc(null, sizeof(f32) * size);
    
    // Initialize data
    for (i32 i = 0; i < size; i++) {
        a[i] = i * 0.5f;
        b[i] = i * 0.25f;
    }
    
    // Perform vector addition
    vector_add(a, b, result, size);
    
    // Clean up
    free(a);
    free(b);
    free(result);
    
    return 0;
}
```

### Matrix Multiplication

```ail
@parallel @hot
func matrix_multiply(
    const f32* a, const f32* b, heap f32* result,
    i32 rows_a, i32 cols_a, i32 cols_b
) -> void {
    @parallel_for (i32 i in 0..rows_a) {
        for (i32 j = 0; j < cols_b; j++) {
            f32 sum = 0.0f;
            
            @vectorize @unroll(4)
            for (i32 k = 0; k < cols_a; k++) {
                sum += a[i * cols_a + k] * b[k * cols_b + j];
            }
            
            result[i * cols_b + j] = sum;
        }
    }
}
```

### GPU Kernel

```ail
@kernel @block_size(16, 16, 1)
func gpu_matrix_add(
    global<f32>* a,
    global<f32>* b,
    global<f32>* result,
    i32 width,
    i32 height
) -> void {
    // Get thread position
    i32 x = thread_idx_x() + block_idx_x() * 16;
    i32 y = thread_idx_y() + block_idx_y() * 16;
    
    // Check bounds
    if (x < width && y < height) {
        i32 idx = y * width + x;
        result[idx] = a[idx] + b[idx];
    }
}

func gpu_example() -> void {
    const i32 width = 1024;
    const i32 height = 1024;
    const i32 size = width * height;
    
    // Allocate GPU memory
    global<f32>* gpu_a = gpu_alloc(sizeof(f32) * size);
    global<f32>* gpu_b = gpu_alloc(sizeof(f32) * size);
    global<f32>* gpu_result = gpu_alloc(sizeof(f32) * size);
    
    // ... initialize data ...
    
    // Launch kernel
    i32 blocks_x = (width + 15) / 16;
    i32 blocks_y = (height + 15) / 16;
    
    launch<{blocks_x, blocks_y, 1}, {16, 16, 1}> 
        gpu_matrix_add(gpu_a, gpu_b, gpu_result, width, height);
    
    // ... copy result back ...
    
    // Clean up
    gpu_free(gpu_a);
    gpu_free(gpu_b);
    gpu_free(gpu_result);
}
```

### Reduction Example

```ail
@parallel
func parallel_sum(const f32* data, i32 count) -> f32 {
    f32 total = 0.0f;
    
    @parallel_for (i32 i in 0..count) 
    reduce(total: add) {
        total += data[i];
    }
    
    return total;
}

@parallel
func parallel_min_max(const f32* data, i32 count) -> (f32, f32) {
    f32 min_val = 1e9f;
    f32 max_val = -1e9f;
    
    @parallel_for (i32 i in 0..count) 
    reduce(min_val: min, max_val: max) {
        min_val = min(min_val, data[i]);
        max_val = max(max_val, data[i]);
    }
    
    return (min_val, max_val);
}
```

### Vector Math Example

```ail
func vector_math_example() -> void {
    // Vector creation
    vec3 a = {1.0f, 2.0f, 3.0f};
    vec3 b = {4.0f, 5.0f, 6.0f};
    
    // Vector operations
    vec3 sum = a + b;
    vec3 diff = a - b;
    vec3 scaled = a * 2.0f;
    
    // Vector functions
    f32 dot_product = dot(a, b);
    vec3 cross_product = cross(a, b);
    f32 magnitude = length(a);
    vec3 normalized = normalize(a);
    
    // Component access
    f32 x = a.x;
    f32 y = a.y;
    f32 z = a.z;
    
    // Matrix operations
    mat4 transform = translation({1.0f, 2.0f, 3.0f}) * 
                    rotation_z(45.0f) * 
                    scale({2.0f, 2.0f, 2.0f});
    
    vec4 point = {1.0f, 1.0f, 1.0f, 1.0f};
    vec4 transformed = transform * point;
}
```

---

## Best Practices

### For AI Agents

1. **Use Consistent Patterns**
   ```ail
   // Consistent function naming
   func process_data(...) -> return_type
   func compute_result(...) -> return_type
   func analyze_input(...) -> return_type
   ```

2. **Explicit Type Annotations**
   ```ail
   // Good - explicit types
   f32 temperature = 25.5f;
   i32 count = 100;
   vec3 position = {0.0f, 1.0f, 2.0f};
   
   // Avoid - implicit types (except when obvious)
   auto x = some_complex_expression();
   ```

3. **Clear Memory Management**
   ```ail
   // Good - clear ownership
   heap f32* allocate_array(i32 size) -> heap f32* {
       return alloc(null, sizeof(f32) * size);
   }
   
   // Good - clear cleanup
   func process_with_cleanup(i32 size) -> void {
       heap f32* data = allocate_array(size);
       // ... use data ...
       free(data);  // Clear cleanup
   }
   ```

4. **Structured Parallelism**
   ```ail
   // Good - structured parallel pattern
   @parallel_for (i32 i in 0..count) {
       output[i] = transform(input[i]);
   }
   
   // Good - clear reduction
   @parallel_for (i32 i in 0..count) 
   reduce(total: add) {
       total += values[i];
   }
   ```

### Performance Best Practices

1. **Use Appropriate Storage**
   ```ail
   // Small, temporary data
   stack f32[100] temp_array;
   
   // Large, long-lived data
   heap f32* large_array = alloc(null, size);
   
   // Many small allocations
   arena* temp_arena = create_arena(1024 * 1024);
   arena f32* temp_data = alloc(temp_arena, small_size);
   ```

2. **Optimize Memory Access**
   ```ail
   // Good - sequential access
   @parallel_for (i32 i in 0..count) {
       output[i] = input[i] * 2.0f;  // Unit stride
   }
   
   // Good - vectorized access
   @parallel_for (i32 i in 0..count step 4) {
       vec4 data = load_vec4(&input[i]);
       vec4 result = data * 2.0f;
       store_vec4(&output[i], result);
   }
   ```

3. **Use Annotations Effectively**
   ```ail
   @hot @vectorize @parallel
   func performance_critical(...) -> void {
       // Frequently called, should be optimized
   }
   
   @cold @no_inline
   func error_handler(...) -> void {
       // Rarely called, don't optimize
   }
   ```

### GPU Best Practices

1. **Coalesced Memory Access**
   ```ail
   @kernel
   func good_memory_access(global<f32>* data, i32 size) -> void {
       i32 tid = global_thread_idx();
       if (tid < size) {
           data[tid] = data[tid] * 2.0f;  // Coalesced access
       }
   }
   ```

2. **Use Shared Memory Effectively**
   ```ail
   @kernel @block_size(32, 32, 1)
   func tiled_computation(global<f32>* data, i32 width) -> void {
       shared<f32>[32][32] tile;
       
       // Load into shared memory
       tile[ty][tx] = data[global_idx];
       sync_threads;
       
       // Compute using shared data
       f32 result = 0.0f;
       for (i32 i = 0; i < 32; i++) {
           result += tile[ty][i];
       }
       
       data[global_idx] = result;
   }
   ```

3. **Avoid Warp Divergence**
   ```ail
   @kernel
   func avoid_divergence(global<f32>* data, i32 size) -> void {
       i32 tid = global_thread_idx();
       
       // Good - all threads in warp take same path
       if (tid < size) {
           data[tid] = compute_value(tid);
       }
       
       // Avoid - threads in warp take different paths
       // if (data[tid] > 0.5f) { ... } else { ... }
   }
   ```

---

## Error Handling

AIL uses explicit error handling with return codes:

### Error Types

```ail
enum ErrorCode {
    OK = 0,
    OUT_OF_MEMORY,
    INVALID_INPUT,
    COMPUTATION_ERROR,
    GPU_ERROR,
    FILE_ERROR
}
```

### Error Handling Patterns

```ail
func safe_operation(i32 input) -> (i32, ErrorCode) {
    if (input < 0) {
        return (0, INVALID_INPUT);
    }
    
    heap f32* temp = alloc(null, input * sizeof(f32));
    if (temp == null) {
        return (0, OUT_OF_MEMORY);
    }
    
    i32 result = process_data(temp, input);
    free(temp);
    
    return (result, OK);
}

func example_with_error_handling() -> ErrorCode {
    (i32 result, ErrorCode err) = safe_operation(100);
    if (err != OK) {
        return err;  // Propagate error
    }
    
    // Use result...
    return OK;
}
```

### GPU Error Handling

```ail
func gpu_computation() -> ErrorCode {
    global<f32>* device_data = gpu_alloc(sizeof(f32) * 1000);
    if (device_data == null) {
        return GPU_ERROR;
    }
    
    // Launch kernel
    launch<{10, 1, 1}, {100, 1, 1}> kernel_function(device_data);
    
    // Check for kernel errors (implementation specific)
    ErrorCode gpu_status = get_last_gpu_error();
    if (gpu_status != OK) {
        gpu_free(device_data);
        return gpu_status;
    }
    
    gpu_free(device_data);
    return OK;
}
```

---

## Language Reference

### Operator Precedence

From highest to lowest precedence:

1. `() [] . ->` (postfix)
2. `! ~ + - * & (unary)` (unary)
3. `* / %` (multiplicative)
4. `+ -` (additive)
5. `<< >>` (shift)
6. `< <= > >=` (relational)
7. `== !=` (equality)
8. `&` (bitwise AND)
9. `^` (bitwise XOR)
10. `|` (bitwise OR)
11. `&&` (logical AND)
12. `||` (logical OR)
13. `= += -= *= /= %= <<= >>= &= ^= |=` (assignment)

### Type Conversion

#### Implicit Conversions

```ail
i32 i = 10;
f32 f = i;        // i32 to f32 (widening)
f64 d = f;        // f32 to f64 (widening)

vec3 v = {1, 2, 3};  // i32 literals to f32 (in vector context)
```

#### Explicit Conversions

```ail
f32 f = 3.14f;
i32 i = (i32)f;      // Explicit cast

vec3 v = {1.0f, 2.0f, 3.0f};
vec4 v4 = {v.x, v.y, v.z, 1.0f};  // Component extraction
```

### Storage Duration

| Storage Qualifier | Duration | Scope | Notes |
|------------------|----------|-------|-------|
| `stack` | Automatic | Block | Fastest allocation |
| `heap` | Manual | Program | Manual free required |
| `arena` | Arena | Arena lifetime | Bulk deallocation |
| `global<T>` | Manual | Program | GPU global memory |
| `shared<T>` | Block | Block | GPU shared memory |
| `local<T>` | Automatic | Thread | GPU thread-local |
| `constant<T>` | Static | Program | GPU constant memory |

### Annotation Summary

| Category | Annotation | Description |
|----------|------------|-------------|
| Performance | `@vectorize` | Enable SIMD vectorization |
| | `@parallel` | Enable parallelization |
| | `@unroll(N)` | Unroll loop N times |
| | `@hot` / `@cold` | Execution frequency hint |
| | `@inline` / `@no_inline` | Inlining control |
| Memory | `@memory_aligned(N)` | Align to N bytes |
| | `@coalesced` | GPU coalesced access |
| | `@no_alias` | No pointer aliasing |
| GPU | `@kernel` | GPU kernel function |
| | `@device` | GPU device function |
| | `@block_size(x,y,z)` | GPU block dimensions |
| | `@occupancy(N)` | Target occupancy % |
| Target | `@target("arch")` | Target architecture |

### Built-in Constants

```ail
// Mathematical constants
const f32 PI = 3.14159265f;
const f32 E = 2.71828183f;
const f32 SQRT_2 = 1.41421356f;

// GPU constants
const i32 WARP_SIZE = 32;       // NVIDIA warp size
const i32 MAX_THREADS_PER_BLOCK = 1024;
```

---

This completes the comprehensive AIL language documentation. The language is designed to be both powerful for high-performance computing and accessible for AI agents to understand and generate code effectively.