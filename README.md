# AI-Optimized Language Grammar & Type System

## Language Name: **AIL** (AI Implementation Language)

## 1. Lexical Structure

### Keywords
```
// Type keywords
i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 bool void
vec2 vec3 vec4 mat2 mat3 mat4
auto

// Storage qualifiers
stack heap arena const volatile

// Control flow
if else while for loop break continue return
switch case default

// Function keywords  
func extern inline

// Memory operations
alloc free sizeof alignof

// Literals
true false null
```

### Operators (Precedence High to Low)
```
1. () [] . ->                    // Postfix
2. ! ~ + - * & (unary)          // Unary
3. * / %                        // Multiplicative
4. + -                          // Additive  
5. << >>                        // Shift
6. < <= > >= == !=              // Relational
7. &                            // Bitwise AND
8. ^                            // Bitwise XOR
9. |                            // Bitwise OR
10. &&                          // Logical AND
11. ||                          // Logical OR
12. = += -= *= /= %= <<= >>= &= ^= |=  // Assignment
```

### Literals
```
// Integer literals
42          // i32 by default
42i64       // explicit type suffix
0xFF        // hexadecimal
0b1010      // binary

// Float literals  
3.14f       // f32
3.14        // f64 by default
1e-6f       // scientific notation

// Vector literals
{1, 2, 3}       // vec3 inferred from context
{1f, 2f, 3f}    // explicit f32 components

// String literals
"hello"     // null-terminated const char*
```

## 2. Type System

### Primitive Types
```
// Signed integers
i8, i16, i32, i64

// Unsigned integers  
u8, u16, u32, u64

// Floating point
f32, f64

// Boolean
bool

// Void
void
```

### Vector Types
```
// 2D vectors
vec2    // f32 x 2
ivec2   // i32 x 2  
uvec2   // u32 x 2

// 3D vectors
vec3, ivec3, uvec3

// 4D vectors
vec4, ivec4, uvec4
```

### Matrix Types
```
mat2    // 2x2 f32 matrix
mat3    // 3x3 f32 matrix  
mat4    // 4x4 f32 matrix

// Column-major storage (OpenGL style)
```

### Storage Qualifiers
```
stack   // Stack-allocated, automatic cleanup
heap    // Heap-allocated, manual management
arena   // Arena-allocated, bulk cleanup
const   // Immutable after initialization
volatile // Prevents optimization
```

### Pointer Types
```
type*           // Pointer to type
const type*     // Pointer to const data
type* const     // Const pointer
```

### Array Types
```
type[N]         // Fixed-size array
type[]          // Flexible array (last struct member only)
```

## 3. Grammar Production Rules

### Program Structure
```
program         → declaration*

declaration     → function_decl
                | variable_decl
                | type_decl
                | annotation

function_decl   → annotation* 'func' IDENTIFIER '(' parameter_list? ')' 
                  ('->' type)? block_stmt

parameter_list  → parameter (',' parameter)*
parameter       → storage_qual? type IDENTIFIER

variable_decl   → storage_qual? type IDENTIFIER ('=' expression)? ';'
```

### Statements
```
statement       → expression_stmt
                | declaration  
                | block_stmt
                | if_stmt
                | while_stmt
                | for_stmt
                | loop_stmt
                | return_stmt
                | break_stmt
                | continue_stmt

block_stmt      → '{' statement* '}'

if_stmt         → 'if' '(' expression ')' statement ('else' statement)?

while_stmt      → 'while' '(' expression ')' statement

for_stmt        → 'for' '(' variable_decl expression ';' expression ')' statement
                | 'for' '(' type IDENTIFIER 'in' range_expr ')' statement

loop_stmt       → 'loop' statement

return_stmt     → 'return' expression? ';'

range_expr      → expression '..' expression ('step' expression)?
```

### Expressions
```
expression      → assignment_expr

assignment_expr → logical_or_expr (('=' | '+=' | '-=' | ...) assignment_expr)?

logical_or_expr → logical_and_expr ('||' logical_and_expr)*

logical_and_expr → equality_expr ('&&' equality_expr)*

equality_expr   → relational_expr (('==' | '!=') relational_expr)*

relational_expr → shift_expr (('<' | '>' | '<=' | '>=') shift_expr)*

shift_expr      → additive_expr (('<<' | '>>') additive_expr)*

additive_expr   → mult_expr (('+' | '-') mult_expr)*

mult_expr       → unary_expr (('*' | '/' | '%') unary_expr)*

unary_expr      → ('!' | '~' | '+' | '-' | '*' | '&') unary_expr
                | postfix_expr

postfix_expr    → primary_expr ('[' expression ']' | '(' arg_list? ')' | '.' IDENTIFIER)*

primary_expr    → IDENTIFIER
                | literal
                | '(' expression ')'
                | vector_literal
                | matrix_literal

vector_literal  → '{' expression (',' expression)* '}'

arg_list        → expression (',' expression)*
```

## 4. Performance Annotations

```
annotation      → '@' IDENTIFIER ('(' annotation_args? ')')?

annotation_args → annotation_arg (',' annotation_arg)*
annotation_arg  → IDENTIFIER ('=' literal)?

// Built-in annotations
@vectorize                  // Enable auto-vectorization
@unroll(N)                 // Loop unrolling factor
@simd_width(N)             // SIMD width hint
@memory_aligned(N)         // Memory alignment requirement  
@inline                    // Force inline
@no_inline                 // Prevent inline
@pure                      // Function has no side effects
@hot                       // Frequently executed code
@cold                      // Rarely executed code
@target("arch")            // Target specific architecture
```

## 5. Built-in Functions

### Vector Operations
```
dot(vec, vec) -> scalar        // Dot product
cross(vec3, vec3) -> vec3      // Cross product  
length(vec) -> scalar          // Vector magnitude
normalize(vec) -> vec          // Unit vector
reflect(vec, vec) -> vec       // Reflection
```

### Matrix Operations  
```
identity() -> mat4             // Identity matrix
transpose(mat) -> mat          // Matrix transpose
inverse(mat) -> mat            // Matrix inverse
translation(vec3) -> mat4      // Translation matrix
rotation(scalar) -> mat4       // Rotation matrix
scale(vec3) -> mat4           // Scale matrix
```

### Memory Operations
```
alloc(arena*, size) -> void*   // Allocate memory
free(void*)                    // Free memory
sizeof(type) -> u64            // Size of type
alignof(type) -> u64           // Alignment of type
```

### Math Functions
```
abs(x) -> x                    // Absolute value
min(x, y) -> x                 // Minimum
max(x, y) -> x                 // Maximum
clamp(x, min, max) -> x        // Clamp to range
lerp(a, b, t) -> a             // Linear interpolation
```

## 6. Example Program

```ail
// Matrix multiplication kernel optimized for AI generation
@vectorize @simd_width(8) @hot
func matrix_multiply(
    const f32* a,           // Input matrix A
    const f32* b,           // Input matrix B  
    heap f32* result,       // Output matrix
    i32 rows,              // Matrix dimensions
    i32 cols,
    i32 inner_dim
) -> void {
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

func main() -> i32 {
    const i32 size = 1024;
    
    // Stack allocation for small arrays
    stack f32[16] small_matrix = {0};
    
    // Heap allocation for large arrays
    heap f32* large_a = alloc(null, sizeof(f32) * size * size);
    heap f32* large_b = alloc(null, sizeof(f32) * size * size);  
    heap f32* result = alloc(null, sizeof(f32) * size * size);
    
    matrix_multiply(large_a, large_b, result, size, size, size);
    
    free(large_a);
    free(large_b);
    free(result);
    
    return 0;
}
```

## Design Rationale for AI Agents

1. **Consistent Syntax**: Every construct follows predictable patterns
2. **Explicit Annotations**: Performance hints are visible and regular
3. **Clear Memory Model**: Storage qualifiers make ownership explicit
4. **Rich Type System**: Built-in support for common numerical operations
5. **Composable**: Small pieces combine in obvious ways
6. **Self-Documenting**: Code clearly expresses intent and constraints

## Designed By Sean Southard

