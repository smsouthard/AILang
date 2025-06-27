#!/usr/bin/env python3
"""
AIL (AI Implementation Language) LLVM Code Generator
Generates LLVM IR from AIL AST with support for CPU parallelization and GPU kernels.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import llvmlite.ir as ir
import llvmlite.binding as llvm
from llvmlite import ir as llvm_ir

# Import our previous modules
from ail_parser import *
from ail_semantic_analyzer import *

# =============================================================================
# LLVM Code Generation Context
# =============================================================================

class TargetType(Enum):
    CPU = auto()
    GPU_CUDA = auto()
    GPU_OPENCL = auto()

@dataclass
class CodeGenContext:
    module: ir.Module
    builder: Optional[ir.IRBuilder]
    function: Optional[ir.Function]
    target_type: TargetType
    
    # Symbol tables for LLVM values
    variables: Dict[str, ir.Value]
    functions: Dict[str, ir.Function]
    types: Dict[str, ir.Type]
    
    # Parallel code generation
    in_parallel_loop: bool = False
    parallel_loop_bounds: Optional[Tuple[ir.Value, ir.Value, ir.Value]] = None
    reduction_variables: Dict[str, Tuple[ir.Value, str]] = None
    
    # GPU-specific context
    gpu_thread_indices: Dict[str, ir.Value] = None
    shared_memory_vars: Dict[str, ir.Value] = None
    
    def __post_init__(self):
        if self.reduction_variables is None:
            self.reduction_variables = {}
        if self.gpu_thread_indices is None:
            self.gpu_thread_indices = {}
        if self.shared_memory_vars is None:
            self.shared_memory_vars = {}

# =============================================================================
# LLVM Code Generator
# =============================================================================

class LLVMCodeGenerator:
    def __init__(self, target_type: TargetType = TargetType.CPU):
        self.target_type = target_type
        self.module = ir.Module(name="ail_module")
        self.context = CodeGenContext(
            module=self.module,
            builder=None,
            function=None,
            target_type=target_type,
            variables={},
            functions={},
            types={}
        )
        
        # Initialize LLVM
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Set up target-specific configuration
        self.setup_target()
        
        # Create built-in types and functions
        self.create_builtin_types()
        self.create_builtin_functions()
    
    def setup_target(self):
        """Set up target-specific configuration"""
        if self.target_type == TargetType.CPU:
            # CPU target configuration
            self.module.data_layout = ""
            self.module.triple = llvm.get_default_triple()
        elif self.target_type == TargetType.GPU_CUDA:
            # CUDA target configuration
            self.module.data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
            self.module.triple = "nvptx64-nvidia-cuda"
    
    def create_builtin_types(self):
        """Create LLVM types for AIL built-in types"""
        # Primitive types
        self.context.types.update({
            'void': ir.VoidType(),
            'bool': ir.IntType(1),
            'i8': ir.IntType(8),
            'i16': ir.IntType(16),
            'i32': ir.IntType(32),
            'i64': ir.IntType(64),
            'u8': ir.IntType(8),
            'u16': ir.IntType(16),
            'u32': ir.IntType(32),
            'u64': ir.IntType(64),
            'f32': ir.FloatType(),
            'f64': ir.DoubleType(),
        })
        
        # Vector types
        self.context.types.update({
            'vec2': ir.VectorType(ir.FloatType(), 2),
            'vec3': ir.VectorType(ir.FloatType(), 3),
            'vec4': ir.VectorType(ir.FloatType(), 4),
        })
        
        # Matrix types (represented as arrays)
        self.context.types.update({
            'mat2': ir.ArrayType(ir.FloatType(), 4),
            'mat3': ir.ArrayType(ir.FloatType(), 9),
            'mat4': ir.ArrayType(ir.FloatType(), 16),
        })
        
        # Pointer types
        self.context.types.update({
            'ptr': ir.PointerType(ir.IntType(8)),  # Generic pointer
        })
    
    def create_builtin_functions(self):
        """Create LLVM function declarations for built-in functions"""
        # Math functions
        self.create_math_functions()
        
        # Memory functions
        self.create_memory_functions()
        
        # GPU functions
        if self.target_type == TargetType.GPU_CUDA:
            self.create_gpu_functions()
        
        # Parallel runtime functions
        self.create_parallel_functions()
    
    def create_math_functions(self):
        """Create mathematical built-in functions"""
        f32_type = self.context.types['f32']
        vec3_type = self.context.types['vec3']
        
        # Scalar math functions
        for func_name in ['abs', 'sqrt', 'sin', 'cos', 'tan', 'exp', 'log']:
            func_type = ir.FunctionType(f32_type, [f32_type])
            func = ir.Function(self.module, func_type, f"llvm.{func_name}.f32")
            self.context.functions[func_name] = func
        
        # Vector functions
        # dot product: vec3 x vec3 -> f32
        dot_type = ir.FunctionType(f32_type, [vec3_type, vec3_type])
        self.context.functions['dot'] = ir.Function(self.module, dot_type, 'ail_dot')
        
        # cross product: vec3 x vec3 -> vec3
        cross_type = ir.FunctionType(vec3_type, [vec3_type, vec3_type])
        self.context.functions['cross'] = ir.Function(self.module, cross_type, 'ail_cross')
        
        # length: vec3 -> f32
        length_type = ir.FunctionType(f32_type, [vec3_type])
        self.context.functions['length'] = ir.Function(self.module, length_type, 'ail_length')
        
        # normalize: vec3 -> vec3
        normalize_type = ir.FunctionType(vec3_type, [vec3_type])
        self.context.functions['normalize'] = ir.Function(self.module, normalize_type, 'ail_normalize')
    
    def create_memory_functions(self):
        """Create memory management functions"""
        ptr_type = self.context.types['ptr']
        i64_type = self.context.types['i64']
        void_type = self.context.types['void']
        
        # malloc: i64 -> ptr
        malloc_type = ir.FunctionType(ptr_type, [i64_type])
        self.context.functions['malloc'] = ir.Function(self.module, malloc_type, 'malloc')
        
        # free: ptr -> void
        free_type = ir.FunctionType(void_type, [ptr_type])
        self.context.functions['free'] = ir.Function(self.module, free_type, 'free')
        
        # aligned_alloc: i64, i64 -> ptr
        aligned_alloc_type = ir.FunctionType(ptr_type, [i64_type, i64_type])
        self.context.functions['aligned_alloc'] = ir.Function(self.module, aligned_alloc_type, 'aligned_alloc')
    
    def create_gpu_functions(self):
        """Create GPU-specific functions"""
        i32_type = self.context.types['i32']
        
        # Thread indexing functions
        gpu_funcs = {
            'thread_idx_x': 'llvm.nvvm.read.ptx.sreg.tid.x',
            'thread_idx_y': 'llvm.nvvm.read.ptx.sreg.tid.y',
            'thread_idx_z': 'llvm.nvvm.read.ptx.sreg.tid.z',
            'block_idx_x': 'llvm.nvvm.read.ptx.sreg.ctaid.x',
            'block_idx_y': 'llvm.nvvm.read.ptx.sreg.ctaid.y',
            'block_idx_z': 'llvm.nvvm.read.ptx.sreg.ctaid.z',
            'block_dim_x': 'llvm.nvvm.read.ptx.sreg.ntid.x',
            'block_dim_y': 'llvm.nvvm.read.ptx.sreg.ntid.y',
            'block_dim_z': 'llvm.nvvm.read.ptx.sreg.ntid.z',
            'grid_dim_x': 'llvm.nvvm.read.ptx.sreg.nctaid.x',
            'grid_dim_y': 'llvm.nvvm.read.ptx.sreg.nctaid.y',
            'grid_dim_z': 'llvm.nvvm.read.ptx.sreg.nctaid.z',
        }
        
        for ail_name, llvm_name in gpu_funcs.items():
            func_type = ir.FunctionType(i32_type, [])
            func = ir.Function(self.module, func_type, llvm_name)
            self.context.functions[ail_name] = func
        
        # Synchronization functions
        void_type = self.context.types['void']
        
        # Block synchronization
        barrier_type = ir.FunctionType(void_type, [])
        self.context.functions['sync_threads'] = ir.Function(self.module, barrier_type, 'llvm.nvvm.barrier0')
        
        # Warp functions
        f32_type = self.context.types['f32']
        
        # Warp shuffle
        shuffle_type = ir.FunctionType(f32_type, [f32_type, i32_type])
        self.context.functions['warp_shuffle'] = ir.Function(self.module, shuffle_type, 'llvm.nvvm.shfl.sync.bfly.f32')
        
        # Warp reduction (implemented as a wrapper around shuffle)
        reduce_type = ir.FunctionType(f32_type, [f32_type])
        self.context.functions['warp_reduce_add'] = ir.Function(self.module, reduce_type, 'ail_warp_reduce_add')
    
    def create_parallel_functions(self):
        """Create parallel runtime functions"""
        if self.target_type == TargetType.CPU:
            void_type = self.context.types['void']
            i32_type = self.context.types['i32']
            ptr_type = self.context.types['ptr']
            
            # OpenMP-style parallel for
            parallel_for_type = ir.FunctionType(void_type, [ptr_type, i32_type, i32_type, i32_type])
            self.context.functions['parallel_for'] = ir.Function(self.module, parallel_for_type, 'ail_parallel_for')
            
            # Thread management
            thread_id_type = ir.FunctionType(i32_type, [])
            self.context.functions['get_thread_id'] = ir.Function(self.module, thread_id_type, 'omp_get_thread_num')
            self.context.functions['get_thread_count'] = ir.Function(self.module, thread_id_type, 'omp_get_num_threads')
    
    def generate(self, program: Program) -> str:
        """Generate LLVM IR for the entire program"""
        try:
            self.visit_program(program)
            return str(self.module)
        except Exception as e:
            print(f"Code generation error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def visit_program(self, node: Program):
        """Visit program node"""
        # Generate all function declarations first
        for decl in node.declarations:
            if isinstance(decl, FunctionDeclaration):
                self.declare_function(decl)
        
        # Then generate function bodies
        for decl in node.declarations:
            self.visit_declaration(decl)
    
    def declare_function(self, node: FunctionDeclaration):
        """Declare function signature"""
        # Get parameter types
        param_types = []
        for param in node.parameters:
            param_type = self.get_llvm_type(param.type)
            if param_type:
                param_types.append(param_type)
        
        # Get return type
        if node.return_type:
            return_type = self.get_llvm_type(node.return_type)
        else:
            return_type = self.context.types['void']
        
        # Create function type
        func_type = ir.FunctionType(return_type, param_types)
        
        # Create function
        func = ir.Function(self.module, func_type, node.name)
        
        # Set function attributes based on annotations
        self.set_function_attributes(func, node)
        
        self.context.functions[node.name] = func
    
    def set_function_attributes(self, func: ir.Function, node: FunctionDeclaration):
        """Set function attributes based on annotations"""
        for annotation in node.annotations:
            if annotation.name == 'kernel':
                if self.target_type == TargetType.GPU_CUDA:
                    func.attributes.add('kernel')
            elif annotation.name == 'inline':
                func.attributes.add('alwaysinline')
            elif annotation.name == 'no_inline':
                func.attributes.add('noinline')
            elif annotation.name == 'pure':
                func.attributes.add('readonly')
                func.attributes.add('nounwind')
            elif annotation.name == 'hot':
                func.attributes.add('hot')
            elif annotation.name == 'cold':
                func.attributes.add('cold')
    
    def visit_declaration(self, node: Declaration):
        """Visit declaration node"""
        if isinstance(node, FunctionDeclaration):
            self.visit_function_declaration(node)
        elif isinstance(node, VariableDeclaration):
            self.visit_global_variable_declaration(node)
    
    def visit_function_declaration(self, node: FunctionDeclaration):
        """Visit function declaration"""
        func = self.context.functions[node.name]
        
        # Create entry block
        entry_block = func.append_basic_block(name="entry")
        builder = ir.IRBuilder(entry_block)
        
        # Save context
        old_builder = self.context.builder
        old_function = self.context.function
        old_variables = self.context.variables.copy()
        
        self.context.builder = builder
        self.context.function = func
        self.context.variables = {}
        
        try:
            # Add parameters to variable map
            for i, param in enumerate(node.parameters):
                func.args[i].name = param.name
                self.context.variables[param.name] = func.args[i]
            
            # Generate function body
            self.visit_statement(node.body)
            
            # Add return if missing
            if not builder.block.is_terminated:
                if node.return_type and node.return_type.name != 'void':
                    # Return zero value
                    zero_value = self.get_zero_value(self.get_llvm_type(node.return_type))
                    builder.ret(zero_value)
                else:
                    builder.ret_void()
        
        finally:
            # Restore context
            self.context.builder = old_builder
            self.context.function = old_function
            self.context.variables = old_variables
    
    def visit_global_variable_declaration(self, node: VariableDeclaration):
        """Visit global variable declaration"""
        var_type = self.get_llvm_type(node.type)
        if not var_type:
            return
        
        # Create global variable
        if node.initializer:
            # TODO: Evaluate constant initializer
            initial_value = self.get_zero_value(var_type)
        else:
            initial_value = self.get_zero_value(var_type)
        
        global_var = ir.GlobalVariable(self.module, var_type, node.name)
        global_var.initializer = initial_value
        
        # Set linkage based on storage qualifier
        if node.storage_qualifier == 'const':
            global_var.global_constant = True
        
        self.context.variables[node.name] = global_var
    
    def visit_statement(self, node: Statement) -> Optional[ir.Value]:
        """Visit statement node"""
        if isinstance(node, BlockStatement):
            return self.visit_block_statement(node)
        elif isinstance(node, ExpressionStatement):
            return self.visit_expression_statement(node)
        elif isinstance(node, IfStatement):
            return self.visit_if_statement(node)
        elif isinstance(node, WhileStatement):
            return self.visit_while_statement(node)
        elif isinstance(node, ForStatement):
            return self.visit_for_statement(node)
        elif isinstance(node, ParallelForStatement):
            return self.visit_parallel_for_statement(node)
        elif isinstance(node, KernelLaunchStatement):
            return self.visit_kernel_launch_statement(node)
        elif isinstance(node, ReturnStatement):
            return self.visit_return_statement(node)
        elif isinstance(node, SyncStatement):
            return self.visit_sync_statement(node)
        elif isinstance(node, VariableDeclaration):
            return self.visit_local_variable_declaration(node)
        return None
    
    def visit_block_statement(self, node: BlockStatement):
        """Visit block statement"""
        # Save variables for scope
        old_variables = self.context.variables.copy()
        
        try:
            for stmt in node.statements:
                self.visit_statement(stmt)
                if self.context.builder.block.is_terminated:
                    break
        finally:
            # Restore variables
            self.context.variables = old_variables
    
    def visit_expression_statement(self, node: ExpressionStatement):
        """Visit expression statement"""
        return self.visit_expression(node.expression)
    
    def visit_if_statement(self, node: IfStatement):
        """Visit if statement"""
        # Evaluate condition
        condition = self.visit_expression(node.condition)
        condition = self.convert_to_bool(condition)
        
        # Create basic blocks
        then_block = self.context.function.append_basic_block(name="if.then")
        else_block = self.context.function.append_basic_block(name="if.else") if node.else_stmt else None
        end_block = self.context.function.append_basic_block(name="if.end")
        
        # Branch
        if else_block:
            self.context.builder.cbranch(condition, then_block, else_block)
        else:
            self.context.builder.cbranch(condition, then_block, end_block)
        
        # Generate then block
        self.context.builder.position_at_end(then_block)
        self.visit_statement(node.then_stmt)
        if not self.context.builder.block.is_terminated:
            self.context.builder.branch(end_block)
        
        # Generate else block
        if node.else_stmt:
            self.context.builder.position_at_end(else_block)
            self.visit_statement(node.else_stmt)
            if not self.context.builder.block.is_terminated:
                self.context.builder.branch(end_block)
        
        # Continue with end block
        self.context.builder.position_at_end(end_block)
    
    def visit_while_statement(self, node: WhileStatement):
        """Visit while statement"""
        # Create basic blocks
        cond_block = self.context.function.append_basic_block(name="while.cond")
        body_block = self.context.function.append_basic_block(name="while.body")
        end_block = self.context.function.append_basic_block(name="while.end")
        
        # Branch to condition
        self.context.builder.branch(cond_block)
        
        # Generate condition block
        self.context.builder.position_at_end(cond_block)
        condition = self.visit_expression(node.condition)
        condition = self.convert_to_bool(condition)
        self.context.builder.cbranch(condition, body_block, end_block)
        
        # Generate body block
        self.context.builder.position_at_end(body_block)
        self.visit_statement(node.body)
        if not self.context.builder.block.is_terminated:
            self.context.builder.branch(cond_block)
        
        # Continue with end block
        self.context.builder.position_at_end(end_block)
    
    def visit_for_statement(self, node: ForStatement):
        """Visit for statement"""
        # Save variables for scope
        old_variables = self.context.variables.copy()
        
        try:
            # Generate initialization
            if node.init:
                if isinstance(node.init, VariableDeclaration):
                    self.visit_local_variable_declaration(node.init)
                else:
                    self.visit_statement(node.init)
            
            # Create basic blocks
            cond_block = self.context.function.append_basic_block(name="for.cond")
            body_block = self.context.function.append_basic_block(name="for.body")
            update_block = self.context.function.append_basic_block(name="for.update")
            end_block = self.context.function.append_basic_block(name="for.end")
            
            # Branch to condition
            self.context.builder.branch(cond_block)
            
            # Generate condition block
            self.context.builder.position_at_end(cond_block)
            if node.condition:
                condition = self.visit_expression(node.condition)
                condition = self.convert_to_bool(condition)
                self.context.builder.cbranch(condition, body_block, end_block)
            else:
                self.context.builder.branch(body_block)
            
            # Generate body block
            self.context.builder.position_at_end(body_block)
            self.visit_statement(node.body)
            if not self.context.builder.block.is_terminated:
                self.context.builder.branch(update_block)
            
            # Generate update block
            self.context.builder.position_at_end(update_block)
            if node.update:
                self.visit_expression(node.update)
            self.context.builder.branch(cond_block)
            
            # Continue with end block
            self.context.builder.position_at_end(end_block)
        
        finally:
            # Restore variables
            self.context.variables = old_variables
    
    def visit_parallel_for_statement(self, node: ParallelForStatement):
        """Visit parallel for statement"""
        if self.target_type == TargetType.CPU:
            return self.visit_cpu_parallel_for(node)
        elif self.target_type == TargetType.GPU_CUDA:
            return self.visit_gpu_parallel_for(node)
    
    def visit_cpu_parallel_for(self, node: ParallelForStatement):
        """Generate CPU parallel for loop using OpenMP-style calls"""
        # For now, generate a regular for loop
        # In a full implementation, this would generate OpenMP parallel for
        
        # Create a regular for loop structure
        old_variables = self.context.variables.copy()
        old_in_parallel = self.context.in_parallel_loop
        
        self.context.in_parallel_loop = True
        
        try:
            # Allocate loop variable
            loop_var_type = self.context.types['i32']
            loop_var = self.context.builder.alloca(loop_var_type, name=node.variable)
            self.context.variables[node.variable] = loop_var
            
            # Initialize loop variable
            start_val = self.visit_expression(node.range_expr.start)
            self.context.builder.store(start_val, loop_var)
            
            # Get loop bounds
            end_val = self.visit_expression(node.range_expr.end)
            step_val = self.visit_expression(node.range_expr.step) if node.range_expr.step else ir.Constant(loop_var_type, 1)
            
            # Store bounds for reduction handling
            self.context.parallel_loop_bounds = (start_val, end_val, step_val)
            
            # Handle reduction variables
            self.setup_reduction_variables(node.reduce_clauses)
            
            # Create loop blocks
            cond_block = self.context.function.append_basic_block(name="pfor.cond")
            body_block = self.context.function.append_basic_block(name="pfor.body")
            update_block = self.context.function.append_basic_block(name="pfor.update")
            end_block = self.context.function.append_basic_block(name="pfor.end")
            
            # Branch to condition
            self.context.builder.branch(cond_block)
            
            # Generate condition block
            self.context.builder.position_at_end(cond_block)
            current_val = self.context.builder.load(loop_var)
            condition = self.context.builder.icmp_signed('<', current_val, end_val)
            self.context.builder.cbranch(condition, body_block, end_block)
            
            # Generate body block
            self.context.builder.position_at_end(body_block)
            self.visit_statement(node.body)
            if not self.context.builder.block.is_terminated:
                self.context.builder.branch(update_block)
            
            # Generate update block
            self.context.builder.position_at_end(update_block)
            current_val = self.context.builder.load(loop_var)
            new_val = self.context.builder.add(current_val, step_val)
            self.context.builder.store(new_val, loop_var)
            self.context.builder.branch(cond_block)
            
            # Continue with end block
            self.context.builder.position_at_end(end_block)
            
            # Finalize reductions
            self.finalize_reduction_variables(node.reduce_clauses)
        
        finally:
            self.context.variables = old_variables
            self.context.in_parallel_loop = old_in_parallel
            self.context.parallel_loop_bounds = None
            self.context.reduction_variables.clear()
    
    def visit_gpu_parallel_for(self, node: ParallelForStatement):
        """Handle GPU parallel for (usually handled by kernel launch)"""
        # For GPU, parallel for is typically handled by thread indexing
        # This is a simplified implementation
        return self.visit_cpu_parallel_for(node)
    
    def setup_reduction_variables(self, reduce_clauses: List[ReduceClause]):
        """Set up reduction variables for parallel loop"""
        for clause in reduce_clauses:
            var_name = clause.variable
            operation = clause.operation
            
            # Get the variable
            if var_name in self.context.variables:
                var = self.context.variables[var_name]
                # Create a local copy for reduction
                var_type = var.type.pointee
                reduction_var = self.context.builder.alloca(var_type, name=f"{var_name}_reduction")
                
                # Initialize with identity value
                identity = self.get_reduction_identity(operation, var_type)
                self.context.builder.store(identity, reduction_var)
                
                self.context.reduction_variables[var_name] = (reduction_var, operation)
    
    def finalize_reduction_variables(self, reduce_clauses: List[ReduceClause]):
        """Finalize reduction variables after parallel loop"""
        for clause in reduce_clauses:
            var_name = clause.variable
            if var_name in self.context.reduction_variables:
                reduction_var, operation = self.context.reduction_variables[var_name]
                original_var = self.context.variables[var_name]
                
                # Store the reduction result back to original variable
                result = self.context.builder.load(reduction_var)
                self.context.builder.store(result, original_var)
    
    def visit_kernel_launch_statement(self, node: KernelLaunchStatement):
        """Visit kernel launch statement"""
        if self.target_type != TargetType.GPU_CUDA:
            return  # Only handle CUDA launches
        
        # Get kernel function
        kernel_func = self.context.functions.get(node.function_name)
        if not kernel_func:
            return
        
        # Generate CUDA kernel launch
        # This would typically generate a call to cudaLaunchKernel
        # For now, we'll generate a placeholder
        
        # Create kernel launch function if not exists
        if 'cuda_launch_kernel' not in self.context.functions:
            ptr_type = self.context.types['ptr']
            i32_type = self.context.types['i32']
            void_type = self.context.types['void']
            
            launch_type = ir.FunctionType(void_type, [
                ptr_type,  # kernel function
                i32_type, i32_type, i32_type,  # grid dimensions
                i32_type, i32_type, i32_type,  # block dimensions
                ptr_type,  # arguments
                i32_type   # shared memory size
            ])
            
            self.context.functions['cuda_launch_kernel'] = ir.Function(
                self.module, launch_type, 'cudaLaunchKernel'
            )
        
        # Generate the call (simplified)
        launch_func = self.context.functions['cuda_launch_kernel']
        
        # Extract grid and block dimensions
        grid_args = [self.visit_expression(expr) for expr in node.grid_config]
        block_args = [self.visit_expression(expr) for expr in node.block_config]
        
        # Pad to 3 dimensions
        i32_type = self.context.types['i32']
        one = ir.Constant(i32_type, 1)
        while len(grid_args) < 3:
            grid_args.append(one)
        while len(block_args) < 3:
            block_args.append(one)
        
        # Create argument array (simplified)
        null_ptr = ir.Constant(self.context.types['ptr'], None)
        zero = ir.Constant(i32_type, 0)
        
        # Call kernel launch
        self.context.builder.call(launch_func, [
            kernel_func, *grid_args, *block_args, null_ptr, zero
        ])
    
    def visit_return_statement(self, node: ReturnStatement):
        """Visit return statement"""
        if node.value:
            value = self.visit_expression(node.value)
            self.context.builder.ret(value)
        else:
            self.context.builder.ret_void()
    
    def visit_sync_statement(self, node: SyncStatement):
        """Visit sync statement"""
        if node.sync_type == 'sync_threads':
            if 'sync_threads' in self.context.functions:
                self.context.builder.call(self.context.functions['sync_threads'], [])
        elif node.sync_type == 'sync_warp':
            # Warp sync is implicit in most operations
            pass
        elif node.sync_type == 'sync':
            # General sync - could be OpenMP barrier for CPU
            pass
    
    def visit_local_variable_declaration(self, node: VariableDeclaration):
        """Visit local variable declaration"""
        var_type = self.get_llvm_type(node.type)
        if not var_type:
            return
        
        # Handle storage qualifiers
        if node.storage_qualifier == 'shared' and self.target_type == TargetType.GPU_CUDA:
            # GPU shared memory
            shared_var = ir.GlobalVariable(self.module, var_type, f"__shared__{node.name}")
            shared_var.storage_class = 'shared'
            self.context.variables[node.name] = shared_var
            self.context.shared_memory_vars[node.name] = shared_var
        else:
            # Regular stack allocation
            var = self.context.builder.alloca(var_type, name=node.name)
            self.context.variables[node.name] = var
            
            # Initialize if needed
            if node.initializer:
                init_value = self.visit_expression(node.initializer)
                self.context.builder.store(init_value, var)
        
        return self.context.variables[node.name]
    
    def visit_expression(self, node: Expression) -> ir.Value:
        """Visit expression node"""
        if isinstance(node, BinaryExpression):
            return self.visit_binary_expression(node)
        elif isinstance(node, UnaryExpression):
            return self.visit_unary_expression(node)
        elif isinstance(node, CallExpression):
            return self.visit_call_expression(node)
        elif isinstance(node, MemberExpression):
            return self.visit_member_expression(node)
        elif isinstance(node, IdentifierExpression):
            return self.visit_identifier_expression(node)
        elif isinstance(node, LiteralExpression):
            return self.visit_literal_expression(node)
        elif isinstance(node, VectorLiteralExpression):
            return self.visit_vector_literal_expression(node)
        elif isinstance(node, AssignmentExpression):
            return self.visit_assignment_expression(node)
        else:
            # Return a placeholder value
            return ir.Constant(self.context.types['i32'], 0)
    
    def visit_binary_expression(self, node: BinaryExpression) -> ir.Value:
        """Visit binary expression"""
        left = self.visit_expression(node.left)
        right = self.visit_expression(node.right)
        
        # Arithmetic operations
        if node.operator == '+':
            if left.type.is_integer:
                return self.context.builder.add(left, right)
            elif left.type.is_float:
                return self.context.builder.fadd(left, right)
        elif node.operator == '-':
            if left.type.is_integer:
                return self.context.builder.sub(left, right)
            elif left.type.is_float:
                return self.context.builder.fsub(left, right)
        elif node.operator == '*':
            if left.type.is_integer:
                return self.context.builder.mul(left, right)
            elif left.type.is_float:
                return self.context.builder.fmul(left, right)
        elif node.operator == '/':
            if left.type.is_integer:
                return self.context.builder.sdiv(left, right)
            elif left.type.is_float:
                return self.context.builder.fdiv(left, right)
        elif node.operator == '%':
            if left.type.is_integer:
                return self.context.builder.srem(left, right)
            elif left.type.is_float:
                return self.context.builder.frem(left, right)
        
        # Comparison operations
        elif node.operator == '<':
            if left.type.is_integer:
                return self.context.builder.icmp_signed('<', left, right)
            elif left.type.is_float:
                return self.context.builder.fcmp_ordered('<', left, right)
        elif node.operator == '<=':
            if left.type.is_integer:
                return self.context.builder.icmp_signed('<=', left, right)
            elif left.type.is_float:
                return self.context.builder.fcmp_ordered('<=', left, right)
        elif node.operator == '>':
            if left.type.is_integer:
                return self.context.builder.icmp_signed('>', left, right)
            elif left.type.is_float:
                return self.context.builder.fcmp_ordered('>', left, right)
        elif node.operator == '>=':
            if left.type.is_integer:
                return self.context.builder.icmp_signed('>=', left, right)
            elif left.type.is_float:
                return self.context.builder.fcmp_ordered('>=', left, right)
        elif node.operator == '==':
            if left.type.is_integer:
                return self.context.builder.icmp_signed('==', left, right)
            elif left.type.is_float:
                return self.context.builder.fcmp_ordered('==', left, right)
        elif node.operator == '!=':
            if left.type.is_integer:
                return self.context.builder.icmp_signed('!=', left, right)
            elif left.type.is_float:
                return self.context.builder.fcmp_ordered('!=', left, right)
        
        # Logical operations
        elif node.operator == '&&':
            return self.context.builder.and_(left, right)
        elif node.operator == '||':
            return self.context.builder.or_(left, right)
        
        # Bitwise operations
        elif node.operator == '&':
            return self.context.builder.and_(left, right)
        elif node.operator == '|':
            return self.context.builder.or_(left, right)
        elif node.operator == '^':
            return self.context.builder.xor(left, right)
        elif node.operator == '<<':
            return self.context.builder.shl(left, right)
        elif node.operator == '>>':
            return self.context.builder.ashr(left, right)
        
        # Default: return left operand
        return left
    
    def visit_unary_expression(self, node: UnaryExpression) -> ir.Value:
        """Visit unary expression"""
        operand = self.visit_expression(node.operand)
        
        if node.operator == '+':
            return operand
        elif node.operator == '-':
            if operand.type.is_integer:
                return self.context.builder.neg(operand)
            elif operand.type.is_float:
                return self.context.builder.fneg(operand)
        elif node.operator == '!':
            return self.context.builder.not_(operand)
        elif node.operator == '~':
            return self.context.builder.not_(operand)
        elif node.operator == '*':
            # Dereference
            return self.context.builder.load(operand)
        elif node.operator == '&':
            # Address-of (operand should be an lvalue)
            return operand
        
        return operand
    
    def visit_call_expression(self, node: CallExpression) -> ir.Value:
        """Visit call expression"""
        if isinstance(node.function, IdentifierExpression):
            func_name = node.function.name
            
            # Handle built-in functions
            if func_name in self.context.functions:
                func = self.context.functions[func_name]
                args = [self.visit_expression(arg) for arg in node.arguments]
                
                # Special handling for certain functions
                if func_name == 'warp_reduce_add':
                    return self.generate_warp_reduce_add(args[0] if args else None)
                elif func_name in ['thread_idx_x', 'thread_idx_y', 'thread_idx_z',
                                  'block_idx_x', 'block_idx_y', 'block_idx_z']:
                    return self.context.builder.call(func, [])
                else:
                    return self.context.builder.call(func, args)
            else:
                # Unknown function - return placeholder
                return ir.Constant(self.context.types['i32'], 0)
        else:
            # Function pointer call
            func_value = self.visit_expression(node.function)
            args = [self.visit_expression(arg) for arg in node.arguments]
            return self.context.builder.call(func_value, args)
    
    def generate_warp_reduce_add(self, value: ir.Value) -> ir.Value:
        """Generate warp reduction using shuffle operations"""
        if not value:
            return ir.Constant(self.context.types['f32'], 0.0)
        
        # This is a simplified warp reduction
        # In a full implementation, this would generate a loop with shuffle operations
        shuffle_func = self.context.functions.get('warp_shuffle')
        if shuffle_func:
            # Simplified: just return the input value
            return value
        
        return value
    
    def visit_member_expression(self, node: MemberExpression) -> ir.Value:
        """Visit member expression"""
        obj = self.visit_expression(node.object)
        
        if node.computed:
            # Array access: obj[index]
            # Note: node.property is empty for computed access, we need the index
            # This is a simplified implementation
            zero = ir.Constant(self.context.types['i32'], 0)
            return self.context.builder.gep(obj, [zero, zero])
        else:
            # Member access: obj.property
            # Handle vector swizzling
            if node.property in ['x', 'y', 'z', 'w']:
                indices = {'x': 0, 'y': 1, 'z': 2, 'w': 3}
                index = ir.Constant(self.context.types['i32'], indices[node.property])
                return self.context.builder.extract_element(obj, index)
            
            # For other member access, return obj
            return obj
    
    def visit_identifier_expression(self, node: IdentifierExpression) -> ir.Value:
        """Visit identifier expression"""
        if node.name in self.context.variables:
            var = self.context.variables[node.name]
            if isinstance(var, ir.GlobalVariable) or isinstance(var.type, ir.PointerType):
                return self.context.builder.load(var)
            else:
                return var
        else:
            # Unknown identifier - return placeholder
            return ir.Constant(self.context.types['i32'], 0)
    
    def visit_literal_expression(self, node: LiteralExpression) -> ir.Value:
        """Visit literal expression"""
        if node.type == 'integer':
            return ir.Constant(self.context.types['i32'], node.value)
        elif node.type == 'float':
            return ir.Constant(self.context.types['f32'], node.value)
        elif node.type == 'boolean':
            return ir.Constant(self.context.types['bool'], node.value)
        elif node.type == 'string':
            # String literals are more complex - simplified here
            return ir.Constant(self.context.types['ptr'], None)
        else:
            return ir.Constant(self.context.types['i32'], 0)
    
    def visit_vector_literal_expression(self, node: VectorLiteralExpression) -> ir.Value:
        """Visit vector literal expression"""
        elements = [self.visit_expression(elem) for elem in node.elements]
        
        # Determine vector type
        if len(elements) == 2:
            vec_type = self.context.types['vec2']
        elif len(elements) == 3:
            vec_type = self.context.types['vec3']
        elif len(elements) == 4:
            vec_type = self.context.types['vec4']
        else:
            # Default to vec4
            vec_type = self.context.types['vec4']
        
        # Create vector
        vec = ir.Constant(vec_type, ir.Undefined)
        for i, elem in enumerate(elements):
            vec = self.context.builder.insert_element(vec, elem, ir.Constant(self.context.types['i32'], i))
        
        return vec
    
    def visit_assignment_expression(self, node: AssignmentExpression) -> ir.Value:
        """Visit assignment expression"""
        right_value = self.visit_expression(node.right)
        
        if isinstance(node.left, IdentifierExpression):
            var_name = node.left.name
            if var_name in self.context.variables:
                var = self.context.variables[var_name]
                
                if node.operator == '=':
                    self.context.builder.store(right_value, var)
                elif node.operator == '+=':
                    current = self.context.builder.load(var)
                    if current.type.is_integer:
                        new_value = self.context.builder.add(current, right_value)
                    else:
                        new_value = self.context.builder.fadd(current, right_value)
                    self.context.builder.store(new_value, var)
                elif node.operator == '-=':
                    current = self.context.builder.load(var)
                    if current.type.is_integer:
                        new_value = self.context.builder.sub(current, right_value)
                    else:
                        new_value = self.context.builder.fsub(current, right_value)
                    self.context.builder.store(new_value, var)
                # Add other compound assignment operators as needed
                
                return right_value
        
        # Handle other types of lvalues
        left_value = self.visit_expression(node.left)
        return right_value
    
    def get_llvm_type(self, ail_type: Type) -> Optional[ir.Type]:
        """Convert AIL type to LLVM type"""
        if ail_type.name in self.context.types:
            base_type = self.context.types[ail_type.name]
            
            if ail_type.is_pointer:
                return ir.PointerType(base_type)
            elif ail_type.is_array and ail_type.array_size:
                # For simplicity, assume array size is a constant
                return ir.ArrayType(base_type, 0)  # Placeholder
            else:
                return base_type
        
        return None
    
    def convert_to_bool(self, value: ir.Value) -> ir.Value:
        """Convert value to boolean"""
        if value.type == self.context.types['bool']:
            return value
        elif value.type.is_integer:
            zero = ir.Constant(value.type, 0)
            return self.context.builder.icmp_signed('!=', value, zero)
        elif value.type.is_float:
            zero = ir.Constant(value.type, 0.0)
            return self.context.builder.fcmp_ordered('!=', value, zero)
        else:
            return value
    
    def get_zero_value(self, llvm_type: ir.Type) -> ir.Value:
        """Get zero value for a given LLVM type"""
        if llvm_type.is_integer:
            return ir.Constant(llvm_type, 0)
        elif llvm_type.is_float:
            return ir.Constant(llvm_type, 0.0)
        elif llvm_type.is_pointer:
            return ir.Constant(llvm_type, None)
        else:
            return ir.Constant(llvm_type, None)
    
    def get_reduction_identity(self, operation: str, llvm_type: ir.Type) -> ir.Value:
        """Get identity value for reduction operation"""
        if operation == 'add':
            return ir.Constant(llvm_type, 0)
        elif operation == 'mul':
            return ir.Constant(llvm_type, 1)
        elif operation == 'min':
            if llvm_type.is_integer:
                return ir.Constant(llvm_type, 2**31 - 1)  # Max int
            else:
                return ir.Constant(llvm_type, float('inf'))
        elif operation == 'max':
            if llvm_type.is_integer:
                return ir.Constant(llvm_type, -2**31)  # Min int
            else:
                return ir.Constant(llvm_type, float('-inf'))
        else:
            return ir.Constant(llvm_type, 0)

# =============================================================================
# Testing Function
# =============================================================================

def test_llvm_codegen():
    """Test the LLVM code generator"""
    from ail_parser import Lexer, Parser
    
    test_code = """
    @kernel @block_size(16, 16, 1)
    func gpu_add(global<f32>* a, global<f32>* b, global<f32>* result, i32 size) -> void {
        i32 idx = thread_idx_x() + block_idx_x() * 16;
        if (idx < size) {
            result[idx] = a[idx] + b[idx];
        }
    }
    
    @vectorize @parallel
    func cpu_multiply(const f32* input, heap f32* output, i32 count) -> void {
        @parallel_for (i32 i in 0..count) {
            output[i] = input[i] * 2.0f;
        }
    }
    
    func vector_ops(vec3 a, vec3 b) -> f32 {
        vec3 c = {a.x + b.x, a.y + b.y, a.z + b.z};
        return dot(c, c);
    }
    
    func main() -> i32 {
        const i32 size = 1024;
        heap f32* data = alloc(null, sizeof(f32) * size);
        
        cpu_multiply(data, data, size);
        
        // Launch GPU kernel
        launch<{64, 1, 1}, {16, 1, 1}> gpu_add(data, data, data, size);
        
        free(data);
        return 0;
    }
    """
    
    print("=== Testing LLVM Code Generator ===\n")
    print("Test code:")
    print(test_code)
    print("\n" + "="*60 + "\n")
    
    # Parse the code
    lexer = Lexer(test_code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    
    try:
        ast = parser.parse()
        print("Parsing successful!\n")
        
        # Generate CPU code
        print("=== CPU Code Generation ===")
        cpu_generator = LLVMCodeGenerator(TargetType.CPU)
        cpu_ir = cpu_generator.generate(ast)
        print(cpu_ir[:1000] + "..." if len(cpu_ir) > 1000 else cpu_ir)
        
        print("\n" + "="*60 + "\n")
        
        # Generate GPU code
        print("=== GPU Code Generation ===")
        gpu_generator = LLVMCodeGenerator(TargetType.GPU_CUDA)
        gpu_ir = gpu_generator.generate(ast)
        print(gpu_ir[:1000] + "..." if len(gpu_ir) > 1000 else gpu_ir)
        
    except Exception as e:
        print(f"Error during code generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llvm_codegen()
