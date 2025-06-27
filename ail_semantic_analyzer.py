#!/usr/bin/env python3
"""
AIL (AI Implementation Language) Semantic Analyzer
Performs type checking, scope resolution, parallel safety analysis, and optimization validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from enum import Enum, auto
import copy

# Import AST nodes from the parser
from ail_parser import *

# =============================================================================
# Type System
# =============================================================================

class TypeKind(Enum):
    PRIMITIVE = auto()
    VECTOR = auto()
    MATRIX = auto()
    POINTER = auto()
    ARRAY = auto()
    TEMPLATE = auto()
    FUNCTION = auto()
    VOID = auto()
    ERROR = auto()

@dataclass
class AILType:
    kind: TypeKind
    name: str
    element_type: Optional['AILType'] = None
    size: Optional[int] = None
    template_args: List['AILType'] = field(default_factory=list)
    storage_qualifier: Optional[str] = None
    is_const: bool = False
    is_atomic: bool = False
    
    def __str__(self) -> str:
        result = ""
        if self.storage_qualifier:
            result += f"{self.storage_qualifier}<"
        if self.is_const:
            result += "const "
        if self.is_atomic:
            result += "atomic "
        
        result += self.name
        
        if self.template_args:
            result += "<" + ", ".join(str(arg) for arg in self.template_args) + ">"
        
        if self.kind == TypeKind.POINTER:
            result += "*"
        elif self.kind == TypeKind.ARRAY:
            result += f"[{self.size if self.size else ''}]"
        
        if self.storage_qualifier:
            result += ">"
        
        return result
    
    def is_numeric(self) -> bool:
        return self.name in ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'f32', 'f64']
    
    def is_integer(self) -> bool:
        return self.name in ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64']
    
    def is_float(self) -> bool:
        return self.name in ['f32', 'f64']
    
    def is_vector(self) -> bool:
        return self.kind == TypeKind.VECTOR or self.name in ['vec2', 'vec3', 'vec4']
    
    def is_matrix(self) -> bool:
        return self.kind == TypeKind.MATRIX or self.name in ['mat2', 'mat3', 'mat4']
    
    def is_gpu_memory(self) -> bool:
        return self.storage_qualifier in ['global', 'shared', 'local', 'constant', 'texture', 'managed']
    
    def is_parallel_safe(self) -> bool:
        """Check if this type is safe for parallel access"""
        return (self.is_const or 
                self.is_atomic or 
                self.storage_qualifier in ['local', 'constant'] or
                self.kind in [TypeKind.PRIMITIVE] and not self.is_pointer())
    
    def is_pointer(self) -> bool:
        return self.kind == TypeKind.POINTER
    
    def get_element_type(self) -> 'AILType':
        """Get the element type for pointers, arrays, or vectors"""
        if self.element_type:
            return self.element_type
        
        # Handle built-in vector/matrix types
        if self.name in ['vec2', 'vec3', 'vec4']:
            return AILType(TypeKind.PRIMITIVE, 'f32')
        elif self.name in ['mat2', 'mat3', 'mat4']:
            return AILType(TypeKind.PRIMITIVE, 'f32')
        
        return self
    
    def can_assign_to(self, other: 'AILType') -> bool:
        """Check if this type can be assigned to another type"""
        if self.kind == TypeKind.ERROR or other.kind == TypeKind.ERROR:
            return True  # Allow assignments involving error types
        
        # Exact match
        if self == other:
            return True
        
        # Numeric conversions
        if self.is_numeric() and other.is_numeric():
            return True  # Allow implicit numeric conversions (with warnings)
        
        # Pointer compatibility
        if self.is_pointer() and other.is_pointer():
            return self.get_element_type().can_assign_to(other.get_element_type())
        
        # Template type compatibility
        if self.storage_qualifier and other.storage_qualifier:
            if self.storage_qualifier == other.storage_qualifier:
                return self.template_args and other.template_args and \
                       self.template_args[0].can_assign_to(other.template_args[0])
        
        return False
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, AILType):
            return False
        return (self.kind == other.kind and 
                self.name == other.name and
                self.element_type == other.element_type and
                self.size == other.size and
                self.template_args == other.template_args and
                self.storage_qualifier == other.storage_qualifier and
                self.is_const == other.is_const and
                self.is_atomic == other.is_atomic)

# =============================================================================
# Symbol Table
# =============================================================================

@dataclass
class Symbol:
    name: str
    type: AILType
    kind: str  # 'variable', 'function', 'parameter', 'type'
    scope_level: int
    is_used: bool = False
    is_modified: bool = False
    declaration_node: Optional[ASTNode] = None
    # Parallel analysis info
    is_thread_local: bool = False
    is_shared_read: bool = False
    is_shared_write: bool = False
    access_pattern: str = 'unknown'  # 'sequential', 'random', 'coalesced'

class SymbolTable:
    def __init__(self):
        self.scopes: List[Dict[str, Symbol]] = [{}]  # Global scope
        self.current_scope_level = 0
        self.built_in_types = self._create_built_in_types()
        self.built_in_functions = self._create_built_in_functions()
    
    def _create_built_in_types(self) -> Dict[str, AILType]:
        """Create built-in types"""
        types = {}
        
        # Primitive types
        for name in ['i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'f32', 'f64', 'bool', 'void']:
            types[name] = AILType(TypeKind.PRIMITIVE, name)
        
        # Vector types
        for name in ['vec2', 'vec3', 'vec4']:
            types[name] = AILType(TypeKind.VECTOR, name, AILType(TypeKind.PRIMITIVE, 'f32'))
        
        # Matrix types
        for name in ['mat2', 'mat3', 'mat4']:
            types[name] = AILType(TypeKind.MATRIX, name, AILType(TypeKind.PRIMITIVE, 'f32'))
        
        return types
    
    def _create_built_in_functions(self) -> Dict[str, Symbol]:
        """Create built-in functions"""
        functions = {}
        
        # Math functions
        f32_type = AILType(TypeKind.PRIMITIVE, 'f32')
        i32_type = AILType(TypeKind.PRIMITIVE, 'i32')
        vec3_type = AILType(TypeKind.VECTOR, 'vec3')
        
        functions['abs'] = Symbol('abs', AILType(TypeKind.FUNCTION, 'abs'), 'function', 0)
        functions['min'] = Symbol('min', AILType(TypeKind.FUNCTION, 'min'), 'function', 0)
        functions['max'] = Symbol('max', AILType(TypeKind.FUNCTION, 'max'), 'function', 0)
        functions['dot'] = Symbol('dot', AILType(TypeKind.FUNCTION, 'dot'), 'function', 0)
        functions['cross'] = Symbol('cross', AILType(TypeKind.FUNCTION, 'cross'), 'function', 0)
        functions['length'] = Symbol('length', AILType(TypeKind.FUNCTION, 'length'), 'function', 0)
        functions['normalize'] = Symbol('normalize', AILType(TypeKind.FUNCTION, 'normalize'), 'function', 0)
        
        # Memory functions
        functions['alloc'] = Symbol('alloc', AILType(TypeKind.FUNCTION, 'alloc'), 'function', 0)
        functions['free'] = Symbol('free', AILType(TypeKind.FUNCTION, 'free'), 'function', 0)
        functions['sizeof'] = Symbol('sizeof', AILType(TypeKind.FUNCTION, 'sizeof'), 'function', 0)
        functions['alignof'] = Symbol('alignof', AILType(TypeKind.FUNCTION, 'alignof'), 'function', 0)
        
        # GPU thread functions
        functions['thread_idx_x'] = Symbol('thread_idx_x', AILType(TypeKind.FUNCTION, 'thread_idx_x'), 'function', 0)
        functions['thread_idx_y'] = Symbol('thread_idx_y', AILType(TypeKind.FUNCTION, 'thread_idx_y'), 'function', 0)
        functions['thread_idx_z'] = Symbol('thread_idx_z', AILType(TypeKind.FUNCTION, 'thread_idx_z'), 'function', 0)
        functions['block_idx_x'] = Symbol('block_idx_x', AILType(TypeKind.FUNCTION, 'block_idx_x'), 'function', 0)
        functions['block_idx_y'] = Symbol('block_idx_y', AILType(TypeKind.FUNCTION, 'block_idx_y'), 'function', 0)
        functions['block_idx_z'] = Symbol('block_idx_z', AILType(TypeKind.FUNCTION, 'block_idx_z'), 'function', 0)
        
        # Warp functions
        functions['warp_reduce_add'] = Symbol('warp_reduce_add', AILType(TypeKind.FUNCTION, 'warp_reduce_add'), 'function', 0)
        functions['warp_shuffle'] = Symbol('warp_shuffle', AILType(TypeKind.FUNCTION, 'warp_shuffle'), 'function', 0)
        functions['warp_ballot'] = Symbol('warp_ballot', AILType(TypeKind.FUNCTION, 'warp_ballot'), 'function', 0)
        
        return functions
    
    def push_scope(self):
        """Enter a new scope"""
        self.scopes.append({})
        self.current_scope_level += 1
    
    def pop_scope(self):
        """Exit current scope"""
        if len(self.scopes) > 1:
            self.scopes.pop()
            self.current_scope_level -= 1
    
    def declare(self, name: str, symbol: Symbol) -> bool:
        """Declare a symbol in current scope"""
        current_scope = self.scopes[-1]
        if name in current_scope:
            return False  # Already declared in this scope
        
        symbol.scope_level = self.current_scope_level
        current_scope[name] = symbol
        return True
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a symbol in all scopes"""
        # Check scopes from innermost to outermost
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        
        # Check built-in functions
        if name in self.built_in_functions:
            return self.built_in_functions[name]
        
        return None
    
    def lookup_type(self, name: str) -> Optional[AILType]:
        """Look up a type"""
        if name in self.built_in_types:
            return self.built_in_types[name]
        
        symbol = self.lookup(name)
        if symbol and symbol.kind == 'type':
            return symbol.type
        
        return None

# =============================================================================
# Semantic Error Types
# =============================================================================

@dataclass
class SemanticError:
    message: str
    node: Optional[ASTNode] = None
    severity: str = 'error'  # 'error', 'warning', 'info'
    category: str = 'general'  # 'type', 'parallel', 'memory', 'gpu', 'annotation'

# =============================================================================
# Semantic Analyzer
# =============================================================================

class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.errors: List[SemanticError] = []
        self.warnings: List[SemanticError] = []
        self.current_function: Optional[FunctionDeclaration] = None
        self.in_kernel = False
        self.in_parallel_loop = False
        self.parallel_variables: Set[str] = set()
        self.shared_reads: Set[str] = set()
        self.shared_writes: Set[str] = set()
    
    def error(self, message: str, node: Optional[ASTNode] = None, category: str = 'general'):
        """Report a semantic error"""
        self.errors.append(SemanticError(message, node, 'error', category))
    
    def warning(self, message: str, node: Optional[ASTNode] = None, category: str = 'general'):
        """Report a semantic warning"""
        self.warnings.append(SemanticError(message, node, 'warning', category))
    
    def analyze(self, program: Program) -> bool:
        """Analyze the entire program"""
        try:
            self.visit_program(program)
            return len(self.errors) == 0
        except Exception as e:
            self.error(f"Internal analyzer error: {e}")
            return False
    
    def visit_program(self, node: Program):
        """Visit program node"""
        # First pass: collect all function and type declarations
        for decl in node.declarations:
            if isinstance(decl, FunctionDeclaration):
                self.declare_function(decl)
        
        # Second pass: analyze function bodies
        for decl in node.declarations:
            self.visit_declaration(decl)
        
        # Check for unused variables
        self.check_unused_symbols()
    
    def declare_function(self, node: FunctionDeclaration):
        """Pre-declare function for forward references"""
        param_types = []
        for param in node.parameters:
            param_type = self.resolve_type(param.type)
            if param_type:
                param_types.append(param_type)
        
        func_type = AILType(TypeKind.FUNCTION, node.name)
        symbol = Symbol(node.name, func_type, 'function', 0, declaration_node=node)
        
        if not self.symbol_table.declare(node.name, symbol):
            self.error(f"Function '{node.name}' is already declared", node)
    
    def visit_declaration(self, node: Declaration):
        """Visit declaration node"""
        if isinstance(node, FunctionDeclaration):
            self.visit_function_declaration(node)
        elif isinstance(node, VariableDeclaration):
            self.visit_variable_declaration(node)
    
    def visit_function_declaration(self, node: FunctionDeclaration):
        """Visit function declaration"""
        old_function = self.current_function
        old_in_kernel = self.in_kernel
        
        self.current_function = node
        self.in_kernel = any(ann.name == 'kernel' for ann in node.annotations)
        
        # Validate annotations
        self.validate_function_annotations(node)
        
        # Enter function scope
        self.symbol_table.push_scope()
        
        try:
            # Add parameters to scope
            for param in node.parameters:
                param_type = self.resolve_type(param.type)
                if param_type:
                    if param.storage_qualifier:
                        param_type.storage_qualifier = param.storage_qualifier
                    
                    symbol = Symbol(param.name, param_type, 'parameter', 
                                  self.symbol_table.current_scope_level, 
                                  declaration_node=param)
                    
                    if not self.symbol_table.declare(param.name, symbol):
                        self.error(f"Parameter '{param.name}' is already declared", param)
            
            # Analyze function body
            self.visit_statement(node.body)
            
            # Check return type consistency
            if node.return_type:
                return_type = self.resolve_type(node.return_type)
                # TODO: Check that all return statements match return type
        
        finally:
            # Exit function scope
            self.symbol_table.pop_scope()
            self.current_function = old_function
            self.in_kernel = old_in_kernel
    
    def validate_function_annotations(self, node: FunctionDeclaration):
        """Validate function annotations"""
        annotation_names = [ann.name for ann in node.annotations]
        
        # Check for conflicting annotations
        gpu_annotations = ['kernel', 'device', 'host', 'host_device']
        gpu_count = sum(1 for ann in annotation_names if ann in gpu_annotations)
        if gpu_count > 1:
            self.error(f"Function cannot have multiple GPU target annotations", node, 'annotation')
        
        # Validate kernel-specific annotations
        if 'kernel' in annotation_names:
            # Kernel functions should have void return type
            if node.return_type and node.return_type.name != 'void':
                self.error("Kernel functions must have void return type", node, 'gpu')
            
            # Check for kernel-specific annotations
            for ann in node.annotations:
                if ann.name == 'block_size':
                    if len(ann.arguments) != 3:
                        self.error("@block_size requires exactly 3 arguments", node, 'gpu')
                elif ann.name == 'occupancy':
                    if len(ann.arguments) != 1:
                        self.error("@occupancy requires exactly 1 argument", node, 'gpu')
        
        # Validate parallel annotations
        if 'parallel' in annotation_names and 'kernel' in annotation_names:
            self.warning("@parallel annotation is redundant on kernel functions", node, 'annotation')
    
    def visit_variable_declaration(self, node: VariableDeclaration):
        """Visit variable declaration"""
        var_type = self.resolve_type(node.type)
        if not var_type:
            return
        
        # Apply storage qualifier
        if node.storage_qualifier:
            var_type.storage_qualifier = node.storage_qualifier
            
            # Validate storage qualifier for context
            if self.in_kernel and node.storage_qualifier in ['stack', 'heap']:
                self.error(f"Cannot use {node.storage_qualifier} storage in kernel function", node, 'gpu')
        
        # Analyze initializer
        if node.initializer:
            init_type = self.visit_expression(node.initializer)
            if init_type and not init_type.can_assign_to(var_type):
                self.error(f"Cannot assign {init_type} to {var_type}", node, 'type')
        
        # Add to symbol table
        symbol = Symbol(node.name, var_type, 'variable', 
                       self.symbol_table.current_scope_level,
                       declaration_node=node)
        
        if not self.symbol_table.declare(node.name, symbol):
            self.error(f"Variable '{node.name}' is already declared", node)
    
    def visit_statement(self, node: Statement) -> Optional[AILType]:
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
        else:
            self.error(f"Unknown statement type: {type(node)}", node)
            return None
    
    def visit_block_statement(self, node: BlockStatement):
        """Visit block statement"""
        self.symbol_table.push_scope()
        try:
            for stmt in node.statements:
                self.visit_statement(stmt)
        finally:
            self.symbol_table.pop_scope()
    
    def visit_expression_statement(self, node: ExpressionStatement):
        """Visit expression statement"""
        return self.visit_expression(node.expression)
    
    def visit_if_statement(self, node: IfStatement):
        """Visit if statement"""
        # Check condition type
        cond_type = self.visit_expression(node.condition)
        if cond_type and cond_type.name != 'bool':
            self.warning(f"If condition has type {cond_type}, expected bool", node, 'type')
        
        # Visit branches
        self.visit_statement(node.then_stmt)
        if node.else_stmt:
            self.visit_statement(node.else_stmt)
    
    def visit_while_statement(self, node: WhileStatement):
        """Visit while statement"""
        # Check condition type
        cond_type = self.visit_expression(node.condition)
        if cond_type and cond_type.name != 'bool':
            self.warning(f"While condition has type {cond_type}, expected bool", node, 'type')
        
        self.visit_statement(node.body)
    
    def visit_for_statement(self, node: ForStatement):
        """Visit for statement"""
        self.symbol_table.push_scope()
        try:
            # Visit initialization
            if node.init:
                if isinstance(node.init, VariableDeclaration):
                    self.visit_variable_declaration(node.init)
                else:
                    self.visit_statement(node.init)
            
            # Visit condition
            if node.condition:
                cond_type = self.visit_expression(node.condition)
                if cond_type and cond_type.name != 'bool':
                    self.warning(f"For condition has type {cond_type}, expected bool", node, 'type')
            
            # Visit update
            if node.update:
                self.visit_expression(node.update)
            
            # Visit body
            self.visit_statement(node.body)
        finally:
            self.symbol_table.pop_scope()
    
    def visit_parallel_for_statement(self, node: ParallelForStatement):
        """Visit parallel for statement"""
        old_in_parallel = self.in_parallel_loop
        old_parallel_vars = self.parallel_variables.copy()
        
        self.in_parallel_loop = True
        self.parallel_variables.add(node.variable)
        
        self.symbol_table.push_scope()
        try:
            # Add loop variable to scope
            var_type = AILType(TypeKind.PRIMITIVE, 'i32')  # Default loop variable type
            symbol = Symbol(node.variable, var_type, 'variable', 
                           self.symbol_table.current_scope_level)
            symbol.is_thread_local = True
            self.symbol_table.declare(node.variable, symbol)
            
            # Analyze range expressions
            self.visit_expression(node.range_expr.start)
            self.visit_expression(node.range_expr.end)
            if node.range_expr.step:
                self.visit_expression(node.range_expr.step)
            
            # Check reduce clauses
            for reduce_clause in node.reduce_clauses:
                reduce_symbol = self.symbol_table.lookup(reduce_clause.variable)
                if not reduce_symbol:
                    self.error(f"Undefined variable in reduce clause: {reduce_clause.variable}", node, 'parallel')
                else:
                    # Mark as shared write
                    reduce_symbol.is_shared_write = True
                    self.shared_writes.add(reduce_clause.variable)
            
            # Analyze body with parallel context
            self.visit_statement(node.body)
            
            # Check for parallel safety violations
            self.check_parallel_safety(node)
            
        finally:
            self.symbol_table.pop_scope()
            self.in_parallel_loop = old_in_parallel
            self.parallel_variables = old_parallel_vars
    
    def visit_kernel_launch_statement(self, node: KernelLaunchStatement):
        """Visit kernel launch statement"""
        # Check if function exists and is a kernel
        func_symbol = self.symbol_table.lookup(node.function_name)
        if not func_symbol:
            self.error(f"Undefined function: {node.function_name}", node, 'gpu')
            return
        
        if func_symbol.declaration_node and isinstance(func_symbol.declaration_node, FunctionDeclaration):
            func_decl = func_symbol.declaration_node
            if not any(ann.name == 'kernel' for ann in func_decl.annotations):
                self.error(f"Function {node.function_name} is not a kernel", node, 'gpu')
        
        # Validate grid and block configurations
        if len(node.grid_config) > 3:
            self.error("Grid configuration cannot have more than 3 dimensions", node, 'gpu')
        if len(node.block_config) > 3:
            self.error("Block configuration cannot have more than 3 dimensions", node, 'gpu')
        
        # Check argument types
        for arg in node.arguments:
            arg_type = self.visit_expression(arg)
            # TODO: Check argument compatibility with kernel parameters
    
    def visit_return_statement(self, node: ReturnStatement):
        """Visit return statement"""
        if node.value:
            return_type = self.visit_expression(node.value)
            # TODO: Check compatibility with function return type
        else:
            # Void return
            if (self.current_function and 
                self.current_function.return_type and 
                self.current_function.return_type.name != 'void'):
                self.error("Non-void function must return a value", node, 'type')
    
    def visit_sync_statement(self, node: SyncStatement):
        """Visit sync statement"""
        if node.sync_type == 'sync_threads' and not self.in_kernel:
            self.error("sync_threads can only be used in kernel functions", node, 'gpu')
        elif node.sync_type == 'sync_warp' and not self.in_kernel:
            self.error("sync_warp can only be used in kernel functions", node, 'gpu')
    
    def visit_expression(self, node: Expression) -> Optional[AILType]:
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
            self.error(f"Unknown expression type: {type(node)}", node)
            return AILType(TypeKind.ERROR, 'error')
    
    def visit_binary_expression(self, node: BinaryExpression) -> AILType:
        """Visit binary expression"""
        left_type = self.visit_expression(node.left)
        right_type = self.visit_expression(node.right)
        
        if not left_type or not right_type:
            return AILType(TypeKind.ERROR, 'error')
        
        # Arithmetic operators
        if node.operator in ['+', '-', '*', '/', '%']:
            if left_type.is_numeric() and right_type.is_numeric():
                # Return the "larger" type (simplified type promotion)
                if left_type.is_float() or right_type.is_float():
                    return AILType(TypeKind.PRIMITIVE, 'f32')
                else:
                    return left_type  # Keep left type for integers
            elif left_type.is_vector() and right_type.is_vector():
                return left_type
            elif left_type.is_vector() and right_type.is_numeric():
                return left_type
            else:
                self.error(f"Invalid binary operation {node.operator} between {left_type} and {right_type}", node, 'type')
                return AILType(TypeKind.ERROR, 'error')
        
        # Comparison operators
        elif node.operator in ['<', '<=', '>', '>=', '==', '!=']:
            if (left_type.is_numeric() and right_type.is_numeric()) or \
               (left_type.name == right_type.name):
                return AILType(TypeKind.PRIMITIVE, 'bool')
            else:
                self.error(f"Cannot compare {left_type} and {right_type}", node, 'type')
                return AILType(TypeKind.ERROR, 'error')
        
        # Logical operators
        elif node.operator in ['&&', '||']:
            if left_type.name == 'bool' and right_type.name == 'bool':
                return AILType(TypeKind.PRIMITIVE, 'bool')
            else:
                self.error(f"Logical operators require bool operands, got {left_type} and {right_type}", node, 'type')
                return AILType(TypeKind.ERROR, 'error')
        
        # Bitwise operators
        elif node.operator in ['&', '|', '^', '<<', '>>']:
            if left_type.is_integer() and right_type.is_integer():
                return left_type
            else:
                self.error(f"Bitwise operators require integer operands", node, 'type')
                return AILType(TypeKind.ERROR, 'error')
        
        else:
            self.error(f"Unknown binary operator: {node.operator}", node)
            return AILType(TypeKind.ERROR, 'error')
    
    def visit_unary_expression(self, node: UnaryExpression) -> AILType:
        """Visit unary expression"""
        operand_type = self.visit_expression(node.operand)
        if not operand_type:
            return AILType(TypeKind.ERROR, 'error')
        
        if node.operator in ['+', '-']:
            if operand_type.is_numeric() or operand_type.is_vector():
                return operand_type
        elif node.operator == '!':
            if operand_type.name == 'bool':
                return operand_type
            else:
                self.error(f"Logical NOT requires bool operand, got {operand_type}", node, 'type')
        elif node.operator == '~':
            if operand_type.is_integer():
                return operand_type
            else:
                self.error(f"Bitwise NOT requires integer operand, got {operand_type}", node, 'type')
        elif node.operator == '*':
            # Dereference
            if operand_type.is_pointer():
                return operand_type.get_element_type()
            else:
                self.error(f"Cannot dereference non-pointer type {operand_type}", node, 'type')
        elif node.operator == '&':
            # Address-of
            return AILType(TypeKind.POINTER, operand_type.name, operand_type)
        
        return AILType(TypeKind.ERROR, 'error')
    
    def visit_call_expression(self, node: CallExpression) -> AILType:
        """Visit call expression"""
        if isinstance(node.function, IdentifierExpression):
            func_name = node.function.name
            func_symbol = self.symbol_table.lookup(func_name)
            
            if not func_symbol:
                self.error(f"Undefined function: {func_name}", node)
                return AILType(TypeKind.ERROR, 'error')
            
            # Check argument count and types
            # TODO: Implement proper function signature checking
            
            # Handle built-in functions
            if func_name in ['dot', 'cross', 'length', 'normalize']:
                # Vector functions
                if node.arguments:
                    arg_type = self.visit_expression(node.arguments[0])
                    if func_name == 'dot' and len(node.arguments) == 2:
                        return AILType(TypeKind.PRIMITIVE, 'f32')
                    elif func_name == 'cross':
                        return AILType(TypeKind.VECTOR, 'vec3')
                    elif func_name == 'length':
                        return AILType(TypeKind.PRIMITIVE, 'f32')
                    elif func_name == 'normalize':
                        return arg_type
            
            elif func_name in ['thread_idx_x', 'thread_idx_y', 'thread_idx_z', 
                              'block_idx_x', 'block_idx_y', 'block_idx_z']:
                if not self.in_kernel:
                    self.error(f"{func_name} can only be used in kernel functions", node, 'gpu')
                return AILType(TypeKind.PRIMITIVE, 'i32')
            
            elif func_name.startswith('warp_'):
                if not self.in_kernel:
                    self.error(f"{func_name} can only be used in kernel functions", node, 'gpu')
                # Return type depends on specific warp function
                if 'reduce' in func_name:
                    if node.arguments:
                        return self.visit_expression(node.arguments[0])
                return AILType(TypeKind.PRIMITIVE, 'i32')
            
            # For other functions, return a generic type
            return AILType(TypeKind.PRIMITIVE, 'i32')  # Placeholder
        
        else:
            # Function pointer call or complex expression
            func_type = self.visit_expression(node.function)
            return AILType(TypeKind.ERROR, 'error')  # Placeholder
    
    def visit_member_expression(self, node: MemberExpression) -> AILType:
        """Visit member expression"""
        object_type = self.visit_expression(node.object)
        if not object_type:
            return AILType(TypeKind.ERROR, 'error')
        
        if node.computed:
            # Array access: obj[index]
            if object_type.kind in [TypeKind.ARRAY, TypeKind.POINTER]:
                return object_type.get_element_type()
            else:
                self.error(f"Cannot index non-array type {object_type}", node, 'type')
                return AILType(TypeKind.ERROR, 'error')
        else:
            # Member access: obj.property
            if object_type.is_vector():
                # Vector swizzling
                if node.property in ['x', 'y', 'z', 'w', 'r', 'g', 'b', 'a']:
                    return AILType(TypeKind.PRIMITIVE, 'f32')
            
            # TODO: Handle struct member access
            return AILType(TypeKind.ERROR, 'error')
    
    def visit_identifier_expression(self, node: IdentifierExpression) -> AILType:
        """Visit identifier expression"""
        symbol = self.symbol_table.lookup(node.name)
        if not symbol:
            self.error(f"Undefined identifier: {node.name}", node)
            return AILType(TypeKind.ERROR, 'error')
        
        # Mark as used
        symbol.is_used = True
        
        # Check parallel access patterns
        if self.in_parallel_loop and symbol.kind == 'variable':
            if node.name not in self.parallel_variables:
                if not symbol.is_parallel_safe():
                    self.shared_reads.add(node.name)
                    symbol.is_shared_read = True
        
        return symbol.type
    
    def visit_literal_expression(self, node: LiteralExpression) -> AILType:
        """Visit literal expression"""
        if node.type == 'integer':
            return AILType(TypeKind.PRIMITIVE, 'i32')
        elif node.type == 'float':
            return AILType(TypeKind.PRIMITIVE, 'f32')
        elif node.type == 'string':
            return AILType(TypeKind.POINTER, 'char', AILType(TypeKind.PRIMITIVE, 'char'))
        elif node.type == 'boolean':
            return AILType(TypeKind.PRIMITIVE, 'bool')
        elif node.type == 'null':
            return AILType(TypeKind.POINTER, 'void', AILType(TypeKind.VOID, 'void'))
        else:
            return AILType(TypeKind.ERROR, 'error')
    
    def visit_vector_literal_expression(self, node: VectorLiteralExpression) -> AILType:
        """Visit vector literal expression"""
        if not node.elements:
            return AILType(TypeKind.ERROR, 'error')
        
        # Check element types
        element_types = []
        for elem in node.elements:
            elem_type = self.visit_expression(elem)
            if elem_type:
                element_types.append(elem_type)
        
        # Determine vector type based on element count
        count = len(node.elements)
        if count == 2:
            return AILType(TypeKind.VECTOR, 'vec2')
        elif count == 3:
            return AILType(TypeKind.VECTOR, 'vec3')
        elif count == 4:
            return AILType(TypeKind.VECTOR, 'vec4')
        else:
            self.error(f"Vector literals must have 2-4 elements, got {count}", node, 'type')
            return AILType(TypeKind.ERROR, 'error')
    
    def visit_assignment_expression(self, node: AssignmentExpression) -> AILType:
        """Visit assignment expression"""
        left_type = self.visit_expression(node.left)
        right_type = self.visit_expression(node.right)
        
        if not left_type or not right_type:
            return AILType(TypeKind.ERROR, 'error')
        
        # Mark left side as modified if it's an identifier
        if isinstance(node.left, IdentifierExpression):
            symbol = self.symbol_table.lookup(node.left.name)
            if symbol:
                symbol.is_modified = True
                
                # Check parallel write safety
                if self.in_parallel_loop and symbol.kind == 'variable':
                    if node.left.name not in self.parallel_variables:
                        self.shared_writes.add(node.left.name)
                        symbol.is_shared_write = True
        
        # Check assignment compatibility
        if not right_type.can_assign_to(left_type):
            self.error(f"Cannot assign {right_type} to {left_type}", node, 'type')
        
        return left_type
    
    def resolve_type(self, type_node: Type) -> Optional[AILType]:
        """Resolve a type node to an AILType"""
        base_type = self.symbol_table.lookup_type(type_node.name)
        if not base_type:
            self.error(f"Unknown type: {type_node.name}")
            return None
        
        result_type = copy.copy(base_type)
        
        # Handle template arguments
        if type_node.template_args:
            template_types = []
            for template_arg in type_node.template_args:
                template_type = self.resolve_type(template_arg)
                if template_type:
                    template_types.append(template_type)
            result_type.template_args = template_types
        
        # Handle pointer
        if type_node.is_pointer:
            result_type = AILType(TypeKind.POINTER, result_type.name, result_type)
        
        # Handle array
        if type_node.is_array:
            size = None
            if type_node.array_size:
                # TODO: Evaluate array size expression
                size = 0  # Placeholder
            result_type = AILType(TypeKind.ARRAY, result_type.name, result_type, size)
        
        return result_type
    
    def check_parallel_safety(self, node: ParallelForStatement):
        """Check parallel safety for a parallel for loop"""
        # Check for data races
        race_variables = self.shared_reads.intersection(self.shared_writes)
        for var in race_variables:
            symbol = self.symbol_table.lookup(var)
            if symbol and not symbol.is_parallel_safe():
                self.error(f"Potential data race on variable '{var}' in parallel loop", node, 'parallel')
        
        # Check for non-atomic shared writes
        for var in self.shared_writes:
            if var not in [rc.variable for rc in node.reduce_clauses]:
                symbol = self.symbol_table.lookup(var)
                if symbol and not symbol.type.is_atomic:
                    self.warning(f"Non-atomic write to shared variable '{var}' in parallel loop", node, 'parallel')
    
    def check_unused_symbols(self):
        """Check for unused symbols and report warnings"""
        for scope in self.symbol_table.scopes:
            for name, symbol in scope.items():
                if not symbol.is_used and symbol.kind in ['variable', 'parameter']:
                    self.warning(f"Unused {symbol.kind} '{name}'", symbol.declaration_node)
    
    def print_diagnostics(self):
        """Print all diagnostics"""
        print("=== Semantic Analysis Results ===\n")
        
        if self.errors:
            print("ERRORS:")
            for error in self.errors:
                print(f"  {error.category.upper()}: {error.message}")
            print()
        
        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning.category.upper()}: {warning.message}")
            print()
        
        if not self.errors and not self.warnings:
            print("No issues found!")

# =============================================================================
# Testing Function
# =============================================================================

def test_semantic_analyzer():
    """Test the semantic analyzer"""
    from ail_parser import Lexer, Parser
    
    # Test code with various semantic issues
    test_code = """
    @kernel @block_size(16, 16, 1)
    func gpu_kernel(global<f32>* data, i32 count) -> void {
        i32 idx = thread_idx_x() + block_idx_x() * 16;
        if (idx < count) {
            data[idx] = data[idx] * 2.0f;
        }
        sync_threads;
    }
    
    @parallel @vectorize
    func parallel_sum(const f32* input, heap f32* output, i32 size) -> void {
        f32 total = 0.0f;
        
        @parallel_for (i32 i in 0..size) 
        reduce(total: add) {
            total += input[i];
        }
        
        *output = total;
    }
    
    func test_function(i32 param) -> f32 {
        heap f32* data = alloc(null, sizeof(f32) * 1024);
        i32 unused_var = 42;  // Should warn about unused
        
        // Type error: assigning string to int
        i32 wrong_type = "hello";
        
        // Parallel safety issue
        f32 shared_var = 0.0f;
        @parallel_for (i32 i in 0..10) {
            shared_var += data[i];  // Data race
        }
        
        // GPU function call from CPU function
        launch<{64, 1, 1}, {16, 1, 1}> gpu_kernel(data, 1024);
        
        free(data);
        return shared_var;
    }
    
    func main() -> i32 {
        test_function(42);
        return 0;
    }
    """
    
    print("=== Testing Semantic Analyzer ===\n")
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
        
        # Analyze semantics
        analyzer = SemanticAnalyzer()
        success = analyzer.analyze(ast)
        
        analyzer.print_diagnostics()
        
        print(f"\nAnalysis {'PASSED' if success else 'FAILED'}")
        print(f"Errors: {len(analyzer.errors)}")
        print(f"Warnings: {len(analyzer.warnings)}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_semantic_analyzer()
