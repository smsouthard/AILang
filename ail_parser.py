#!/usr/bin/env python3
"""
AIL (AI Implementation Language) Parser
A recursive descent parser for the AI-optimized language specification.
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from abc import ABC, abstractmethod

# =============================================================================
# Token Types and Lexer
# =============================================================================

class TokenType(Enum):
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BOOLEAN = auto()
    
    # Identifiers and Keywords
    IDENTIFIER = auto()
    
    # Type keywords
    I8 = auto()
    I16 = auto()
    I32 = auto()
    I64 = auto()
    U8 = auto()
    U16 = auto()
    U32 = auto()
    U64 = auto()
    F32 = auto()
    F64 = auto()
    BOOL = auto()
    VOID = auto()
    VEC2 = auto()
    VEC3 = auto()
    VEC4 = auto()
    MAT2 = auto()
    MAT3 = auto()
    MAT4 = auto()
    AUTO = auto()
    
    # Storage qualifiers
    STACK = auto()
    HEAP = auto()
    ARENA = auto()
    CONST = auto()
    VOLATILE = auto()
    GLOBAL = auto()
    SHARED = auto()
    LOCAL = auto()
    CONSTANT = auto()
    TEXTURE = auto()
    MANAGED = auto()
    ATOMIC = auto()
    
    # Control flow
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    LOOP = auto()
    BREAK = auto()
    CONTINUE = auto()
    RETURN = auto()
    SWITCH = auto()
    CASE = auto()
    DEFAULT = auto()
    
    # Function keywords
    FUNC = auto()
    EXTERN = auto()
    INLINE = auto()
    KERNEL = auto()
    DEVICE = auto()
    HOST = auto()
    HOST_DEVICE = auto()
    
    # Parallel keywords
    PARALLEL_FOR = auto()
    LAUNCH = auto()
    SYNC = auto()
    SYNC_THREADS = auto()
    SYNC_WARP = auto()
    CRITICAL = auto()
    REDUCE = auto()
    IN = auto()
    STEP = auto()
    
    # Memory operations
    ALLOC = auto()
    FREE = auto()
    SIZEOF = auto()
    ALIGNOF = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    MULTIPLY_ASSIGN = auto()
    DIVIDE_ASSIGN = auto()
    MODULO_ASSIGN = auto()
    
    # Logical operators
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # Bitwise operators
    BIT_AND = auto()
    BIT_OR = auto()
    BIT_XOR = auto()
    BIT_NOT = auto()
    LEFT_SHIFT = auto()
    RIGHT_SHIFT = auto()
    
    # Comparison operators
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    
    # Punctuation
    SEMICOLON = auto()
    COMMA = auto()
    DOT = auto()
    ARROW = auto()
    DOUBLE_DOT = auto()
    
    # Brackets
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    LEFT_BRACKET = auto()
    RIGHT_BRACKET = auto()
    LEFT_ANGLE = auto()
    RIGHT_ANGLE = auto()
    
    # Special
    AT = auto()
    HASH = auto()
    DOLLAR = auto()
    
    # Literals
    TRUE = auto()
    FALSE = auto()
    NULL = auto()
    
    # End of file
    EOF = auto()
    
    # Error
    ERROR = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        # Keywords mapping
        self.keywords = {
            # Types
            'i8': TokenType.I8, 'i16': TokenType.I16, 'i32': TokenType.I32, 'i64': TokenType.I64,
            'u8': TokenType.U8, 'u16': TokenType.U16, 'u32': TokenType.U32, 'u64': TokenType.U64,
            'f32': TokenType.F32, 'f64': TokenType.F64, 'bool': TokenType.BOOL, 'void': TokenType.VOID,
            'vec2': TokenType.VEC2, 'vec3': TokenType.VEC3, 'vec4': TokenType.VEC4,
            'mat2': TokenType.MAT2, 'mat3': TokenType.MAT3, 'mat4': TokenType.MAT4,
            'auto': TokenType.AUTO,
            
            # Storage qualifiers
            'stack': TokenType.STACK, 'heap': TokenType.HEAP, 'arena': TokenType.ARENA,
            'const': TokenType.CONST, 'volatile': TokenType.VOLATILE,
            'global': TokenType.GLOBAL, 'shared': TokenType.SHARED, 'local': TokenType.LOCAL,
            'constant': TokenType.CONSTANT, 'texture': TokenType.TEXTURE, 'managed': TokenType.MANAGED,
            'atomic': TokenType.ATOMIC,
            
            # Control flow
            'if': TokenType.IF, 'else': TokenType.ELSE, 'while': TokenType.WHILE,
            'for': TokenType.FOR, 'loop': TokenType.LOOP, 'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE, 'return': TokenType.RETURN,
            'switch': TokenType.SWITCH, 'case': TokenType.CASE, 'default': TokenType.DEFAULT,
            
            # Functions
            'func': TokenType.FUNC, 'extern': TokenType.EXTERN, 'inline': TokenType.INLINE,
            'kernel': TokenType.KERNEL, 'device': TokenType.DEVICE, 'host': TokenType.HOST,
            'host_device': TokenType.HOST_DEVICE,
            
            # Parallel
            'parallel_for': TokenType.PARALLEL_FOR, 'launch': TokenType.LAUNCH,
            'sync': TokenType.SYNC, 'sync_threads': TokenType.SYNC_THREADS, 'sync_warp': TokenType.SYNC_WARP,
            'critical': TokenType.CRITICAL, 'reduce': TokenType.REDUCE,
            'in': TokenType.IN, 'step': TokenType.STEP,
            
            # Memory
            'alloc': TokenType.ALLOC, 'free': TokenType.FREE, 'sizeof': TokenType.SIZEOF, 'alignof': TokenType.ALIGNOF,
            
            # Literals
            'true': TokenType.TRUE, 'false': TokenType.FALSE, 'null': TokenType.NULL,
        }
    
    def current_char(self) -> Optional[str]:
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        pos = self.position + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self) -> None:
        if self.position < len(self.source):
            if self.source[self.position] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.position += 1
    
    def skip_whitespace(self) -> None:
        while self.current_char() and self.current_char().isspace():
            self.advance()
    
    def skip_comment(self) -> None:
        if self.current_char() == '/' and self.peek_char() == '/':
            # Single line comment
            while self.current_char() and self.current_char() != '\n':
                self.advance()
        elif self.current_char() == '/' and self.peek_char() == '*':
            # Multi-line comment
            self.advance()  # Skip '/'
            self.advance()  # Skip '*'
            while self.current_char():
                if self.current_char() == '*' and self.peek_char() == '/':
                    self.advance()  # Skip '*'
                    self.advance()  # Skip '/'
                    break
                self.advance()
    
    def read_number(self) -> Token:
        start_line, start_column = self.line, self.column
        number_str = ""
        is_float = False
        
        # Read digits
        while self.current_char() and (self.current_char().isdigit() or self.current_char() in '._'):
            if self.current_char() == '.':
                if is_float:  # Second dot
                    break
                is_float = True
            number_str += self.current_char()
            self.advance()
        
        # Check for type suffix (f, i32, etc.)
        suffix = ""
        if self.current_char() and self.current_char().isalpha():
            while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
                suffix += self.current_char()
                self.advance()
        
        full_number = number_str + suffix
        token_type = TokenType.FLOAT if is_float else TokenType.INTEGER
        
        return Token(token_type, full_number, start_line, start_column)
    
    def read_string(self) -> Token:
        start_line, start_column = self.line, self.column
        quote_char = self.current_char()
        string_value = ""
        
        self.advance()  # Skip opening quote
        
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == '\\':
                self.advance()
                if self.current_char():
                    # Handle escape sequences
                    escape_chars = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}
                    string_value += escape_chars.get(self.current_char(), self.current_char())
                    self.advance()
            else:
                string_value += self.current_char()
                self.advance()
        
        if self.current_char() == quote_char:
            self.advance()  # Skip closing quote
        
        return Token(TokenType.STRING, string_value, start_line, start_column)
    
    def read_identifier(self) -> Token:
        start_line, start_column = self.line, self.column
        identifier = ""
        
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            identifier += self.current_char()
            self.advance()
        
        # Check if it's a keyword
        token_type = self.keywords.get(identifier, TokenType.IDENTIFIER)
        return Token(token_type, identifier, start_line, start_column)
    
    def tokenize(self) -> List[Token]:
        while self.position < len(self.source):
            self.skip_whitespace()
            
            if not self.current_char():
                break
            
            # Skip comments
            if self.current_char() == '/' and self.peek_char() in ['/', '*']:
                self.skip_comment()
                continue
            
            start_line, start_column = self.line, self.column
            char = self.current_char()
            
            # Numbers
            if char.isdigit():
                self.tokens.append(self.read_number())
                continue
            
            # Strings
            if char in ['"', "'"]:
                self.tokens.append(self.read_string())
                continue
            
            # Identifiers and keywords
            if char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
                continue
            
            # Two-character operators
            if char == '=' and self.peek_char() == '=':
                self.tokens.append(Token(TokenType.EQUAL, '==', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '!' and self.peek_char() == '=':
                self.tokens.append(Token(TokenType.NOT_EQUAL, '!=', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '<' and self.peek_char() == '=':
                self.tokens.append(Token(TokenType.LESS_EQUAL, '<=', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '>' and self.peek_char() == '=':
                self.tokens.append(Token(TokenType.GREATER_EQUAL, '>=', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '&' and self.peek_char() == '&':
                self.tokens.append(Token(TokenType.AND, '&&', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '|' and self.peek_char() == '|':
                self.tokens.append(Token(TokenType.OR, '||', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '<' and self.peek_char() == '<':
                self.tokens.append(Token(TokenType.LEFT_SHIFT, '<<', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '>' and self.peek_char() == '>':
                self.tokens.append(Token(TokenType.RIGHT_SHIFT, '>>', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '+' and self.peek_char() == '=':
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, '+=', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '-' and self.peek_char() == '=':
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, '-=', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '*' and self.peek_char() == '=':
                self.tokens.append(Token(TokenType.MULTIPLY_ASSIGN, '*=', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '/' and self.peek_char() == '=':
                self.tokens.append(Token(TokenType.DIVIDE_ASSIGN, '/=', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '-' and self.peek_char() == '>':
                self.tokens.append(Token(TokenType.ARROW, '->', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            if char == '.' and self.peek_char() == '.':
                self.tokens.append(Token(TokenType.DOUBLE_DOT, '..', start_line, start_column))
                self.advance()
                self.advance()
                continue
            
            # Single character tokens
            single_char_tokens = {
                '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.MULTIPLY, '/': TokenType.DIVIDE,
                '%': TokenType.MODULO, '=': TokenType.ASSIGN, '!': TokenType.NOT,
                '&': TokenType.BIT_AND, '|': TokenType.BIT_OR, '^': TokenType.BIT_XOR, '~': TokenType.BIT_NOT,
                '<': TokenType.LESS, '>': TokenType.GREATER,
                ';': TokenType.SEMICOLON, ',': TokenType.COMMA, '.': TokenType.DOT,
                '(': TokenType.LEFT_PAREN, ')': TokenType.RIGHT_PAREN,
                '{': TokenType.LEFT_BRACE, '}': TokenType.RIGHT_BRACE,
                '[': TokenType.LEFT_BRACKET, ']': TokenType.RIGHT_BRACKET,
                '@': TokenType.AT, '#': TokenType.HASH, '$': TokenType.DOLLAR,
            }
            
            if char in single_char_tokens:
                self.tokens.append(Token(single_char_tokens[char], char, start_line, start_column))
                self.advance()
                continue
            
            # Unknown character
            self.tokens.append(Token(TokenType.ERROR, char, start_line, start_column))
            self.advance()
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens

# =============================================================================
# Abstract Syntax Tree (AST) Nodes
# =============================================================================

class ASTNode(ABC):
    """Base class for all AST nodes"""
    pass

@dataclass
class Program(ASTNode):
    declarations: List['Declaration']

@dataclass
class Declaration(ASTNode):
    pass

@dataclass
class FunctionDeclaration(Declaration):
    annotations: List['Annotation']
    name: str
    parameters: List['Parameter']
    return_type: Optional['Type']
    body: 'BlockStatement'

@dataclass
class VariableDeclaration(Declaration):
    storage_qualifier: Optional[str]
    type: 'Type'
    name: str
    initializer: Optional['Expression']

@dataclass
class Parameter(ASTNode):
    storage_qualifier: Optional[str]
    type: 'Type'
    name: str

@dataclass
class Type(ASTNode):
    name: str
    template_args: List['Type']
    is_pointer: bool
    is_array: bool
    array_size: Optional['Expression']

@dataclass
class Annotation(ASTNode):
    name: str
    arguments: List['AnnotationArgument']

@dataclass
class AnnotationArgument(ASTNode):
    name: Optional[str]
    value: 'Expression'

# Statements
@dataclass
class Statement(ASTNode):
    pass

@dataclass
class BlockStatement(Statement):
    statements: List[Statement]

@dataclass
class ExpressionStatement(Statement):
    expression: 'Expression'

@dataclass
class IfStatement(Statement):
    condition: 'Expression'
    then_stmt: Statement
    else_stmt: Optional[Statement]

@dataclass
class WhileStatement(Statement):
    condition: 'Expression'
    body: Statement

@dataclass
class ForStatement(Statement):
    init: Optional[Declaration]
    condition: Optional['Expression']
    update: Optional['Expression']
    body: Statement

@dataclass
class ParallelForStatement(Statement):
    variable: str
    range_expr: 'RangeExpression'
    reduce_clauses: List['ReduceClause']
    body: Statement

@dataclass
class KernelLaunchStatement(Statement):
    grid_config: List['Expression']
    block_config: List['Expression']
    function_name: str
    arguments: List['Expression']

@dataclass
class ReturnStatement(Statement):
    value: Optional['Expression']

@dataclass
class SyncStatement(Statement):
    sync_type: str  # 'sync', 'sync_threads', 'sync_warp'

@dataclass
class RangeExpression(ASTNode):
    start: 'Expression'
    end: 'Expression'
    step: Optional['Expression']

@dataclass
class ReduceClause(ASTNode):
    variable: str
    operation: str  # 'add', 'mul', 'min', 'max', etc.

# Expressions
@dataclass
class Expression(ASTNode):
    pass

@dataclass
class BinaryExpression(Expression):
    left: Expression
    operator: str
    right: Expression

@dataclass
class UnaryExpression(Expression):
    operator: str
    operand: Expression

@dataclass
class CallExpression(Expression):
    function: Expression
    arguments: List[Expression]

@dataclass
class MemberExpression(Expression):
    object: Expression
    property: str
    computed: bool  # True for a[b], False for a.b

@dataclass
class IdentifierExpression(Expression):
    name: str

@dataclass
class LiteralExpression(Expression):
    value: Union[int, float, str, bool]
    type: str

@dataclass
class VectorLiteralExpression(Expression):
    elements: List[Expression]

@dataclass
class AssignmentExpression(Expression):
    left: Expression
    operator: str
    right: Expression

# =============================================================================
# Parser
# =============================================================================

class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"{message} at line {token.line}, column {token.column}")

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
    
    def current_token(self) -> Token:
        if self.position >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[self.position]
    
    def peek_token(self, offset: int = 1) -> Token:
        pos = self.position + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[pos]
    
    def advance(self) -> Token:
        token = self.current_token()
        if self.position < len(self.tokens) - 1:
            self.position += 1
        return token
    
    def match(self, *token_types: TokenType) -> bool:
        return self.current_token().type in token_types
    
    def consume(self, token_type: TokenType, message: str = "") -> Token:
        if self.current_token().type == token_type:
            return self.advance()
        else:
            if not message:
                message = f"Expected {token_type}, got {self.current_token().type}"
            raise ParseError(message, self.current_token())
    
    def parse(self) -> Program:
        """Parse the entire program"""
        declarations = []
        
        while not self.match(TokenType.EOF):
            try:
                decl = self.parse_declaration()
                if decl:
                    declarations.append(decl)
            except ParseError as e:
                print(f"Parse error: {e}")
                # Skip to next statement for error recovery
                self.skip_to_next_statement()
        
        return Program(declarations)
    
    def skip_to_next_statement(self):
        """Skip tokens until we find a likely statement start"""
        while not self.match(TokenType.EOF, TokenType.SEMICOLON, TokenType.RIGHT_BRACE, TokenType.FUNC):
            self.advance()
        if self.match(TokenType.SEMICOLON):
            self.advance()
    
    def parse_declaration(self) -> Optional[Declaration]:
        """Parse a top-level declaration"""
        # Parse annotations first
        annotations = []
        while self.match(TokenType.AT):
            annotations.append(self.parse_annotation())
        
        # Check for function declaration
        if self.match(TokenType.FUNC, TokenType.EXTERN, TokenType.INLINE, TokenType.KERNEL, TokenType.DEVICE, TokenType.HOST, TokenType.HOST_DEVICE):
            return self.parse_function_declaration(annotations)
        
        # Otherwise, it's a variable declaration
        return self.parse_variable_declaration()
    
    def parse_function_declaration(self, annotations: List[Annotation]) -> FunctionDeclaration:
        """Parse a function declaration"""
        # Skip function modifiers
        while self.match(TokenType.EXTERN, TokenType.INLINE, TokenType.KERNEL, TokenType.DEVICE, TokenType.HOST, TokenType.HOST_DEVICE):
            self.advance()
        
        self.consume(TokenType.FUNC, "Expected 'func'")
        
        # Function name
        name_token = self.consume(TokenType.IDENTIFIER, "Expected function name")
        name = name_token.value
        
        # Parameters
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after function name")
        parameters = []
        
        if not self.match(TokenType.RIGHT_PAREN):
            parameters.append(self.parse_parameter())
            while self.match(TokenType.COMMA):
                self.advance()
                parameters.append(self.parse_parameter())
        
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after parameters")
        
        # Return type
        return_type = None
        if self.match(TokenType.ARROW):
            self.advance()
            return_type = self.parse_type()
        
        # Function body
        body = self.parse_block_statement()
        
        return FunctionDeclaration(annotations, name, parameters, return_type, body)
    
    def parse_parameter(self) -> Parameter:
        """Parse a function parameter"""
        storage_qualifier = None
        if self.match(TokenType.CONST, TokenType.STACK, TokenType.HEAP, TokenType.ARENA, TokenType.GLOBAL, TokenType.SHARED, TokenType.LOCAL):
            storage_qualifier = self.advance().value
        
        param_type = self.parse_type()
        name_token = self.consume(TokenType.IDENTIFIER, "Expected parameter name")
        
        return Parameter(storage_qualifier, param_type, name_token.value)
    
    def parse_variable_declaration(self) -> VariableDeclaration:
        """Parse a variable declaration"""
        storage_qualifier = None
        if self.match(TokenType.CONST, TokenType.STACK, TokenType.HEAP, TokenType.ARENA, TokenType.GLOBAL, TokenType.SHARED, TokenType.LOCAL, TokenType.VOLATILE, TokenType.ATOMIC):
            storage_qualifier = self.advance().value
        
        var_type = self.parse_type()
        name_token = self.consume(TokenType.IDENTIFIER, "Expected variable name")
        
        initializer = None
        if self.match(TokenType.ASSIGN):
            self.advance()
            initializer = self.parse_expression()
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after variable declaration")
        
        return VariableDeclaration(storage_qualifier, var_type, name_token.value, initializer)
    
    def parse_type(self) -> Type:
        """Parse a type specification"""
        # Handle template types like global<f32>
        if self.match(TokenType.GLOBAL, TokenType.SHARED, TokenType.LOCAL, TokenType.CONSTANT, TokenType.TEXTURE, TokenType.MANAGED, TokenType.ATOMIC):
            base_name = self.advance().value
            template_args = []
            
            if self.match(TokenType.LESS):
                self.advance()
                template_args.append(self.parse_type())
                while self.match(TokenType.COMMA):
                    self.advance()
                    template_args.append(self.parse_type())
                self.consume(TokenType.GREATER, "Expected '>' after template arguments")
            
            return Type(base_name, template_args, False, False, None)
        
        # Basic types
        if self.match(TokenType.I8, TokenType.I16, TokenType.I32, TokenType.I64,
                     TokenType.U8, TokenType.U16, TokenType.U32, TokenType.U64,
                     TokenType.F32, TokenType.F64, TokenType.BOOL, TokenType.VOID,
                     TokenType.VEC2, TokenType.VEC3, TokenType.VEC4,
                     TokenType.MAT2, TokenType.MAT3, TokenType.MAT4, TokenType.AUTO):
            type_name = self.advance().value
        elif self.match(TokenType.IDENTIFIER):
            type_name = self.advance().value
        else:
            raise ParseError("Expected type name", self.current_token())
        
        # Check for pointer
        is_pointer = False
        if self.match(TokenType.MULTIPLY):
            is_pointer = True
            self.advance()
        
        # Check for array
        is_array = False
        array_size = None
        if self.match(TokenType.LEFT_BRACKET):
            is_array = True
            self.advance()
            if not self.match(TokenType.RIGHT_BRACKET):
                array_size = self.parse_expression()
            self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after array size")
        
        return Type(type_name, [], is_pointer, is_array, array_size)
    
    def parse_annotation(self) -> Annotation:
        """Parse an annotation like @vectorize or @block_size(16, 16, 1)"""
        self.consume(TokenType.AT, "Expected '@'")
        name_token = self.consume(TokenType.IDENTIFIER, "Expected annotation name")
        
        arguments = []
        if self.match(TokenType.LEFT_PAREN):
            self.advance()
            if not self.match(TokenType.RIGHT_PAREN):
                arguments.append(self.parse_annotation_argument())
                while self.match(TokenType.COMMA):
                    self.advance()
                    arguments.append(self.parse_annotation_argument())
            self.consume(TokenType.RIGHT_PAREN, "Expected ')' after annotation arguments")
        
        return Annotation(name_token.value, arguments)
    
    def parse_annotation_argument(self) -> AnnotationArgument:
        """Parse an annotation argument"""
        # Check for named argument (name = value)
        if self.match(TokenType.IDENTIFIER) and self.peek_token().type == TokenType.ASSIGN:
            name = self.advance().value
            self.advance()  # consume '='
            value = self.parse_expression()
            return AnnotationArgument(name, value)
        else:
            # Positional argument
            value = self.parse_expression()
            return AnnotationArgument(None, value)
    
    def parse_block_statement(self) -> BlockStatement:
        """Parse a block statement { ... }"""
        self.consume(TokenType.LEFT_BRACE, "Expected '{'")
        
        statements = []
        while not self.match(TokenType.RIGHT_BRACE, TokenType.EOF):
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        
        self.consume(TokenType.RIGHT_BRACE, "Expected '}'")
        return BlockStatement(statements)
    
    def parse_statement(self) -> Optional[Statement]:
        """Parse a statement"""
        try:
            # Block statement
            if self.match(TokenType.LEFT_BRACE):
                return self.parse_block_statement()
            
            # If statement
            if self.match(TokenType.IF):
                return self.parse_if_statement()
            
            # While statement
            if self.match(TokenType.WHILE):
                return self.parse_while_statement()
            
            # For statement
            if self.match(TokenType.FOR):
                return self.parse_for_statement()
            
            # Parallel for statement
            if self.match(TokenType.AT) and self.peek_token().type == TokenType.PARALLEL_FOR:
                return self.parse_parallel_for_statement()
            
            # Kernel launch statement
            if self.match(TokenType.LAUNCH):
                return self.parse_kernel_launch_statement()
            
            # Return statement
            if self.match(TokenType.RETURN):
                return self.parse_return_statement()
            
            # Sync statements
            if self.match(TokenType.SYNC, TokenType.SYNC_THREADS, TokenType.SYNC_WARP):
                sync_type = self.advance().value
                self.consume(TokenType.SEMICOLON, "Expected ';' after sync")
                return SyncStatement(sync_type)
            
            # Declaration
            if self.is_declaration():
                return self.parse_variable_declaration()
            
            # Expression statement
            expr = self.parse_expression()
            self.consume(TokenType.SEMICOLON, "Expected ';' after expression")
            return ExpressionStatement(expr)
            
        except ParseError as e:
            print(f"Statement parse error: {e}")
            self.skip_to_next_statement()
            return None
    
    def is_declaration(self) -> bool:
        """Check if current position starts a declaration"""
        return self.match(TokenType.CONST, TokenType.STACK, TokenType.HEAP, TokenType.ARENA,
                         TokenType.GLOBAL, TokenType.SHARED, TokenType.LOCAL, TokenType.VOLATILE,
                         TokenType.ATOMIC, TokenType.I8, TokenType.I16, TokenType.I32, TokenType.I64,
                         TokenType.U8, TokenType.U16, TokenType.U32, TokenType.U64,
                         TokenType.F32, TokenType.F64, TokenType.BOOL, TokenType.VOID,
                         TokenType.VEC2, TokenType.VEC3, TokenType.VEC4,
                         TokenType.MAT2, TokenType.MAT3, TokenType.MAT4, TokenType.AUTO)
    
    def parse_if_statement(self) -> IfStatement:
        """Parse an if statement"""
        self.consume(TokenType.IF, "Expected 'if'")
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'if'")
        condition = self.parse_expression()
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after if condition")
        
        then_stmt = self.parse_statement()
        
        else_stmt = None
        if self.match(TokenType.ELSE):
            self.advance()
            else_stmt = self.parse_statement()
        
        return IfStatement(condition, then_stmt, else_stmt)
    
    def parse_while_statement(self) -> WhileStatement:
        """Parse a while statement"""
        self.consume(TokenType.WHILE, "Expected 'while'")
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'while'")
        condition = self.parse_expression()
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after while condition")
        
        body = self.parse_statement()
        return WhileStatement(condition, body)
    
    def parse_for_statement(self) -> ForStatement:
        """Parse a for statement"""
        self.consume(TokenType.FOR, "Expected 'for'")
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'for'")
        
        # Check for range-based for loop
        if self.match(TokenType.I8, TokenType.I16, TokenType.I32, TokenType.I64,
                     TokenType.U8, TokenType.U16, TokenType.U32, TokenType.U64) and \
           self.peek_token().type == TokenType.IDENTIFIER and \
           self.peek_token(2).type == TokenType.IN:
            # Range-based for: for (i32 i in 0..10)
            var_type = self.parse_type()
            var_name = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
            self.consume(TokenType.IN, "Expected 'in'")
            
            # Parse range expression
            start = self.parse_expression()
            self.consume(TokenType.DOUBLE_DOT, "Expected '..'")
            end = self.parse_expression()
            
            step = None
            if self.match(TokenType.STEP):
                self.advance()
                step = self.parse_expression()
            
            self.consume(TokenType.RIGHT_PAREN, "Expected ')' after for range")
            body = self.parse_statement()
            
            # Convert to regular for loop structure
            init = VariableDeclaration(None, var_type, var_name, start)
            condition = BinaryExpression(IdentifierExpression(var_name), '<', end)
            
            if step:
                update = AssignmentExpression(IdentifierExpression(var_name), '+=', step)
            else:
                update = AssignmentExpression(IdentifierExpression(var_name), '+=', LiteralExpression(1, 'i32'))
            
            return ForStatement(init, condition, update, body)
        
        # Regular for loop
        init = None
        if not self.match(TokenType.SEMICOLON):
            if self.is_declaration():
                init = self.parse_variable_declaration()
            else:
                init = ExpressionStatement(self.parse_expression())
                self.consume(TokenType.SEMICOLON, "Expected ';' after for init")
        else:
            self.advance()  # consume ';'
        
        condition = None
        if not self.match(TokenType.SEMICOLON):
            condition = self.parse_expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after for condition")
        
        update = None
        if not self.match(TokenType.RIGHT_PAREN):
            update = self.parse_expression()
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after for update")
        
        body = self.parse_statement()
        return ForStatement(init, condition, update, body)
    
    def parse_parallel_for_statement(self) -> ParallelForStatement:
        """Parse a parallel for statement"""
        self.consume(TokenType.AT, "Expected '@'")
        self.consume(TokenType.PARALLEL_FOR, "Expected 'parallel_for'")
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after '@parallel_for'")
        
        # Variable declaration
        var_type = self.parse_type()
        var_name = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
        self.consume(TokenType.IN, "Expected 'in'")
        
        # Range expression
        start = self.parse_expression()
        self.consume(TokenType.DOUBLE_DOT, "Expected '..'")
        end = self.parse_expression()
        
        step = None
        if self.match(TokenType.STEP):
            self.advance()
            step = self.parse_expression()
        
        range_expr = RangeExpression(start, end, step)
        
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after parallel for range")
        
        # Parse reduce clauses
        reduce_clauses = []
        if self.match(TokenType.REDUCE):
            self.advance()
            self.consume(TokenType.LEFT_PAREN, "Expected '(' after 'reduce'")
            
            reduce_clauses.append(self.parse_reduce_clause())
            while self.match(TokenType.COMMA):
                self.advance()
                reduce_clauses.append(self.parse_reduce_clause())
            
            self.consume(TokenType.RIGHT_PAREN, "Expected ')' after reduce clauses")
        
        body = self.parse_statement()
        return ParallelForStatement(var_name, range_expr, reduce_clauses, body)
    
    def parse_reduce_clause(self) -> ReduceClause:
        """Parse a reduce clause like 'sum: add'"""
        var_name = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
        self.consume(TokenType.COLON, "Expected ':' after reduce variable")
        
        # Parse operation
        if self.match(TokenType.IDENTIFIER):
            operation = self.advance().value
        else:
            raise ParseError("Expected reduce operation", self.current_token())
        
        return ReduceClause(var_name, operation)
    
    def parse_kernel_launch_statement(self) -> KernelLaunchStatement:
        """Parse a kernel launch statement"""
        self.consume(TokenType.LAUNCH, "Expected 'launch'")
        self.consume(TokenType.LESS, "Expected '<' after 'launch'")
        
        # Grid configuration
        self.consume(TokenType.LEFT_BRACE, "Expected '{' for grid config")
        grid_config = [self.parse_expression()]
        while self.match(TokenType.COMMA):
            self.advance()
            grid_config.append(self.parse_expression())
        self.consume(TokenType.RIGHT_BRACE, "Expected '}' after grid config")
        
        self.consume(TokenType.COMMA, "Expected ',' between grid and block config")
        
        # Block configuration
        self.consume(TokenType.LEFT_BRACE, "Expected '{' for block config")
        block_config = [self.parse_expression()]
        while self.match(TokenType.COMMA):
            self.advance()
            block_config.append(self.parse_expression())
        self.consume(TokenType.RIGHT_BRACE, "Expected '}' after block config")
        
        self.consume(TokenType.GREATER, "Expected '>' after launch config")
        
        # Function name and arguments
        function_name = self.consume(TokenType.IDENTIFIER, "Expected function name").value
        self.consume(TokenType.LEFT_PAREN, "Expected '(' after function name")
        
        arguments = []
        if not self.match(TokenType.RIGHT_PAREN):
            arguments.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                self.advance()
                arguments.append(self.parse_expression())
        
        self.consume(TokenType.RIGHT_PAREN, "Expected ')' after arguments")
        self.consume(TokenType.SEMICOLON, "Expected ';' after kernel launch")
        
        return KernelLaunchStatement(grid_config, block_config, function_name, arguments)
    
    def parse_return_statement(self) -> ReturnStatement:
        """Parse a return statement"""
        self.consume(TokenType.RETURN, "Expected 'return'")
        
        value = None
        if not self.match(TokenType.SEMICOLON):
            value = self.parse_expression()
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after return")
        return ReturnStatement(value)
    
    def parse_expression(self) -> Expression:
        """Parse an expression (assignment level)"""
        return self.parse_assignment()
    
    def parse_assignment(self) -> Expression:
        """Parse assignment expressions"""
        expr = self.parse_logical_or()
        
        if self.match(TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
                     TokenType.MULTIPLY_ASSIGN, TokenType.DIVIDE_ASSIGN, TokenType.MODULO_ASSIGN):
            operator = self.advance().value
            right = self.parse_assignment()
            return AssignmentExpression(expr, operator, right)
        
        return expr
    
    def parse_logical_or(self) -> Expression:
        """Parse logical OR expressions"""
        expr = self.parse_logical_and()
        
        while self.match(TokenType.OR):
            operator = self.advance().value
            right = self.parse_logical_and()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_logical_and(self) -> Expression:
        """Parse logical AND expressions"""
        expr = self.parse_equality()
        
        while self.match(TokenType.AND):
            operator = self.advance().value
            right = self.parse_equality()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_equality(self) -> Expression:
        """Parse equality expressions"""
        expr = self.parse_relational()
        
        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator = self.advance().value
            right = self.parse_relational()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_relational(self) -> Expression:
        """Parse relational expressions"""
        expr = self.parse_additive()
        
        while self.match(TokenType.LESS, TokenType.LESS_EQUAL, TokenType.GREATER, TokenType.GREATER_EQUAL):
            operator = self.advance().value
            right = self.parse_additive()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_additive(self) -> Expression:
        """Parse additive expressions"""
        expr = self.parse_multiplicative()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.advance().value
            right = self.parse_multiplicative()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_multiplicative(self) -> Expression:
        """Parse multiplicative expressions"""
        expr = self.parse_unary()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            operator = self.advance().value
            right = self.parse_unary()
            expr = BinaryExpression(expr, operator, right)
        
        return expr
    
    def parse_unary(self) -> Expression:
        """Parse unary expressions"""
        if self.match(TokenType.NOT, TokenType.BIT_NOT, TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY, TokenType.BIT_AND):
            operator = self.advance().value
            operand = self.parse_unary()
            return UnaryExpression(operator, operand)
        
        return self.parse_postfix()
    
    def parse_postfix(self) -> Expression:
        """Parse postfix expressions (function calls, member access, array access)"""
        expr = self.parse_primary()
        
        while True:
            if self.match(TokenType.LEFT_PAREN):
                # Function call
                self.advance()
                arguments = []
                if not self.match(TokenType.RIGHT_PAREN):
                    arguments.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        self.advance()
                        arguments.append(self.parse_expression())
                self.consume(TokenType.RIGHT_PAREN, "Expected ')' after function arguments")
                expr = CallExpression(expr, arguments)
            
            elif self.match(TokenType.LEFT_BRACKET):
                # Array access
                self.advance()
                index = self.parse_expression()
                self.consume(TokenType.RIGHT_BRACKET, "Expected ']' after array index")
                expr = MemberExpression(expr, "", True)  # computed member access
            
            elif self.match(TokenType.DOT):
                # Member access
                self.advance()
                property_name = self.consume(TokenType.IDENTIFIER, "Expected property name").value
                expr = MemberExpression(expr, property_name, False)
            
            else:
                break
        
        return expr
    
    def parse_primary(self) -> Expression:
        """Parse primary expressions"""
        # Literals
        if self.match(TokenType.INTEGER):
            token = self.advance()
            return LiteralExpression(int(token.value.split('i')[0].split('u')[0]), 'integer')
        
        if self.match(TokenType.FLOAT):
            token = self.advance()
            return LiteralExpression(float(token.value.rstrip('f')), 'float')
        
        if self.match(TokenType.STRING):
            token = self.advance()
            return LiteralExpression(token.value, 'string')
        
        if self.match(TokenType.TRUE, TokenType.FALSE):
            token = self.advance()
            return LiteralExpression(token.value == 'true', 'boolean')
        
        if self.match(TokenType.NULL):
            self.advance()
            return LiteralExpression(None, 'null')
        
        # Identifier
        if self.match(TokenType.IDENTIFIER):
            token = self.advance()
            return IdentifierExpression(token.value)
        
        # Parenthesized expression
        if self.match(TokenType.LEFT_PAREN):
            self.advance()
            expr = self.parse_expression()
            self.consume(TokenType.RIGHT_PAREN, "Expected ')' after expression")
            return expr
        
        # Vector literal
        if self.match(TokenType.LEFT_BRACE):
            self.advance()
            elements = []
            if not self.match(TokenType.RIGHT_BRACE):
                elements.append(self.parse_expression())
                while self.match(TokenType.COMMA):
                    self.advance()
                    elements.append(self.parse_expression())
            self.consume(TokenType.RIGHT_BRACE, "Expected '}' after vector literal")
            return VectorLiteralExpression(elements)
        
        raise ParseError("Expected expression", self.current_token())

# =============================================================================
# Pretty Printer for AST
# =============================================================================

class ASTPrinter:
    def __init__(self, indent_size: int = 2):
        self.indent_size = indent_size
        self.indent_level = 0
    
    def indent(self) -> str:
        return " " * (self.indent_level * self.indent_size)
    
    def print_ast(self, node: ASTNode) -> str:
        method_name = f"print_{type(node).__name__}"
        method = getattr(self, method_name, self.print_generic)
        return method(node)
    
    def print_generic(self, node: ASTNode) -> str:
        return f"{self.indent()}{type(node).__name__}"
    
    def print_Program(self, node: Program) -> str:
        result = f"{self.indent()}Program:\n"
        self.indent_level += 1
        for decl in node.declarations:
            result += self.print_ast(decl) + "\n"
        self.indent_level -= 1
        return result
    
    def print_FunctionDeclaration(self, node: FunctionDeclaration) -> str:
        result = f"{self.indent()}FunctionDeclaration: {node.name}\n"
        self.indent_level += 1
        
        if node.annotations:
            result += f"{self.indent()}Annotations:\n"
            self.indent_level += 1
            for ann in node.annotations:
                result += self.print_ast(ann) + "\n"
            self.indent_level -= 1
        
        if node.parameters:
            result += f"{self.indent()}Parameters:\n"
            self.indent_level += 1
            for param in node.parameters:
                result += self.print_ast(param) + "\n"
            self.indent_level -= 1
        
        if node.return_type:
            result += f"{self.indent()}ReturnType:\n"
            self.indent_level += 1
            result += self.print_ast(node.return_type) + "\n"
            self.indent_level -= 1
        
        result += f"{self.indent()}Body:\n"
        self.indent_level += 1
        result += self.print_ast(node.body) + "\n"
        self.indent_level -= 1
        
        self.indent_level -= 1
        return result
    
    def print_VariableDeclaration(self, node: VariableDeclaration) -> str:
        result = f"{self.indent()}VariableDeclaration: {node.name}"
        if node.storage_qualifier:
            result += f" ({node.storage_qualifier})"
        result += "\n"
        
        self.indent_level += 1
        result += f"{self.indent()}Type:\n"
        self.indent_level += 1
        result += self.print_ast(node.type) + "\n"
        self.indent_level -= 1
        
        if node.initializer:
            result += f"{self.indent()}Initializer:\n"
            self.indent_level += 1
            result += self.print_ast(node.initializer) + "\n"
            self.indent_level -= 1
        
        self.indent_level -= 1
        return result
    
    def print_BlockStatement(self, node: BlockStatement) -> str:
        result = f"{self.indent()}BlockStatement:\n"
        self.indent_level += 1
        for stmt in node.statements:
            result += self.print_ast(stmt) + "\n"
        self.indent_level -= 1
        return result
    
    def print_BinaryExpression(self, node: BinaryExpression) -> str:
        result = f"{self.indent()}BinaryExpression: {node.operator}\n"
        self.indent_level += 1
        result += f"{self.indent()}Left:\n"
        self.indent_level += 1
        result += self.print_ast(node.left) + "\n"
        self.indent_level -= 1
        result += f"{self.indent()}Right:\n"
        self.indent_level += 1
        result += self.print_ast(node.right) + "\n"
        self.indent_level -= 1
        self.indent_level -= 1
        return result
    
    def print_IdentifierExpression(self, node: IdentifierExpression) -> str:
        return f"{self.indent()}Identifier: {node.name}"
    
    def print_LiteralExpression(self, node: LiteralExpression) -> str:
        return f"{self.indent()}Literal: {node.value} ({node.type})"

# =============================================================================
# Main Testing Function
# =============================================================================

def main():
    # Test the parser with sample AIL code
    sample_code = """
    @vectorize @simd_width(8)
    func matrix_multiply(
        const f32* a,
        const f32* b,
        heap f32* result,
        i32 size
    ) -> void {
        @parallel_for (i32 i in 0..size) {
            for (i32 j in 0..size) {
                f32 sum = 0.0f;
                
                @parallel_for (i32 k in 0..size step 4) 
                reduce(sum: add) {
                    sum += a[i * size + k] * b[k * size + j];
                }
                
                result[i * size + j] = sum;
            }
        }
    }
    
    @kernel @block_size(16, 16, 1)
    func gpu_kernel(global<f32>* data, i32 count) -> void {
        i32 idx = thread_idx_x() + block_idx_x() * 16;
        if (idx < count) {
            data[idx] = data[idx] * 2.0f;
        }
    }
    
    func main() -> i32 {
        const i32 size = 1024;
        heap f32* matrix_a = alloc(null, sizeof(f32) * size * size);
        heap f32* matrix_b = alloc(null, sizeof(f32) * size * size);
        heap f32* result = alloc(null, sizeof(f32) * size * size);
        
        matrix_multiply(matrix_a, matrix_b, result, size);
        
        launch<{64, 64, 1}, {16, 16, 1}> gpu_kernel(result, size * size);
        
        free(matrix_a);
        free(matrix_b);
        free(result);
        
        return 0;
    }
    """
    
    print("=== AIL Parser Test ===\n")
    print("Source code:")
    print(sample_code)
    print("\n" + "="*50 + "\n")
    
    # Tokenize
    lexer = Lexer(sample_code)
    tokens = lexer.tokenize()
    
    print("Tokens:")
    for token in tokens[:20]:  # Show first 20 tokens
        print(f"  {token.type.name}: '{token.value}' at {token.line}:{token.column}")
    if len(tokens) > 20:
        print(f"  ... and {len(tokens) - 20} more tokens")
    
    print("\n" + "="*50 + "\n")
    
    # Parse
    parser = Parser(tokens)
    try:
        ast = parser.parse()
        print("Parse successful!")
        print("\nAST:")
        printer = ASTPrinter()
        print(printer.print_ast(ast))
    except ParseError as e:
        print(f"Parse error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
