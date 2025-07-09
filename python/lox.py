#!/usr/bin/env python
from argparse import ArgumentParser
from enum import StrEnum, auto
from typing import Any, Callable, Optional, Union, Generic, TypeVar, cast
import abc
import sys


R = TypeVar("R")


class TokenType(StrEnum):
    # singe char tokens
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    COMMA = auto()
    DOT = auto()
    MINUS = auto()
    PLUS = auto()
    SEMICOLON = auto()
    SLASH = auto()
    STAR = auto()
    # one or two char tokens
    BANG = auto()
    BANG_EQUAL = auto()
    EQUAL = auto()
    EQUAL_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    # literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    # keywords
    AND = auto()
    CLASS = auto()
    ELSE = auto()
    FALSE = auto()
    FUN = auto()
    FOR = auto()
    IF = auto()
    NIL = auto()
    OR = auto()
    PRINT = auto()
    RETURN = auto()
    SUPER = auto()
    THIS = auto()
    TRUE = auto()
    VAR = auto()
    WHILE = auto()
    BREAK = auto()
    CONTINUE = auto()

    EOF = auto()


class Token:
    def __init__(self, type: TokenType, lexeme: str, literal: Any, line: int):
        self.type: TokenType = type
        self.lexeme: str = lexeme
        self.literal: Any = literal
        self.line: int = line

    def __str__(self):
        # print like in java compiler to compare results
        type = self.type.upper()
        lexeme = self.lexeme or ''
        literal = self.literal or 'null'
        return f"{type} {lexeme} {literal}"


class LoxRuntimeError(RuntimeError):
    def __init__(self, token: Token, message: str):
        super().__init__(message)
        self.token: Token = token


class LoxParseError(RuntimeError):
    pass


class LoxBreakException(Exception):
    pass


class LoxContinueException(Exception):
    pass


class Environment:
    def __init__(self, enclosing: Optional["Environment"] = None):
        self.values: dict[str, object] = {}
        self.enclosing: Optional[Environment] = enclosing

    def define(self, name: str, value: object):
        self.values[name] = value

    def assign(self, name: Token, value: object):
        if name.lexeme in self.values:
            self.values[name.lexeme] = value
            return
        if self.enclosing != None:
            return self.enclosing.assign(name, value)
        raise LoxRuntimeError(name, f"Undefined variable '{name.lexeme}'")

    def get(self, name: Token):
        if name.lexeme in self.values:
            return self.values[name.lexeme]
        if self.enclosing != None:
            return self.enclosing.get(name)
        raise LoxRuntimeError(name, f"Undefined variable '{name.lexeme}'")


class Expr(abc.ABC):
    @abc.abstractmethod
    def accept(self, visitor: "ExprVisitor[R]") -> R: 
        pass


class Stmt(abc.ABC):
    @abc.abstractmethod
    def accept(self, visitor: "StmtVisitor[R]") -> R:
        pass


class ExprVisitor(abc.ABC, Generic[R]):
    @abc.abstractmethod
    def visitBinaryExpr(self, expr: "Binary") -> R:
        pass

    @abc.abstractmethod
    def visitGroupingExpr(self, expr: "Grouping") -> R:
        pass

    @abc.abstractmethod
    def visitLiteralExpr(self, expr: "Literal") -> R:
        pass

    @abc.abstractmethod
    def visitLogicalExpr(self, expr: "Logical") -> R:
        pass

    @abc.abstractmethod
    def visitUnaryExpr(self, expr: "Unary") -> R:
        pass

    @abc.abstractmethod
    def visitVariableExpr(self, expr: "Variable") -> R:
        pass

    @abc.abstractmethod
    def visitAssignExpr(self, expr: "Assign") -> R:
        pass


class StmtVisitor(abc.ABC, Generic[R]):
    @abc.abstractmethod
    def visitIfStmt(self, stmt: "IfStmt") -> R:
        pass

    @abc.abstractmethod
    def visitPrintStmt(self, stmt: "PrintStmt") -> R:
        pass

    @abc.abstractmethod
    def visitExpressionStmt(self, stmt: "ExpressionStmt") -> R:
        pass

    @abc.abstractmethod
    def visitVarStmt(self, stmt: "VarStmt") -> R:
        pass

    @abc.abstractmethod
    def visitWhileStmt(self, stmt: "WhileStmt") -> R:
        pass

    @abc.abstractmethod
    def visitBreakStmt(self, stmt: "BreakStmt") -> R:
        pass

    @abc.abstractmethod
    def visitContinueStmt(self, stmt: "ContinueStmt") -> R:
        pass

    @abc.abstractmethod
    def visitBlock(self, stmt: "BlockStmt") -> R:
        pass


class Assign(Expr):
    def __init__(self, name: Token, value: Expr):
        self.name: Token = name
        self.value: Expr = value
    
    def accept(self, visitor: ExprVisitor[R]) -> R:
        return visitor.visitAssignExpr(self)
    
    def __str__(self):
        return f"Assign({self.name}, {self.value})"


class Binary(Expr):
    def __init__(self, left: Expr, operator: Token, right: Expr):
        self.left: Expr = left
        self.operator: Token = operator
        self.right: Expr = right

    def accept(self, visitor: ExprVisitor[R]) -> R:
        return visitor.visitBinaryExpr(self)
    
    def __str__(self):
        return f"Binary({self.left}, {self.operator}, {self.right})"
    

class Grouping(Expr):
    def __init__(self, expression: Expr):
        self.expression: Expr = expression

    def accept(self, visitor: ExprVisitor[R]) -> R:
        return visitor.visitGroupingExpr(self)
    
    def __str__(self):
        return f"({self.expression})"


class Literal(Expr):
    def __init__(self, value: Union[str, int, float, bool, None]):
        self.value: Union[str, int, float, bool, None] = value

    def accept(self, visitor: ExprVisitor[R]) -> R:
        return visitor.visitLiteralExpr(self)
    
    def __str__(self):
        return str(self.value)


class Logical(Expr):
    def __init__(self, left: Expr, operator: Token, right: Expr):
        self.left: Expr = left
        self.operator: Token = operator
        self.right: Expr = right
    
    def accept(self, visitor: ExprVisitor[R]) -> R:
        return visitor.visitLogicalExpr(self)


class Unary(Expr):
    def __init__(self, operator: Token, right: Expr):
        self.operator: Token = operator
        self.right: Expr = right

    def accept(self, visitor: ExprVisitor[R]) -> R:
        return visitor.visitUnaryExpr(self)
    
    def __str__(self):
        return f"Unary({self.operator}, {self.right})"


class Variable(Expr):
    def __init__(self, name: Token):
        self.name: Token = name
    
    def accept(self, visitor: ExprVisitor[R]) -> R:
        return visitor.visitVariableExpr(self)

    def __str__(self):
        return f"Var({self.name})"


class ExpressionStmt(Stmt):
    def __init__(self, expression: Expr):
        self.expression: Expr = expression

    def accept(self, visitor: StmtVisitor[R]) -> R:
        return visitor.visitExpressionStmt(self)
    
    def __str__(self):
        return f"Stmt[{self.expression}]"


class IfStmt(Stmt):
    def __init__(self, condition: Expr, thenBranch: Stmt, elseBranch: Union[Stmt, None]):
        self.condition: Expr = condition
        self.thenBranch: Stmt = thenBranch
        self.elseBranch: Union[Stmt, None] = elseBranch

    def accept(self, visitor: StmtVisitor[R]) -> R:
        return visitor.visitIfStmt(self)
    
    def __str__(self):
        if self.elseBranch:
            return f"(if ({self.condition}) ({self.thenBranch}) ({self.elseBranch}))"
        else:
            return f"(if ({self.condition}) ({self.thenBranch}) ())"


class PrintStmt(Stmt):
    def __init__(self, expression: Expr):
        self.expression: Expr = expression

    def accept(self, visitor: StmtVisitor[R]) -> R:
        return visitor.visitPrintStmt(self)
    
    def __str__(self):
        return f"Print[{self.expression}]"


class VarStmt(Stmt):
    def __init__(self, name: Token, initializer: Optional[Expr] = None):
        self.name: Token = name
        self.initializer: Optional[Expr] = initializer

    def accept(self, visitor: StmtVisitor[R]) -> R:
        return visitor.visitVarStmt(self)
    
    def __str__(self):
        return f"VarStmt[{self.name} = {self.initializer}]"


class WhileStmt(Stmt):
    def __init__(self, condition: Expr, body: Stmt):
        self.condition: Expr = condition
        self.body: Stmt = body

    def accept(self, visitor: StmtVisitor[R]) -> R:
        return visitor.visitWhileStmt(self)
    
    def __str__(self):
        return f"(if ({self.condition}) ({self.body})))"


class BreakStmt(Stmt):
    def __init__(self, name: Token):
        self.name: Token = name

    def accept(self, visitor: StmtVisitor[R]) -> R:
        return visitor.visitBreakStmt(self)
    
    def __str__(self):
        return f"(break)"
    

class ContinueStmt(Stmt):
    def __init__(self, name: Token):
        self.name: Token = name

    def accept(self, visitor: StmtVisitor[R]) -> R:
        return visitor.visitContinueStmt(self)
    
    def __str__(self):
        return f"(continue)"
        

class BlockStmt(Stmt):
    def __init__(self, statements: list[Stmt]):
        self.statements = statements
    
    def accept(self, visitor: StmtVisitor[R]) -> R:
        return visitor.visitBlock(self)
    
    def __str__(self):
        return f"Block[{self.statements}]"


class AstPrinter(ExprVisitor[str], StmtVisitor[str]):
    def _parenthesize(self, name: str, *exprs: Union[Expr, Stmt]):
        string_container = ["(", name]
        for expr in exprs:
            string_container.append(" ")
            string_container.append(expr.accept(self))
        string_container.append(")")
        return "".join(string_container)
    
    def print(self, expr: Expr) -> str:
        return expr.accept(self)
    
    def printStmt(self, stmt: Stmt) -> str:
        return stmt.accept(self)
    
    def printProgram(self, statements: list[Stmt]) -> str:
        string_container = ["(program"]
        for idx, stmt in enumerate(statements):
            string_container.append(" ")
            string_container.append(stmt.accept(self))
            if idx < len(statements) - 1:
                string_container.append("\n")
        string_container.append(")")
        return "".join(string_container)
    
    def visitBlock(self, stmt: BlockStmt) -> str:
        string_container = ["(block"]
        for idx, statement in enumerate(stmt.statements):
            string_container.append(statement.accept(self))
            if idx < len(stmt.statements) - 1:
                string_container.append("\n\t")
        string_container.append(")")
        return " ".join(string_container)
    
    def visitExpressionStmt(self, stmt: ExpressionStmt) -> str:
        return self._parenthesize("expression", stmt.expression)
    
    def visitPrintStmt(self, stmt: PrintStmt) -> str:
        return self._parenthesize("print", stmt.expression)
    
    def visitVarStmt(self, stmt: VarStmt) -> str:
        var_container = ["(var", stmt.name.lexeme, ")"]
        if stmt.initializer is None:
            stmt_container = var_container
        else:
            stmt_container = ["(="] + var_container + [stmt.initializer.accept(self), ")"]
        return " ".join(stmt_container)
    
    def visitAssignExpr(self, expr: Assign) -> str:
        string_container = []
        string_container.append("(")
        string_container.append(expr.name.lexeme)
        string_container.append("=")
        string_container.append(expr.value.accept(self))
        string_container.append(")")
        return "".join(string_container)

    def visitBinaryExpr(self, expr: Binary) -> str:
        return self._parenthesize(expr.operator.lexeme, expr.left, expr.right)

    def visitGroupingExpr(self, expr: Grouping) -> str:
        return self._parenthesize("group", expr.expression)
    
    def visitLiteralExpr(self, expr: Literal) -> str:
        if expr.value is None:
            return "nil"
        elif isinstance(expr.value, str):
            return f'"{expr.value}"'
        else:
            return str(expr.value)
        
    def visitIfStmt(self, stmt: IfStmt) -> str:
        if stmt.elseBranch:
            return self._parenthesize("If", stmt.condition, stmt.thenBranch, stmt.elseBranch)
        else:
            return self._parenthesize("If", stmt.condition, stmt.thenBranch)
        
    def visitLogicalExpr(self, expr: Logical):
        return self._parenthesize(expr.operator.lexeme, expr.left, expr.right)
    
    def visitUnaryExpr(self, expr: Unary) -> str:
        return self._parenthesize(expr.operator.lexeme, expr.right)

    def visitVariableExpr(self, expr: Variable) -> str:
        return expr.name.lexeme
    
    def visitWhileStmt(self, stmt: WhileStmt) -> str:
        string_container = ["(while"]
        string_container.append(stmt.condition.accept(self))
        string_container.append(stmt.body.accept(self))
        string_container.append(")")
        return " ".join(string_container)

    def visitBreakStmt(self, stmt: BreakStmt) -> str:
        return "(break)"

    def visitContinueStmt(self, stmt: ContinueStmt) -> str:
        return "(continue)"

    def test(self):
        expr = Binary(
            Unary(
                Token(TokenType.MINUS, "-", None, 1),
                Literal(123)
            ),
            Token(TokenType.STAR, "*", None, 1),
            Grouping(
                Literal(45.67),
            )
        )
        print(self.print(expr))


class Interpreter(ExprVisitor[object], StmtVisitor[None]):

    def __init__(self):
        self.environment = Environment()
        self.loop_depth: int = 0
    
    def interpret(self, statements: list[Stmt], onError: Callable):
        try:
            for statement in statements:
                self.execute(statement)
        except LoxRuntimeError as err:
            onError(err)

    def stringify(self, obj: object) -> str:
        if obj == None:
            return "nil"
        return str(obj)

    def evaluate(self, expr: Expr) -> object:
        return expr.accept(self)
    
    def execute(self, stmt: Stmt):
        stmt.accept(self)

    def executeBlock(self, statements: list[Stmt], environment: Environment):
        prevEnvironment = self.environment
        try:
            self.environment = environment
            for statement in statements:
                self.execute(statement)
        finally:
            self.environment = prevEnvironment
    
    def isTruthy(self, obj: object) -> bool:
        if obj == None:
            return False
        if isinstance(obj, bool):
            return bool(obj)
        return True

    def checkNumberOperand(self, operator: Token, operand: object) -> None:
        if isinstance(operand, (int, float)):
            return
        raise LoxRuntimeError(operator, "Operand must be a number")

    def checkNumberOperands(self, operator: Token, left: object, right: object) -> None:
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return
        raise LoxRuntimeError(operator, "Operands must be numbers")

    def visitLiteralExpr(self, expr: Literal) -> object:
        return expr.value
    
    def visitLogicalExpr(self, expr: Logical) -> object:
        left = self.evaluate(expr.left)
        if expr.operator.type == TokenType.OR:
            if self.isTruthy(left):
                return left
        else:
            if not self.isTruthy(left):
                return left
        return self.evaluate(expr.right)
    
    def visitGroupingExpr(self, expr: Grouping) -> object:
        return self.evaluate(expr.expression)
    
    def visitUnaryExpr(self, expr: Unary) -> object:
        right: object = self.evaluate(expr.right)
        if expr.operator.type == TokenType.BANG:
            return not self.isTruthy(right)
        if expr.operator.type == TokenType.MINUS:
            self.checkNumberOperand(expr.operator, right)
            return -float(cast(float, right))
        
    def visitVariableExpr(self, expr: Variable) -> object:
        return self.environment.get(expr.name)

    def visitBinaryExpr(self, expr: Binary) -> object:
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)
        if expr.operator.type == TokenType.GREATER:
            self.checkNumberOperands(expr.operator, left, right)
            return float(cast(float, left)) > float(cast(float, right))
        elif expr.operator.type == TokenType.GREATER_EQUAL:
            self.checkNumberOperands(expr.operator, left, right)
            return float(cast(float, left)) >= float(cast(float, right))
        elif expr.operator.type == TokenType.LESS:
            self.checkNumberOperands(expr.operator, left, right)
            return float(cast(float, left)) < float(cast(float, right))
        elif expr.operator.type == TokenType.LESS_EQUAL:
            self.checkNumberOperands(expr.operator, left, right)
            return float(cast(float, left)) <= float(cast(float, right))
        elif expr.operator.type == TokenType.BANG_EQUAL:
            return left != right
        elif expr.operator.type == TokenType.EQUAL_EQUAL:
            return left == right
        elif expr.operator.type == TokenType.MINUS:
            self.checkNumberOperands(expr.operator, left, right)
            return float(cast(float, left)) - float(cast(float, right))
        elif expr.operator.type == TokenType.PLUS:
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return float(left) + float(right)
            if isinstance(left, str) and isinstance(right, str):
                return str(left) + str(right)
            raise LoxRuntimeError(expr.operator, "Operands must be two numbers or two strings")
        elif expr.operator.type == TokenType.SLASH:
            self.checkNumberOperands(expr.operator, left, right)
            return float(cast(float, left)) / float(cast(float, right))
        elif expr.operator.type == TokenType.STAR:
            self.checkNumberOperands(expr.operator, left, right)
            return float(cast(float, left)) * float(cast(float, right))

    def visitAssignExpr(self, expr: Assign) -> object:
        value = self.evaluate(expr.value)
        self.environment.assign(expr.name, value)
        return value
        
    def visitPrintStmt(self, stmt: PrintStmt):
        value = self.evaluate(stmt.expression)
        print(self.stringify(value))

    def visitExpressionStmt(self, stmt: ExpressionStmt):
        self.evaluate(stmt.expression)

    def visitIfStmt(self, stmt: IfStmt):
        if self.isTruthy(self.evaluate(stmt.condition)):
            self.execute(stmt.thenBranch)
        elif stmt.elseBranch != None:
            self.execute(stmt.elseBranch)

    def visitVarStmt(self, stmt: VarStmt):
        value = None
        if stmt.initializer != None:
            value = self.evaluate(stmt.initializer)
        self.environment.define(stmt.name.lexeme, value)

    def visitWhileStmt(self, stmt: WhileStmt):
        self.loop_depth += 1
        try:
            while self.isTruthy(self.evaluate(stmt.condition)):
                try:
                    self.execute(stmt.body)
                except LoxContinueException:
                    continue
                except LoxBreakException:
                    break
        finally:
            self.loop_depth -= 1

    def visitBreakStmt(self, stmt: BreakStmt):
        raise LoxBreakException()
    
    def visitContinueStmt(self, stmt: ContinueStmt):
        raise LoxContinueException()

    def visitBlock(self, stmt: BlockStmt):
        self.executeBlock(stmt.statements, Environment(self.environment))


class Scanner:
    keywords: dict[str, TokenType] = {
        "and":      TokenType.AND,
        "class":    TokenType.CLASS,
        "class":    TokenType.CLASS,
        "else":     TokenType.ELSE,
        "false":    TokenType.FALSE,
        "for":      TokenType.FOR,
        "fun":      TokenType.FUN,
        "if":       TokenType.IF,
        "nil":      TokenType.NIL,
        "or":       TokenType.OR,
        "print":    TokenType.PRINT,
        "return":   TokenType.RETURN,
        "super":    TokenType.SUPER,
        "this":     TokenType.THIS,
        "true":     TokenType.TRUE,
        "var":      TokenType.VAR,
        "while":    TokenType.WHILE,
        "break":    TokenType.BREAK,
        "continue": TokenType.CONTINUE,
    }

    def __init__(self, source: str, onError: Callable):
        self.onError: Callable[[int, str], None] = onError
        self.source: str = source
        self.tokens: list[Token] = []
        self.start: int = 0 # current token start
        self.current: int = 0 # current scan position 
        self.line: int = 1 # current line
    
    def isAtEnd(self) -> bool:
        return self.current >= len(self.source)

    def peek(self) -> str:
        if self.isAtEnd():
            return "\0"
        else:
            return self.source[self.current]

    def peekNext(self):
        if self.current + 1 >= len(self.source):
            return "\0"
        return self.source[self.current + 1]
        
    def advance(self):
        c = self.source[self.current]
        self.current += 1
        return c
    
    def addToken(self, type: TokenType, literal: Any = None):
        text = self.source[self.start:self.current]
        self.tokens.append(Token(type, text, literal, self.line))

    def isDigit(self, c: str) -> bool:
        return c >= "0" and c <= "9"
    
    def isAlpha(self, c: str) -> bool:
        return (
            (c >= "a" and c <= "z") or
            (c >= "A" and c <= "Z") or 
            (c == "_")
        )

    def isAlphaNumeric(self, c: str) -> bool:
        return self.isAlpha(c) or self.isDigit(c)
    
    def string(self):
        while self.peek() != '"' and not self.isAtEnd():
            if self.peek() == "\n":
                self.line += 1
            self.advance()
        if self.isAtEnd():
            self.onError(self.line, "Unterminated string.")
            return
        self.advance()
        value = self.source[self.start+1:self.current-1]
        self.addToken(TokenType.STRING, value)
    
    def number(self):
        while self.isDigit(self.peek()):
            self.advance()
        if self.peek() == "." and self.isDigit(self.peekNext()):
            self.advance()
            while self.isDigit(self.peek()):
                self.advance()
        val = self.source[self.start:self.current]
        self.addToken(TokenType.NUMBER, float(val))
    
    def identifier(self):
        while self.isAlphaNumeric(self.peek()):
            self.advance()
        val = self.source[self.start:self.current]
        type = self.keywords.get(val, TokenType.IDENTIFIER)
        self.addToken(type)

    def handleTwoCharToken(self, second_char: str, twochar_type: TokenType, onechar_type: TokenType):
        if self.peek() == second_char:
            self.addToken(twochar_type)
            self.advance()
        else:
            self.addToken(onechar_type)

    def handleComment(self):
        self.advance()
        while self.peek() != "\n" and not self.isAtEnd():
            self.advance()
        
    def scanToken(self):
        c = self.advance()
        onechar_tokens = {
            "(": TokenType.LEFT_PAREN,
            ")": TokenType.RIGHT_PAREN,
            "{": TokenType.LEFT_BRACE,
            "}": TokenType.RIGHT_BRACE,
            ",": TokenType.COMMA,
            ".": TokenType.DOT,
            "-": TokenType.MINUS,
            "+": TokenType.PLUS,
            ";": TokenType.SEMICOLON,
            "*": TokenType.STAR,
        }
        if c in onechar_tokens:
            self.addToken(onechar_tokens[c])
        elif c == "!":
            self.handleTwoCharToken("=", TokenType.BANG_EQUAL, TokenType.BANG)
        elif c == "=":
            self.handleTwoCharToken("=", TokenType.EQUAL_EQUAL, TokenType.EQUAL)
        elif c == "<":
            self.handleTwoCharToken("=", TokenType.LESS_EQUAL, TokenType.LESS)
        elif c == ">":
            self.handleTwoCharToken("=", TokenType.GREATER_EQUAL, TokenType.GREATER)
        elif c == "/":
            if self.peek() == "/":
                self.handleComment()
            else:
                self.addToken(TokenType.SLASH)
        elif c == "\n":
            self.line += 1
        elif c == '"':
            self.string()
        elif self.isDigit(c):
            self.number()
        elif self.isAlpha(c):
            self.identifier()
        elif c not in (" ", "\r", "\t"):
            self.onError(self.line, f"Unexpected character {c}")

    def scanTokens(self) -> list[Token]:
        while not self.isAtEnd():
            self.start = self.current
            self.scanToken()
        self.tokens.append(Token(TokenType.EOF, "", None, self.line))
        return self.tokens


class Parser:
    """Grammar
    program        → declaration* EOF ;
    declaration    → varDecl | statement;
    varDecl        → "var" IDENTIFIER ( "=" expression )? ";" ;
    statement      → exprStmt | forStmt | ifStmt | printStmt | whileStmt | breakStmt | continueStmt | block ;
    breakStmt      → "break" ";" ;
    continueStmt   → "continue" ";" ; 
    forStmt        → "for" "(" (varDecl | exprStmt | ";" ) expression? ";" expression? ")" statement ; 
    whileStmt      → "while" "(" expression ")" statement ;
    ifStmt         → "if" "(" expression ")" statement ( "else" statement )? ;
    block          → "{" declaration* "}" ;
    exprStmt       → expression ";" ;
    printStmt      → "print" expression ";" ;
    expression     → assignment ;
    assignment     → IDENTIFIER "=" assignment | logic_or ;
    logic_or       → logic_and ( "or" logic_and )* ;
    logic_and      → equality ( "and" equality )* ;
    equality       → comparison ( ( "!=" | "==" ) comparison )* ;
    comparison     → term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
    term           → factor ( ( "-" | "+" ) factor )* ;
    factor         → unary ( ( "/" | "*" ) unary )* ;
    unary          → ( "!" | "-" ) unary | primary ;
    primary        → NUMBER | STRING | "true" | "false" | "nil" | "(" expression ")" | IDENTIFIER ;
    """
    def __init__(self, tokens: list[Token], onError: Callable):
        self.tokens: list[Token] = tokens
        self.current: int = 0
        self.loop_depth: int = 0
        self.onError: Callable = onError

    def parse(self) -> list[Stmt]:
        statements: list[Stmt] = []
        while not self.isAtEnd():
            next_statement = self.declaration()
            if next_statement is not None:
                statements.append(next_statement)
        return statements
    
    def declaration(self) -> Optional[Stmt]:
        try:
            if self.match(TokenType.VAR):
                return self.varDeclaration()
            return self.statement()
        except LoxParseError as err:
            self.synchronize()
        
    def statement(self) -> Stmt:
        if self.match(TokenType.CONTINUE):
            return self.continueStatement()
        if self.match(TokenType.BREAK):
            return self.breakStatement()
        if self.match(TokenType.FOR):
            return self.forStatement()
        if self.match(TokenType.IF):
            return self.ifStatement()
        if self.match(TokenType.PRINT):
            return self.printStatement()
        if self.match(TokenType.WHILE):
            return self.whileStatement()
        if self.match(TokenType.LEFT_BRACE):
            return BlockStmt(self.block())
        return self.expressionStatement()
    
    def breakStatement(self) -> Stmt:
        keyword = self.previous()
        if self.loop_depth == 0:
            self.error(keyword, "break statement outside a loop is not allowed")
        self.consume(TokenType.SEMICOLON, "Expect ; after break")
        return BreakStmt(keyword)
    
    def continueStatement(self) -> Stmt:
        keyword = self.previous()
        if self.loop_depth == 0:
            self.error(keyword, "continue statement outside a loop is not allowed")
        self.consume(TokenType.SEMICOLON, "Expect ; after continue")
        return ContinueStmt(keyword)
    
    def forStatement(self) -> Stmt:
        self.consume(TokenType.LEFT_PAREN, "Expect ( after for")
        initializer = None
        if self.match(TokenType.SEMICOLON):
            initializer = None
        elif self.match(TokenType.VAR):
            initializer = self.varDeclaration()
        else:
            initializer = self.expressionStatement()
        
        condition = None
        if not self.check(TokenType.SEMICOLON):
            condition = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ; after loop condition")

        increment = None
        if not self.check(TokenType.RIGHT_PAREN):
            increment = self.expression()
        self.consume(TokenType.RIGHT_PAREN, "Expect ) after for clauses")

        self.loop_depth += 1
        body = self.statement()
        self.loop_depth -= 1

        # desugaring - implement for using while.
        if increment is not None:
            body = BlockStmt([body, ExpressionStmt(increment)])
        if condition is None:
            condition = Literal(True)
        body = WhileStmt(condition, body)
        if initializer is not None:
            body = BlockStmt([initializer, body])
        
        return body
    
    def ifStatement(self) -> Stmt:
        self.consume(TokenType.LEFT_PAREN, "Expect ( after if")
        condition = self.expression()
        self.consume(TokenType.RIGHT_PAREN, "Expect ) after if")
        thenBranch = self.statement()
        elseBranch = None
        if self.match(TokenType.ELSE):
            elseBranch = self.statement()
        return IfStmt(condition, thenBranch, elseBranch)
    
    def varDeclaration(self) -> Stmt:
        name = self.consume(TokenType.IDENTIFIER, "Expect variable name")
        initializer: Optional[Expr] = None
        if self.match(TokenType.EQUAL):
            initializer = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ; after variable declaration")
        return VarStmt(name, initializer)
    
    def whileStatement(self) -> Stmt:
        self.consume(TokenType.LEFT_PAREN, "Expect ( after while")
        condition: Expr = self.expression()
        self.consume(TokenType.RIGHT_PAREN, "Expect ) after condition")
        self.loop_depth += 1
        body: Stmt = self.statement()
        self.loop_depth -= 1
        return WhileStmt(condition, body)

    def printStatement(self) -> Stmt:
        value: Expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ; after value")
        return PrintStmt(value)
    
    def block(self) -> list[Stmt]:
        statements: list[Stmt] = []
        while not self.check(TokenType.RIGHT_BRACE) and not self.isAtEnd():
            next_statement = self.declaration()
            if next_statement is not None:
                statements.append(next_statement)
        self.consume(TokenType.RIGHT_BRACE, "Expect } after block")
        return statements
    
    def expressionStatement(self) -> Stmt:
        value: Expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expect ; after value")
        return ExpressionStmt(value)

    def expression(self) -> Expr:
        return self.assignment()
    
    def assignment(self) -> Expr:
        expr: Expr = self.logicalOr()
        if self.match(TokenType.EQUAL):
            equals: Token = self.previous()
            value: Expr = self.assignment()
            if isinstance(expr, Variable):
                return Assign(expr.name, value)
            self.error(equals, "Invalid assignment target")
        return expr
    
    def logicalOr(self) -> Expr:
        expr: Expr = self.logicalAnd()
        while self.match(TokenType.OR):
            operator = self.previous()
            right = self.logicalAnd()
            expr = Logical(expr, operator, right)
        return expr
    
    def logicalAnd(self) -> Expr:
        expr: Expr = self.equality()
        while self.match(TokenType.AND):
            operator = self.previous()
            right = self.equality()
            expr = Logical(expr, operator, right)
        return expr

    def equality(self) -> Expr:
        expr: Expr = self.comparison()
        while self.match(TokenType.BANG_EQUAL, TokenType.EQUAL_EQUAL):
            operator: Token = self.previous()
            right: Expr = self.comparison()
            expr: Expr = Binary(expr, operator, right)
        return expr
    
    def comparison(self) -> Expr:
        expr: Expr = self.term()
        while self.match(TokenType.GREATER, TokenType.GREATER_EQUAL, TokenType.LESS, TokenType.LESS_EQUAL):
            operator: Token = self.previous()
            right: Expr = self.term()
            expr: Expr = Binary(expr, operator, right)
        return expr
    
    def term(self) -> Expr:
        expr: Expr = self.factor()
        while self.match(TokenType.MINUS, TokenType.PLUS):
            operator = self.previous()
            right = self.factor()
            expr = Binary(expr, operator, right)
        return expr
    
    def factor(self) -> Expr:
        expr: Expr = self.unary()
        while self.match(TokenType.SLASH, TokenType.STAR):
            operator = self.previous()
            right = self.unary()
            expr = Binary(expr, operator, right)
        return expr
    
    def unary(self) -> Expr:
        if self.match(TokenType.BANG, TokenType.MINUS):
            operator = self.previous()
            right = self.unary()
            return Unary(operator, right)
        return self.primary()
    
    def primary(self) -> Expr:
        if self.match(TokenType.FALSE):
            return Literal(False)
        if self.match(TokenType.TRUE):
            return Literal(True)
        if self.match(TokenType.NIL):
            return Literal(None)
        if self.match(TokenType.NUMBER, TokenType.STRING):
            return Literal(self.previous().literal)
        if self.match(TokenType.IDENTIFIER):
            return Variable(self.previous())
        if self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expect ')' after expression")
            return Grouping(expr)
        raise self.error(self.peek(), "Expect expression")

    def consume(self, ttype: TokenType, message: str) -> Token:
        if self.check(ttype):
            return self.advance()
        raise self.error(self.peek(), message)
    
    def error(self, token: Token, message: str):
        self.onError(token, message)
        return LoxParseError()
    
    def synchronize(self):
        self.advance()
        while not self.isAtEnd():
            end_stmts = (TokenType.SEMICOLON,)
            if self.previous().type in end_stmts:
                return
            start_stmts = (
                TokenType.CLASS,
                TokenType.FUN,
                TokenType.VAR,
                TokenType.FOR,
                TokenType.IF,
                TokenType.WHILE,
                TokenType.PRINT,
                TokenType.RETURN,
            )
            if self.peek().type in start_stmts:
                return
            self.advance()

    def match(self, *types: TokenType) -> bool:
        for ttype in types:
            if self.check(ttype):
                self.advance()
                return True
        return False
    
    def check(self, ttype: TokenType) -> bool:
        if self.isAtEnd():
            return False
        return self.peek().type == ttype

    def advance(self) -> Token:
        if not self.isAtEnd():
            self.current += 1
        return self.previous()
    
    def previous(self) -> Token:
        return self.tokens[self.current-1]

    def isAtEnd(self) -> bool:
        return self.peek().type == TokenType.EOF

    def peek(self) -> Token:
        return self.tokens[self.current]
    

class Lox:
    def __init__(self):
        self.hasError = False
        self.hasRuntimeError = False
        self.interpreter = Interpreter()
    
    def runFile(self, path: str, showAst: bool):
        with open(path) as fp:
            source = fp.read()
        self.run(source, showAst)
    
    def runPrompt(self, showAst: bool):
        print("> Lox Language interpreter")
        print("> Enter \\q to quit")
        while True:
            line = input("> ")
            if line.strip() == "\\q":
                break
            if line.strip() == "":
                continue
            self.run(line, showAst)
            self.hasError = False
    
    def run(self, source: str, showAst: bool = False):
        scanner = Scanner(source, self.error)
        tokens = scanner.scanTokens()
        parser = Parser(tokens, self.errorByToken)
        statements = parser.parse()
        # for token in tokens:
        #     print(token)
        if self.hasError:
            sys.exit(65)
        self.interpreter.interpret(statements, self.runtimeError)
        if self.hasRuntimeError:
            sys.exit(70)
        if showAst:
            print(AstPrinter().printProgram(statements))
    
    def error(self, line: int, message: str):
        self.report(line, message, "")

    def runtimeError(self, error: LoxRuntimeError):
        print(f"{error}\n[line {error.token.line}]")
        self.hasRuntimeError = True

    def errorByToken(self, token: Token, message: str):
        if token.type == TokenType.EOF:
            self.report(token.line, " at end", message)
        else:
            self.report(token.line, f"at '{token.lexeme}'", message)
    
    def report(self, line: int, message: str, where: Optional[str] = None):
        if where:
            print(f"[line {line}] Error {where}: {message}")
        else:
            print(f"[line {line}]: {message}")
        self.hasError = True


def main(args):
    if args.script:
        Lox().runFile(args.script, args.ast)
    else:
        Lox().runPrompt(args.ast)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--script', default=None)
    parser.add_argument('--ast', action="store_true", default=False)
    main(parser.parse_args())
    