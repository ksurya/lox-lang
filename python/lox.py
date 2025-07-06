#!/usr/bin/env python
from argparse import ArgumentParser
from enum import StrEnum, auto
from typing import Any, Callable, Optional, Union, Generic, TypeVar
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


class Expr(abc.ABC):
    @abc.abstractmethod
    def accept(self, visitor: "Visitor[R]") -> R: 
        pass


class Visitor(abc.ABC, Generic[R]):
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
    def visitUnaryExpr(self, expr: "Unary") -> R:
        pass


class Binary(Expr):
    def __init__(self, left: Expr, operator: Token, right: Expr):
        self.left: Expr = left
        self.operator: Token = operator
        self.right: Expr = right

    def accept(self, visitor: Visitor[R]) -> R:
        return visitor.visitBinaryExpr(self)


class Grouping(Expr):
    def __init__(self, expression: Expr):
        self.expression: Expr = expression

    def accept(self, visitor: Visitor[R]) -> R:
        return visitor.visitGroupingExpr(self)


class Literal(Expr):
    def __init__(self, value: Union[str, int, float, None]):
        self.value: Union[str, int, float, None] = value

    def accept(self, visitor: Visitor[R]) -> R:
        return visitor.visitLiteralExpr(self)


class Unary(Expr):
    def __init__(self, operator: Token, right: Expr):
        self.operator: Token = operator
        self.right: Expr = right

    def accept(self, visitor: Visitor[R]) -> R:
        return visitor.visitUnaryExpr(self)


class AstPrinter(Visitor[str]):
    def _parenthesize(self, name: str, *exprs: Expr):
        string_container = ["(", name]
        for expr in exprs:
            string_container.append(" ")
            string_container.append(expr.accept(self))
        string_container.append(")")
        return "".join(string_container)
    
    def print(self, expr: Expr) -> str:
        return expr.accept(self)

    def visitBinaryExpr(self, expr: Binary) -> str:
        return self._parenthesize(expr.operator.lexeme, expr.left, expr.right)

    def visitGroupingExpr(self, expr: Grouping) -> str:
        return self._parenthesize("group", expr.expression)
    
    def visitLiteralExpr(self, expr: Literal) -> str:
        if expr.value is None:
            return "nil"
        else:
            return str(expr.value)
    
    def visitUnaryExpr(self, expr: Unary) -> str:
        return self._parenthesize(expr.operator.lexeme, expr.right)

    @staticmethod
    def test():
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
        print(AstPrinter().print(expr))


class LoxRuntimeError(RuntimeError):
    def __init__(self, token: Token, message: str):
        super().__init__(message)
        self.token: Token = token


class Interpreter(Visitor[object]):

    def interpret(self, expr: Expr, onError: Callable):
        try:
            value: object = self.evaluate(expr)
            print(self.stringify(value))
        except LoxRuntimeError as err:
            onError(err)

    def stringify(self, obj: object) -> str:
        if obj == None:
            return "nil"
        return str(obj)

    def evaluate(self, expr: Expr) -> object:
        return expr.accept(self)
    
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
    
    def visitGroupingExpr(self, expr: Grouping) -> object:
        return self.evaluate(expr.expression)
    
    def visitUnaryExpr(self, expr: Unary) -> object:
        right: object = self.evaluate(expr.right)
        if expr.operator.type == TokenType.BANG:
            return not self.isTruthy(right)
        if expr.operator.type == TokenType.MINUS:
            self.checkNumberOperand(expr.operator, right)
            return -float(right)

    def visitBinaryExpr(self, expr: Binary) -> object:
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)
        if expr.operator.type == TokenType.GREATER:
            self.checkNumberOperands(expr.operator, left, right)
            return float(left) > float(right)
        elif expr.operator.type == TokenType.GREATER_EQUAL:
            self.checkNumberOperands(expr.operator, left, right)
            return float(left) >= float(right)
        elif expr.operator.type == TokenType.LESS:
            self.checkNumberOperands(expr.operator, left, right)
            return float(left) < float(right)
        elif expr.operator.type == TokenType.LESS_EQUAL:
            self.checkNumberOperands(expr.operator, left, right)
            return float(left) <= float(right)
        elif expr.operator.type == TokenType.BANG_EQUAL:
            return left != right
        elif expr.operator.type == TokenType.EQUAL_EQUAL:
            return left == right
        elif expr.operator.type == TokenType.MINUS:
            self.checkNumberOperands(expr.operator, left, right)
            return float(left) - float(right)
        elif expr.operator.type == TokenType.PLUS:
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return float(left) + float(right)
            if isinstance(left, str) and isinstance(right, str):
                return str(left) + str(right)
            raise LoxRuntimeError(expr.operator, "Operands must be two numbers or two strings")
        elif expr.operator.type == TokenType.SLASH:
            self.checkNumberOperands(expr.operator, left, right)
            return float(left) / float(right)
        elif expr.operator.type == TokenType.STAR:
            self.checkNumberOperands(expr.operator, left, right)
            return float(left) * float(right)


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


class ParseError(RuntimeError):
    pass


class Parser:
    """Grammar
    expression     → equality ;
    equality       → comparison ( ( "!=" | "==" ) comparison )* ;
    comparison     → term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
    term           → factor ( ( "-" | "+" ) factor )* ;
    factor         → unary ( ( "/" | "*" ) unary )* ;
    unary          → ( "!" | "-" ) unary | primary ;
    primary        → NUMBER | STRING | "true" | "false" | "nil" | "(" expression ")" ;
    """
    def __init__(self, tokens: list[Token], onError: Callable):
        self.tokens: list[Token] = tokens
        self.current: int = 0
        self.onError: Callable = onError

    def parse(self) -> Expr:
        try:
            return self.expression()
        except ParseError as err:
            return None

    def expression(self) -> Expr:
        return self.equality()

    def equality(self) -> Expr:
        expr: Expr = self.comparison()
        while self.match(TokenType.BANG_EQUAL, TokenType.EQUAL_EQUAL):
            operator: Token = self.previous()
            right: Token = self.comparison()
            expr: Expr = Binary(expr, operator, right)
        return expr
    
    def comparison(self) -> Expr:
        expr: Expr = self.term()
        while self.match(TokenType.GREATER, TokenType.GREATER_EQUAL, TokenType.LESS, TokenType.LESS_EQUAL):
            operator: Expr = self.previous()
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
        if self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN, "Expec ')' after expression")
            return Grouping(expr)
        raise self.error(self.peek(), "Expect expression")

    def consume(self, ttype: TokenType, message: str) -> Token:
        if self.check(ttype):
            return self.advance()
        raise self.error(self.peek(), message)
    
    def error(self, token: Token, message: str):
        self.onError(token, message)
        return ParseError()
    
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
    
    def runFile(self, path: str):
        with open(path) as fp:
            source = fp.read()
        self.run(source)
    
    def runPrompt(self):
        print("> Lox interpreter (enter \\q to exit)")
        while True:
            line = input("> ")
            if line.strip() == "\\q":
                break
            self.run(line)
            self.hasError = False
    
    def run(self, source: str):
        scanner = Scanner(source, self.error)
        tokens = scanner.scanTokens()
        parser = Parser(tokens, self.errorByToken)
        expression = parser.parse()
        # for token in tokens:
        #     print(token)
        if self.hasError:
            sys.exit(65)
        self.interpreter.interpret(expression, self.runtimeError)
        if self.hasRuntimeError:
            sys.exit(70)
        # print(AstPrinter().print(expression))
    
    def error(self, line: int, message: str):
        self.report(line, message, "")

    def runtimeError(self, error: RuntimeError):
        print(f"{error}\n[line {error.token.line}]")
        self.hasRuntimeError = True

    def errorByToken(self, token: Token, message: str):
        if token.type == TokenType.EOF:
            self.report(token.line, " at end", message)
        else:
            self.report(f"{token.line} at '{token.lexeme}'", message)
    
    def report(self, line: int, message: str, where: Optional[str] = None):
        if where:
            print(f"[line {line}] Error {where}: {message}")
        else:
            print(f"[line {line}]: {message}")
        self.hasError = True


def main(args):
    lox = Lox()
    if args.script:
        lox.runFile(args.script)
    elif args.ast:
        AstPrinter.test()
    else:
        lox.runPrompt()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--script', default=None)
    parser.add_argument('--ast', action="store_true")
    main(parser.parse_args())
    