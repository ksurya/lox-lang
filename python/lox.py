#!/usr/bin/env python
from argparse import ArgumentParser
from enum import StrEnum, auto
from typing import Any, Callable, Optional
import sys


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
    FLASE = auto()
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


class Scanner:
    keywords: dict[str, TokenType] = {
        "and":      TokenType.AND,
        "class":    TokenType.CLASS,
        "class":    TokenType.CLASS,
        "else":     TokenType.ELSE,
        "false":    TokenType.FLASE,
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
                self.advance()
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


class Lox:
    def __init__(self):
        self.hasError = False
    
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
        for token in tokens:
            print(token)
        if self.hasError:
            sys.exit(65)
    
    def error(self, line: int, message: str):
        self.report(line, message, "")
    
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
    else:
        lox.runPrompt()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--script', default=None)
    main(parser.parse_args())
    