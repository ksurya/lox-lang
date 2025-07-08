package com.craftinginterpreters.lox;
import java.util.List;
import java.util.ListIterator;

class AstPrinter implements Expr.Visitor<String>, Stmt.Visitor<String> {
  String print(Expr expr) {
    return expr.accept(this);
  }

  String print(Stmt stmt) {
    return stmt.accept(this);
  }

  String print(List<Stmt> statements) {
    StringBuilder builder = new StringBuilder();
    ListIterator<Stmt> iterator = statements.listIterator();
    builder.append("(program ");
    while (iterator.hasNext()) {
      Stmt statement = iterator.next();
      builder.append(statement.accept(this));
      if (iterator.hasNext()) {
        builder.append("\n");
      }
    }
    builder.append(")");
    return builder.toString();
  }


  @Override
  public String visitBlockStmt(Stmt.Block stmt) {
    StringBuilder builder = new StringBuilder();
    builder.append("(block ");
    for (Stmt statement : stmt.statements) {
      builder.append(statement.accept(this));
    }
    builder.append(")");
    return builder.toString();
  }

  @Override
  public String visitExpressionStmt(Stmt.Expression stmt) {
    return parenthesize(";", stmt.expression);
  }

  @Override
  public String visitPrintStmt(Stmt.Print stmt) {
    return parenthesize("print", stmt.expression);
  }

  @Override
  public String visitVarStmt(Stmt.Var stmt) {
    StringBuilder builder = new StringBuilder();
    builder.append("(var ");
    builder.append(stmt.name.lexeme);
    if (stmt.initializer != null) {
      builder.append("=");
      builder.append(stmt.initializer.accept(this));
    }
    builder.append(")");
    return builder.toString();
  }

  @Override
  public String visitAssignExpr(Expr.Assign expr) {
    StringBuilder builder = new StringBuilder();
    builder.append("(");
    builder.append(expr.name.lexeme);
    builder.append("=");
    builder.append(expr.value.accept(this));
    builder.append(")");
    return builder.toString();
  }

  @Override
  public String visitBinaryExpr(Expr.Binary expr) {
    return parenthesize(expr.operator.lexeme,
                        expr.left, expr.right);
  }

  @Override
  public String visitGroupingExpr(Expr.Grouping expr) {
    return parenthesize("", expr.expression);
  }

  @Override
  public String visitLiteralExpr(Expr.Literal expr) {
    if (expr.value == null) {
      return "nil";
    }
    return expr.value.toString();
  }

  @Override
  public String visitUnaryExpr(Expr.Unary expr) {
    return parenthesize(expr.operator.lexeme, expr.right);
  }

  @Override
  public String visitVariableExpr(Expr.Variable expr) {
    return expr.name.lexeme;
  }

  private String parenthesize(String name, Expr... exprs) {
    StringBuilder builder = new StringBuilder();

    builder.append("(").append(name);
    for (Expr expr : exprs) {
      builder.append(" ");
      builder.append(expr.accept(this));
    }
    builder.append(")");

    return builder.toString();
  }

  public static void main(String[] args) {
    Expr expression = new Expr.Binary(
        new Expr.Unary(
            new Token(TokenType.MINUS, "-", null, 1),
            new Expr.Literal(123)),
        new Token(TokenType.STAR, "*", null, 1),
        new Expr.Grouping(
            new Expr.Literal(45.67)));

    System.out.println(new AstPrinter().print(expression));
  }
}