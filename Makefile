# Default: build Java
all: build-java

# Java targets
build-java: clean-java
build-java:
	javac -d java java/com/craftinginterpreters/tool/GenerateAst.java
	java -cp java com.craftinginterpreters.tool.GenerateAst ./java/com/craftinginterpreters/lox
	javac -d java java/com/craftinginterpreters/*/*.java

run-java:
	java -cp java com.craftinginterpreters.lox.Lox
	# java -cp java com.craftinginterpreters.lox.AstPrinter

runfile-java:
	java -cp java com.craftinginterpreters.lox.Lox $(FILE)

clean-java:
	rm java/com/craftinginterpreters/lox/Expr.java
	rm java/com/craftinginterpreters/*/*.class

# Python targets
run-python:
	python python/lox.py

runfile-python:
	python python/lox.py --script $(FILE)

clean-python:
	rm -rf python/__pycache__

# Clean all
clean: clean-java clean-python

.PHONY: all build-java run-java runfile-java clean-java clean-python run-python runfile-python clean