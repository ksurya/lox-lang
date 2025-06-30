# Default: build Java
all: build-java

# Java targets
build-java:
	javac -d java java/com/craftinginterpreters/lox/*.java

run-java: build-java
	java -cp java com.craftinginterpreters.lox.Lox

runfile-java: build-java
	java -cp java com.craftinginterpreters.lox.Lox $(FILE)

clean-java:
	rm java/com/craftinginterpreters/lox/*.class

# Python targets
run-python:
	python3 $(PY_MAIN)

runfile-python:
	python3 $(PY_MAIN) $(FILE)

# Clean all
clean: clean-java

.PHONY: all build-java run-java runfile-java clean-java run-python runfile-python clean