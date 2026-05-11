CC      = gcc
SRCDIR  = src
UNAME_S := $(shell uname -s)

ifeq ($(OS),Windows_NT)
    EXT         = .dll
    CFLAGS_SO   = -O3 -march=native -fopenmp -shared -lm
    TEST_BIN    = $(SRCDIR)/kernel_test.exe
    CFLAGS_TEST = -O1 -g -Wall -Wextra -lm
    CLEAN_FILES = $(subst /,\,$(SRCDIR)/kernel.so $(SRCDIR)/kernel.dll $(SRCDIR)/kernel_test $(SRCDIR)/kernel_test.exe)
    CLEAN_CMD   = cmd /C del /Q $(CLEAN_FILES) 2>NUL
else ifeq ($(UNAME_S),Darwin)
    EXT         = .so
    CFLAGS_SO   = -O3 -march=native -shared -fPIC -lm
    TEST_BIN    = $(SRCDIR)/kernel_test
    CFLAGS_TEST = -O1 -g -Wall -Wextra -lm
    CLEAN_CMD   = rm -rf $(SRCDIR)/kernel.so $(SRCDIR)/kernel.dll \
                         $(SRCDIR)/kernel_test $(SRCDIR)/kernel_test.exe \
                         $(SRCDIR)/kernel_test.dSYM
else
    EXT         = .so
    CFLAGS_SO   = -O3 -march=native -fopenmp -shared -fPIC -lm
    TEST_BIN    = $(SRCDIR)/kernel_test
    CFLAGS_TEST = -O1 -g -fsanitize=address,undefined -Wall -Wextra -lm
    CLEAN_CMD   = rm -f $(SRCDIR)/kernel.so $(SRCDIR)/kernel.dll \
                        $(SRCDIR)/kernel_test $(SRCDIR)/kernel_test.exe
endif

TARGET = $(SRCDIR)/kernel$(EXT)

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SRCDIR)/kernel.c
	$(CC) $(CFLAGS_SO) -o $@ $<

$(TEST_BIN): $(SRCDIR)/kernel.c
	$(CC) $(CFLAGS_TEST) -DKERNEL_TEST_MAIN -o $@ $<

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	-$(CLEAN_CMD)
