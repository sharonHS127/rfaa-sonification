# macOS (Apple Silicon, Homebrew FFTW):  make          Ubuntu: sudo apt install libfftw3-dev && make
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  FFTW_PREFIX ?= /opt/homebrew
  CPPFLAGS += -I$(FFTW_PREFIX)/include
  LDFLAGS  += -L$(FFTW_PREFIX)/lib
endif
CC      ?= cc
CFLAGS  ?= -std=gnu11 -O2 -Wall -Wextra
LDLIBS  = -lfftw3f -lm

all: test_golden sonify_cli

sonify.o: sonify.c sonify.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@
test_golden: test_golden.c sonify.o sonify.h
	$(CC) $(CFLAGS) $(CPPFLAGS) test_golden.c sonify.o $(LDFLAGS) $(LDLIBS) -o $@
sonify_cli: sonify_cli.c sonify.o sonify.h
	$(CC) $(CFLAGS) $(CPPFLAGS) sonify_cli.c sonify.o $(LDFLAGS) $(LDLIBS) -o $@

check: test_golden
	./test_golden golden
clean:
	rm -f sonify.o test_golden sonify_cli
.PHONY: all check clean
