CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS ?=

PREFIX ?= /usr/local

all: ss_shm_tap

ss_shm_tap: ss_shm_tap.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

strip: ss_shm_tap
	$(STRIP) ss_shm_tap

install: ss_shm_tap
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 0755 ss_shm_tap $(DESTDIR)$(PREFIX)/bin/ss_shm_tap

clean:
	rm -f ss_shm_tap

.PHONY: all strip install clean
