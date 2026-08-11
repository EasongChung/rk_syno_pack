CC ?= gcc
HOSTCC ?= cc
CFLAGS ?= -O2 -Wall -Wextra
LDFLAGS ?=

OBJS := ss_motion_bridge.o ss_reclog.o ss_camera.o

all: ss_motion_bridge

ss_motion_bridge: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c ss_reclog.h ss_camera.h
	$(CC) $(CFLAGS) -c -o $@ $<

strip: ss_motion_bridge
	$(STRIP) ss_motion_bridge

clean:
	rm -f ss_motion_bridge test_reclog $(OBJS)

test: test_reclog
	./test_reclog

test_reclog: test_reclog.c ss_reclog.c ss_reclog.h
	$(HOSTCC) -O2 -Wall -Wextra -o $@ test_reclog.c ss_reclog.c

.PHONY: all strip clean test
