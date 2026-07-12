CC ?= gcc
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
	rm -f ss_motion_bridge $(OBJS)

.PHONY: all strip clean
