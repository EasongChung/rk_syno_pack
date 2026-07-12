CC ?= gcc
STAGE ?= ../../build/ffmpeg-spk-work/stage
MPP_INC ?= $(STAGE)/usr/include/rockchip
MPP_LIB ?= $(STAGE)/usr/lib
CFLAGS ?= -O2 -Wall -Wextra
CPPFLAGS += -I$(MPP_INC) -I../ss_motion_bridge
LDFLAGS += -L$(MPP_LIB)
LDLIBS += -lrockchip_mpp -lpthread

OBJS := ss_mpp_motiond.o ss_reclog.o ss_camera.o

all: ss_mpp_motiond

ss_mpp_motiond: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

ss_mpp_motiond.o: ss_mpp_motiond.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

ss_reclog.o: ../ss_motion_bridge/ss_reclog.c ../ss_motion_bridge/ss_reclog.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

ss_camera.o: ../ss_motion_bridge/ss_camera.c ../ss_motion_bridge/ss_camera.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f ss_mpp_motiond $(OBJS)

.PHONY: all clean
