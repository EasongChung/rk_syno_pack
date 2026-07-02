#!/bin/sh

PLAYER="/usr/syno/bin/rk_bootanim"
ANIM="/usr/share/bootanimation.zip"

[ -x "$PLAYER" ] || exit 0
[ -f "$ANIM" ] || exit 0

(
	i=0
	while [ "$i" -lt 30 ]; do
		if [ -e /dev/dri/card0 ]; then
			exec "$PLAYER" "$ANIM" >/dev/null 2>&1
		fi
		i=$((i + 1))
		sleep 1
	done
) &
