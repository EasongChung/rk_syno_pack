#!/bin/sh
set -eu

_exit()
{
	printf '[rk3399-initrd] error: %s\n' "$*" >&2
	exit 1
}

log()
{
	printf '[rk3399-initrd] %s\n' "$*"
}

need_file()
{
	[ -r "$1" ] || _exit "$1 is missing"
}

install_model_sync_hook()
{
	root="$1"
	linuxrc="$root/linuxrc.syno.impl"
	script="$root/usr/syno/sbin/rk3399_sync_model_dtb.sh"
	hook='/bin/sh /usr/syno/sbin/rk3399_sync_model_dtb.sh "$Mnt"'

	need_file "$linuxrc"
	mkdir -p "$(dirname "$script")"
	cat >"$script" <<'EOF'
#!/bin/sh

RootMnt="$1"

[ -n "$RootMnt" ] || exit 0
[ -d "$RootMnt" ] || exit 0

set_synoinfo_kv()
{
	file="$1"
	key="$2"
	value="$3"
	tmp="${file}.rk3399.$$"

	[ -f "$file" ] || return 0

	/bin/grep -v "^${key}=" "$file" > "$tmp" 2>/dev/null || true
	echo "${key}=\"${value}\"" >> "$tmp"
	/bin/mv -f "$tmp" "$file"
}

copy_model_dtb()
{
	dir="$1"

	[ -d "$RootMnt/$dir" ] || return 0
	[ -f "/$dir/model.dtb" ] || return 0

	/bin/cp -f "/$dir/model.dtb" "$RootMnt/$dir/model.dtb"
}

sync_synoinfo_disk_count()
{
	max_disks="$(/bin/get_key_value /etc.defaults/synoinfo.conf maxdisks 2>/dev/null)"
	[ -n "$max_disks" ] || max_disks=6

	for dir in etc.defaults etc; do
		[ -d "$RootMnt/$dir" ] || continue
		set_synoinfo_kv "$RootMnt/$dir/synoinfo.conf" maxdisks "$max_disks"
		set_synoinfo_kv "$RootMnt/$dir/synoinfo.conf" max_sys_raid_disks "$max_disks"
	done
}

copy_model_dtb etc.defaults
copy_model_dtb etc
sync_synoinfo_disk_count
EOF
	chmod 0755 "$script"

	if grep -Fq "$hook" "$linuxrc"; then
		log "model.dtb root sync hook already installed"
		return 0
	fi

	log "installing model.dtb root sync hook"
	awk -v hook="$hook" '
		!done && $0 == "if [ -f ${Mnt}/.noroot ]; then" {
			print hook
			print ""
			done = 1
		}
		{ print }
		END {
			if (!done)
				exit 1
		}
	' "$linuxrc" >"$linuxrc.rk3399"
	mv -f "$linuxrc.rk3399" "$linuxrc"
	chmod 0755 "$linuxrc"
}

fix_webman_reboot()
{
	root="$1"
	reboot_cgi="$root/usr/syno/web/webman/reboot.cgi"

	if [ ! -f "$reboot_cgi" ]; then
		log "webman reboot.cgi not found, skipping"
		return 0
	fi
	if grep -Fq '/sbin/reboot -f' "$reboot_cgi"; then
		log "webman reboot already uses forced reboot"
		return 0
	fi

	log "patching webman reboot to use /sbin/reboot -f"
	sed -i 's|^\([[:space:]]*\)reboot[[:space:]]*$|\1/sbin/reboot -f|' "$reboot_cgi"
	grep -Fq '/sbin/reboot -f' "$reboot_cgi" || \
		_exit "failed to patch $reboot_cgi"
}

main()
{
	root="${1:-.}"

	[ -d "$root" ] || _exit "$root is not a directory"

	install_model_sync_hook "$root"
	fix_webman_reboot "$root"
}

main "$@"
