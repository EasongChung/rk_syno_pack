#!/bin/sh
# 准备 aarch64 + glibc 的 sysroot（本机 Alpine musl → 目标 DSM glibc）
#
# 背景：本机与目标同为 aarch64，唯一差异是 libc（musl vs glibc），
# 因此不需要异构交叉编译器，只需要一个 glibc sysroot 供 gcc 使用。
#
# 幂等：已存在则跳过下载。用法：./prep-sysroot.sh [sysroot 目录]

set -eu

SR="${1:-/root/workspace/oes-teflon/sysroot}"
DEB_BASE="http://deb.debian.org/debian/pool/main"
GLIBC_VER="2.31-13+deb11u11"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> sysroot 目标: $SR"
mkdir -p "$SR"

# 1) 下载 Debian arm64 包
fetch() {
	url="$1"; out="$2"
	if [ ! -f "$out" ]; then
		echo "    下载 $out"
		wget -q -O "$out" "$url"
	else
		echo "    已存在 $out，跳过"
	fi
}

echo "==> 获取 glibc ${GLIBC_VER} (arm64)"
fetch "$DEB_BASE/g/glibc/libc6_${GLIBC_VER}_arm64.deb"            "$TMP/libc6.deb"
fetch "$DEB_BASE/g/glibc/libc6-dev_${GLIBC_VER}_arm64.deb"        "$TMP/libc6-dev.deb"
fetch "$DEB_BASE/l/linux/linux-libc-dev_6.1.176-1_arm64.deb"     "$TMP/linux-libc-dev.deb"

# 2) 解包合并
echo "==> 解包"
for deb in libc6 libc6-dev linux-libc-dev; do
	rm -rf "$TMP/x"; mkdir -p "$TMP/x"; ( cd "$TMP/x" \
		&& ar x "$TMP/$deb.deb" && tar xf data.tar.* \
		&& rm -f data.tar.* control.tar.* debian-binary )
	cp -a "$TMP/x/." "$SR/"
done

# 3) stub：Alpine gcc 的默认规格会链 -lssp_nonshared，
#    而 Debian libc6-dev 不提供该库，必须补一个空档，否则链接失败。
#    实测报错：cannot find -lssp_nonshared: No such file or directory
echo "==> 补 libssp_nonshared stub"
STUB_LIB="$SR/usr/lib/aarch64-linux-gnu/libssp_nonshared.a"
if [ ! -f "$STUB_LIB" ]; then
	echo 'int __oca_ssp_stub_unused;' > "$TMP/stub.c"
	( cd "$TMP" && gcc -c -o stub.o stub.c && ar rcs "$STUB_LIB" stub.o )
	echo "    已生成 stub"
else
	echo "    已存在，跳过"
fi

echo "==> sysroot 就绪: $SR"
echo
echo "使用方式（注意 multiarch include 与 -B/-L）："
cat <<EOF
CFLAGS="--sysroot=$SR \\
  -isystem $SR/usr/include/aarch64-linux-gnu \\
  -B $SR/usr/lib/aarch64-linux-gnu \\
  -L $SR/usr/lib/aarch64-linux-gnu"
EOF