#!/bin/sh
# 组装 aarch64 + glibc 的 C/C++ sysroot（Debian bullseye）
#
# 用途：为 TFLite 运行时交叉编译提供 sysroot。
# 与 build-teflon.sh 的区别：
#   1. build-teflon.sh 只装 libc6/libc6-dev/linux-libc-dev（C 足够，Mesa 用）
#   2. TFLite 是 C++，还必须有 libstdc++ 的头与库，否则 g++ 编不过
#
# 为什么必须用 Debian bullseye（glibc 2.31）而不是 apt 的 arm64 交叉包：
#   Ubuntu 24.04(noble) 的 libc6-dev-arm64-cross 是 glibc 2.39，
#   而设备只有 glibc 2.36。glibc ≥2.38 在 stdlib.h 里做了
#       #define strtol  __isoc23_strtol      （版本节点 GLIBC_2.38）
#   设备 2.36 没有这些符号 → 产物上机 dlopen 直接失败。
#   bullseye 2.31 产出最高 GLIBC_2.34，安全低于设备 2.36。
#
# 包版本从 Debian 的 Packages 索引里动态解析，不写死版本串
#   （写死会在 Debian 安全更新后失效，且难排查）。
#
# 用法： sh make-sysroot-cxx.sh <目标 sysroot 目录>
# 输出： sysroot 就绪后打印其绝对路径

set -eu

BUILD_DIR="${BUILD_DIR:-/tmp/sysroot-cxx}"
DEB_SUITE="${DEB_SUITE:-bullseye}"
DEB_BASE="http://deb.debian.org/debian"
PKGLIST="http://deb.debian.org/debian/dists/${DEB_SUITE}/main/binary-arm64/Packages.gz"

SYS_TRIPLET="aarch64-linux-gnu"

echo "==> 组装 aarch64 C++ sysroot"
echo "    Debian 套件: $DEB_SUITE（glibc 2.31，设备需 ≥ 2.34 符号 ≤ 2.36）"

command -v wget >/dev/null 2>&1 || { echo "ERROR: 需要 wget"; exit 1; }

# --- 1. 拉包索引 -------------------------------------------------------
IDX="$BUILD_DIR/Packages"
if [ ! -f "$IDX" ]; then
	mkdir -p "$BUILD_DIR"
	echo "    下载包索引（用于解析包版本，避免写死）"
	wget -q -O "$BUILD_DIR/Packages.gz" "$PKGLIST"
	gzip -dc "$BUILD_DIR/Packages.gz" > "$IDX"
fi

# resolve_pkg <包名> → 打印 pool 相对路径
# 用 awk 在该包的 stanza 里取 Filename 字段。
resolve_pkg() {
	awk -v want="Package: $1" '
		$0 == want { inblk = 1; next }
		inblk && /^Filename:/ { print $2; exit }
		inblk && /^$/ { exit }
	' "$IDX"
}

# --- 2. 下载并解包 -----------------------------------------------------
# fetch <包名> <目标 deb 文件名>
fetch() {
	pkg="$1"; name="$2"
	dst="$BUILD_DIR/$name"
	[ -f "$dst" ] && { echo "    已存在 $name"; return; }
	path=$(resolve_pkg "$pkg")
	if [ -z "$path" ]; then
		echo "ERROR: 索引里找不到包 $pkg"
		exit 1
	fi
	echo "    下载 $name  ← $path"
	wget -q -O "$dst" "$DEB_BASE/$path"
}

# unpack <deb 文件名> <目标 sysroot>
unpack() {
	deb="$1"; dst="$2"
	echo "    解包 $deb"
	rm -rf "$BUILD_DIR/_x"; mkdir -p "$BUILD_DIR/_x"
	( cd "$BUILD_DIR/_x" \
		&& ar x "$BUILD_DIR/$deb" \
		&& tar xf data.tar.* \
		&& rm -f data.tar.* control.tar.* debian-binary )
	cp -a "$BUILD_DIR/_x/." "$dst/"
}

SR="$1"
if [ -z "$SR" ]; then
	echo "用法: sh $0 <目标 sysroot 目录>"
	exit 1
fi
mkdir -p "$SR"

# C 部分：libc6 / libc6-dev / linux-libc-dev
# 这些走固定路径（与 build-teflon.sh 已验证的一致），
# 但版本仍从索引解析，只是不依赖它是否存在。
echo "    [C] libc6 / libc6-dev / linux-libc-dev"
for pkg in libc6 libc6-dev linux-libc-dev; do
	fetch "$pkg" "$pkg.deb"
	unpack "$pkg.deb" "$SR"
done

# C++ 部分：libstdc++ 的开发头 + 运行时 .so
# 缺了这两样，g++ 交叉编译 TFLite 会直接失败。
#
# ⚠️ 重要：本脚本下载的 libstdc++ 是 **Debian bullseye 的 GCC 10** 版本。
#    若你的交叉编译器是 Ubuntu 的 g++-aarch64-linux-gnu（Ubuntu 24.04 = GCC 13），
#    **绝对不要把 $SR/usr/include/c++ 加进 -isystem**：
#
#      /usr/aarch64-linux-gnu/include/c++/13/...   ← 编译器自带（GCC 13）
#      $SR/usr/include/c++/10/type_traits          ← 本脚本放的（GCC 10）
#      → _GLIBCXX20_DEPRECATED 宏未定义 → 一片语法错误
#
#    libstdc++ 头与编译器版本强绑定，跨版本混用必然失败。
#    正确做法：sysroot 只提供 libc，libstdc++ 用编译器自带的，
#    并加 -static-libstdc++ -static-libgcc 静态链入。
#    （保留这里的下载是为了在「编译器恰好也是 GCC 10」时可选用。）
#
# 包名坑：Debian bullseye 里没有「libstdc++-dev」这个包名，它只是
# libstdc++-10-dev 的 Provides（虚拟包），索引里查不到，
# 实际包名是版本化的「libstdc++-10-dev」（gcc-10 对应）。
# 故先用模式匹配取版本，再回退到固定名。
echo "    [C++] libstdc++-dev / libstdc++6"

# resolve_pkg_pat <正则> —— 打印首个匹配包的 Filename
#
# 为什么不用「取最高版本」：同一 suite 里 native 的 libstdc++-N-dev 只有一个，
# 取首个即可；而且 awk 里做版本字符串比较容易在 9.x / 10.x 上判错
# （字符串比较下 "10.2.1" < "9.3.0"），反而不可靠。
resolve_pkg_pat() {
	awk -v pat="$1" '
		BEGIN { re = "^Package: " pat "$" }
		$0 ~ re { f = 1; next }
		f && /^Filename:/ { print $2; exit }
		f && /^Package: / { exit }
	' "$IDX"
}

# ⚠️ 模式里不要写 \+ ：`awk -v pat="$1"` 会对赋值值做反斜杠转义处理，
#    `\+\+` 会被吃掉反斜杠变成 `++`，正则语义从「两个字面加号」变成
#    「一个加号重复一次以上」，静默匹配不到。
#    这里用 `..` 匹配 `++`，在 busybox awk 与 GNU awk 下行为一致。
cxxdev=$(resolve_pkg_pat 'libstdc..-[0-9]+-dev')
[ -z "$cxxdev" ] && cxxdev=$(resolve_pkg "libstdc++-dev")
if [ -z "$cxxdev" ]; then
	echo "ERROR: 找不到 libstdc++ 开发包（试过 libstdc++-N-dev 与 libstdc++-dev）"
	exit 1
fi

for pkg in libstdc++6; do
	fetch "$pkg" "$pkg.deb"
	unpack "$pkg.deb" "$SR"
done

echo "    下载 libstdc++-dev.deb  ← $cxxdev"
wget -q -O "$BUILD_DIR/libstdc++-dev.deb" "$DEB_BASE/$cxxdev"
unpack "libstdc++-dev.deb" "$SR"

# libc6-dev 不提供 libssp_nonshared.a，
# 而部分 gcc 规格会链 -lssp_nonshared → 补空档，否则链接报 undefined。
echo "    补 libssp_nonshared stub"
STUB_LIB="$SR/usr/lib/${SYS_TRIPLET}/libssp_nonshared.a"
if [ ! -f "$STUB_LIB" ]; then
	echo 'int __oca_ssp_stub_unused;' > "$BUILD_DIR/_stub.c"
	( cd "$BUILD_DIR" && gcc -c -o _stub.o _stub.c && ar rcs "$STUB_LIB" _stub.o )
	rm -f "$BUILD_DIR/_stub.c" "$BUILD_DIR/_stub.o"
fi

# g++ 在 sysroot 里找 libstdc++ 头时看的是这个路径
# （不建的话 CMake 探测 stdlib.h 会失败）
mkdir -p "$SR/usr/include/c++"

echo "    sysroot 就绪: $SR"
echo "    glibc 头:    $SR/usr/include"
echo "    aarch64 库:  $SR/usr/lib/${SYS_TRIPLET}"
echo "    aarch64 库:  $SR/lib/aarch64-linux-gnu"