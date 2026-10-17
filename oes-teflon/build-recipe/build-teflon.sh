#!/bin/sh
# Teflon 交叉编译 —— 本地/容器可复用逻辑
#
# 与 .github/workflows/build-teflon.yml 共用同一套步骤。
# 目标：aarch64 + glibc 的 libteflon.so
#
# 关键设计：
#   - Mesa 自带 vendored TFLite 头（include/tensorflow，来源 TF 2.13.0）
#     因此**不需要**下载/编译 TensorFlow，这是最容易踩空的一点
#   - libdrm 需要 cross 版本的 dev 包（宿主 x86_64 的 libdrm 不可用）
#   - glibc 版本由 GLIBC_VER 控制，默认 2.31（保守值）
#
# 用法：
#   ./build-teflon.sh [构建目录]
# 环境变量：
#   MESA_TAG       Mesa 版本/tag，默认 mesa-24.3.0
#   GLIBC_VER      sysroot glibc 版本，默认 2.31
#   DEB_SUITE      Debian 套件，默认 bullseye
#   CROSS_PREFIX   交叉前缀，默认 aarch64-linux-gnu-
#
# ── 诊断/修复开关（**默认全关**，不设时产物与生产构建完全一致）────────────
#   TEFLON_DEBUG_INFO=1   c_args/cpp_args 追加 -g3（带调试信息）
#   TEFLON_NO_STRIP=1     跳过 strip，保留 DWARF（.debug_info/.debug_line）
#                         → addr2line 可直出源码行；产物明显变大且**不可交付**
#   TEFLON_FIX_V7=1       打 v7 calculate_bias_correction 的 depthwise 特判补丁
#                         （真实的上游不一致：v8 有、v7 没有）
#   TEFLON_DIAG=1         打越界取证补丁（打印尺寸/指针 + 预测到越界时钳界）
#                         ⚠️ 与 TEFLON_FIX_V7 语义互斥，只能二选一
#
#   典型诊断构建：TEFLON_FIX_V7=1 TEFLON_DEBUG_INFO=1 TEFLON_NO_STRIP=1
#   ⚠️ 带任一诊断开关的产物**只能用于诊断，不可交付**。
#   ⚠️ 不要手改脚本里的 strip / c_args 段来「等效实现」这些开关 ——
#      那会绕过日志断言与互斥检查（这个坑已经踩过一次）。
# ------------------------------------------------------------------------

set -eu

# 脚本自身所在目录。**必须在任何 cd 之前捕获** —— 脚本后面会 `cd "$BUILD_DIR"`，
# 而传入的 $0 通常是相对路径（如 ./oes-teflon/build-recipe/build-teflon.sh），
# cd 之后再解析就找不到自己了（CI 上实测报 `can't cd to ./oes-teflon/build-recipe`）。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BUILD_DIR="${1:-mesa-build}"
MESA_TAG="${MESA_TAG:-mesa-24.3.0}"
GLIBC_VER="${GLIBC_VER:-2.31}"
DEB_SUITE="${DEB_SUITE:-bullseye}"
# 模式选择：
#   native   —— 宿主本身就是 aarch64，直接用本机 gcc（需 glibc sysroot）
#   cross    —— 用 aarch64-linux-gnu-* 交叉编译器（CI 默认）
# 原生模式典型用法：NATIVE=1 SYSROOT_MODE=deb ./build-teflon.sh
NATIVE="${NATIVE:-0}"
if [ "$NATIVE" = "1" ]; then
	CROSS_PREFIX=""
	 SYS_TRIPLET="aarch64-linux-gnu"   # sysroot 内部布局仍是 aarch64-linux-gnu
	[ "$SYSROOT_MODE" = "apt" ] && SYSROOT_MODE="deb"
else
	CROSS_PREFIX="${CROSS_PREFIX:-aarch64-linux-gnu-}"
fi
# sysroot 内部 triplet 恒为 aarch64-linux-gnu（Debian/Ubuntu 布局），
# 与编译器前缀无关——本机原生编译时 CROSS_PREFIX 为空，
# 但 sysroot 里的路径仍必须是 aarch64-linux-gnu。
# 注意：这里是目录名（不带尾部连字符），不能复用 CROSS_PREFIX 的值。
SYS_TRIPLET="${SYS_TRIPLET:-aarch64-linux-gnu}"

# Debian 的 glibc 完整版本串（含 deb11u11 之类修订号）
GLIBC_FULL="${GLIBC_FULL:-$GLIBC_VER-13+deb11u11}"
DEB_BASE="http://deb.debian.org/debian/pool/main"

echo "=============================================="
echo " Mesa tag    : $MESA_TAG"
echo " glibc target: $GLIBC_FULL ($DEB_SUITE)"
echo " cross prefix: $CROSS_PREFIX"
echo " build dir   : $BUILD_DIR"
echo "=============================================="

# ---------------------------------------------------------------- #
# 1. sysroot
#
# 两种模式：
#   SYSROOT_MODE=apt    （CI 默认）sysroot 由 Ubuntu 的 arm64 包自动
#                       填到 /usr/aarch64-linux-gnu，无需手工下载
#   SYSROOT_MODE=deb    （musl 宿主）从 Debian 池下载 .deb 自行拼 sysroot
# ---------------------------------------------------------------- #
# 默认用 deb 模式（从 Debian 池装 glibc 2.31 sysroot），而不是 apt 的
# aarch64 交叉包。原因：Ubuntu 24.04 的 libc6-dev-arm64-cross 是 glibc 2.39，
# 而设备只有 2.36；glibc ≥2.38 的 stdlib.h 会把 strtol/strtoll 重定向到
# __isoc23_*（GLIBC_2.38 版本节点），设备 2.36 无此符号 → dlopen 必失败。
# deb 模式 sysroot 固定在 2.31，产物最高 GLIBC_2.29，安全。
# 若确实要用 apt sysroot，需显式 SYSROOT_MODE=apt，并自行确认版本 ≤ 2.36。
SR_MODE="${SYSROOT_MODE:-deb}"
mkdir -p "$BUILD_DIR/sysroot"

if [ "$SR_MODE" = "apt" ]; then
	SR="/usr/${CROSS_PREFIX}"
	echo "==> [1/6] 使用 apt 提供的 cross sysroot: $SR"
	if [ ! -d "$SR/include" ]; then
		echo "ERROR: sysroot 不存在。请确认已安装："
		echo "  libc6-dev-arm64-cross linux-libc-dev-arm64-cross"
		echo "  libdrm-dev:arm64 zlib1g-dev:arm64"
		exit 1
	fi

	# Ubuntu 上有两套布局，.pc 文件位置不同，必须都覆盖：
	#   A) libc6-dev-arm64-cross 等交叉包 → $SR/lib/pkgconfig
	#   B) libdrm-dev:arm64 等 foreign-arch 包 → /usr/lib/<triplet>/pkgconfig
	# 只设 A 会导致 meson 找不到 libdrm/zlib 的 .pc，
	# 进而 "Package zlib was not found" 并在链接期报 cannot find -lz。
	echo "    探测 pkg-config 路径"
	for cand in "$SR/lib/pkgconfig" "/usr/lib/${SYS_TRIPLET}/pkgconfig" \
	           "/usr/share/pkgconfig"; do
		if [ -d "$cand" ]; then
			APT_PC="$APT_PC${APT_PC:+:}$cand"
			echo "      + $cand"
		fi
	done
	if [ -z "$APT_PC" ]; then
		echo "ERROR: 未找到任何 pkgconfig 目录。"
		echo "       请确认已安装 libdrm-dev:arm64 / zlib1g-dev:arm64"
		echo "       可用 'dpkg -L libdrm-dev:arm64 | grep pkgconfig' 定位。"
		exit 1
	fi
	export PKG_CONFIG_PATH="$APT_PC${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
	export PKG_CONFIG_SYSROOT_DIR="$SR"
	echo "    sysroot 就绪"
else
	SR="$BUILD_DIR/sysroot"
	echo "==> [1/6] 从 Debian 池构建 sysroot: $SR"

	fetch_deb() {
		# $1=url 路径  $2=目标文件名
		dst="$BUILD_DIR/$2"
		[ -f "$dst" ] && { echo "    已存在 $2"; return; }
		echo "    下载 $2"
		wget -q -O "$dst" "$DEB_BASE/$1"
	}

echo "==> [1/6] 准备 aarch64 glibc sysroot"
fetch_deb "g/glibc/libc6_${GLIBC_FULL}_arm64.deb"       libc6.deb
fetch_deb "g/glibc/libc6-dev_${GLIBC_FULL}_arm64.deb"   libc6-dev.deb
fetch_deb "l/linux/linux-libc-dev_6.1.176-1_arm64.deb"  linux-libc-dev.deb

for deb in libc6 libc6-dev linux-libc-dev; do
	echo "    解包 $deb"
	rm -rf "$BUILD_DIR/_x"; mkdir -p "$BUILD_DIR/_x"
	( cd "$BUILD_DIR/_x" \
		&& ar x "$BUILD_DIR/$deb.deb" \
		&& tar xf data.tar.* \
		&& rm -f data.tar.* control.tar.* debian-binary )
	cp -a "$BUILD_DIR/_x/." "$SR/"
done
rm -rf "$BUILD_DIR/_x"

# Alpine gcc 规格会链 -lssp_nonshared，Debian libc6-dev 不提供 → 补空档
echo "    补 libssp_nonshared stub"
STUB_LIB="$SR/usr/lib/${SYS_TRIPLET}/libssp_nonshared.a"
if [ ! -f "$STUB_LIB" ]; then
	echo 'int __oca_ssp_stub_unused;' > "$BUILD_DIR/_stub.c"
	( cd "$BUILD_DIR" && gcc -c -o _stub.o _stub.c && ar rcs "$STUB_LIB" _stub.o )
	rm -f "$BUILD_DIR/_stub.c" "$BUILD_DIR/_stub.o"
fi

# ---------------------------------------------------------------- #
# 2. libdrm for cross
# ---------------------------------------------------------------- #
echo "==> [2/6] 准备 cross libdrm"
# libdrm 不是 glibc 包，需从 Debian 取并装进 sysroot。
# 版本需 >= 2.4.109（Mesa _drm_ver 硬性下限，见 meson.build L774）
#
# 注意包名与版本串：Debian bullseye 里 libdrm2 / libdrm-dev / libdrm-etnaviv1
# 带 +b1 后缀，而 libdrm-common 是纯 _all 包、版本串**不带** +b1。
# （已实测：libdrm-common_2.4.114-1+b1_all.deb → 404，正确为 ..._2.4.114-1_all.deb）
LIBDRM_VER="${LIBDRM_VER:-2.4.114-1+b1}"
LIBDRM_COMMON_VER="${LIBDRM_COMMON_VER:-2.4.114-1}"
fetch_deb "libd/libdrm/libdrm2_${LIBDRM_VER}_arm64.deb"            libdrm2.deb
fetch_deb "libd/libdrm/libdrm-etnaviv1_${LIBDRM_VER}_arm64.deb"    libdrm-etnaviv1.deb
fetch_deb "libd/libdrm/libdrm-dev_${LIBDRM_VER}_arm64.deb"        libdrm-dev.deb
fetch_deb "libd/libdrm/libdrm-common_${LIBDRM_COMMON_VER}_all.deb" libdrm-common.deb

# expat：Mesa 里 dep_expat 的 required 绑定到 -Dexpat 选项（默认 auto）。
# teflon/etnaviv 不需要 XML 配置解析，故直接 disabled —— 这比从 Debian 取
# cross expat 更稳（bullseye 的 libexpat1 无配对 arm64 -dev 包）。
# 若改回 auto，meson 会退回 subproject，而 subproject 在交叉编译下会因
# arc4random_buf（glibc 2.36+ 才有）编译失败。

for deb in libdrm2 libdrm-etnaviv1 libdrm-dev libdrm-common; do
	rm -rf "$BUILD_DIR/_x"; mkdir -p "$BUILD_DIR/_x"
	( cd "$BUILD_DIR/_x" \
		&& ar x "$BUILD_DIR/$deb.deb" \
		&& tar xf data.tar.* \
		&& rm -f data.tar.* control.tar.* debian-binary )
	cp -a "$BUILD_DIR/_x/." "$SR/"
done
rm -rf "$BUILD_DIR/_x"

# 关键修正：glibc 的架构相关头（bits/ 等）在 usr/include/aarch64-linux-gnu/ 下，
# 是 multiarch 布局。meson 的 [built-in options] c_args **只作用于 C，
# 不作用于 C++**（cpp_args 需单独给），否则编 .cpp 时报
#   fatal error: bits/libc-header-start.h: No such file or directory
#
# 注意：不能无脑把 multiarch 下所有目录软链进主 include ——
# `asm/` 会遮蔽 libstdc++ 自带的 asm/ 目录，导致 C++ 编译报
#   '__gnuc_va_list' has not been declared
# 因此这里只软链确实缺失且无冲突的 bits/ 与 gnu/，
# 其余靠下面的 cpp_args 显式指定 multiarch 路径。
echo "    软链缺失的 multiarch 头（仅 bits/gnu，避免遮蔽 asm/）"
for d in bits gnu; do
	if [ -d "$SR/usr/include/${SYS_TRIPLET}/$d" ] \
	   && [ ! -e "$SR/usr/include/$d" ]; then
		ln -sfn "${SYS_TRIPLET}/$d" "$SR/usr/include/$d" || true
	fi
done
ZLIB_VER="${ZLIB_VER:-1.3.dfsg+really1.3.1-1+b1}"
fetch_deb "z/zlib/zlib1g_${ZLIB_VER}_arm64.deb"          zlib1g.deb
fetch_deb "z/zlib/zlib1g-dev_${ZLIB_VER}_arm64.deb"      zlib1g-dev.deb
for deb in zlib1g zlib1g-dev; do
	rm -rf "$BUILD_DIR/_x"; mkdir -p "$BUILD_DIR/_x"
	( cd "$BUILD_DIR/_x" \
		&& ar x "$BUILD_DIR/$deb.deb" \
		&& tar xf data.tar.* \
		&& rm -f data.tar.* control.tar.* debian-binary )
	cp -a "$BUILD_DIR/_x/." "$SR/"
done
rm -rf "$BUILD_DIR/_x"

# pkg-config .pc 里 libdir 写死 /usr/lib/aarch64-linux-gnu，
# 交叉链接时必须让 pkg-config 指向 sysroot，否则会去链接宿主 x86_64 的库。
# 注意：这里**必须 export 到整个脚本**（不只是本段），
# 否则后面的 meson setup 找不到 sysroot 里的 zlib/libdrm，
# 会退化为 "Package zlib was not found" 并在链接期报 cannot find -lz。
export PKG_CONFIG_SYSROOT_DIR="$SR"
export PKG_CONFIG_PATH="$SR/usr/lib/${SYS_TRIPLET}/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
# PKG_CONFIG_LIBDIR 会**替换**默认搜索路径，比 PKG_CONFIG_PATH 更严格。
# 交叉编译时必须用它：否则若 runner 上装了 libdrm-dev:arm64 /
# zlib1g-dev:arm64（Ubuntu 24.04 的 glibc 2.39），pkg-config 会优先命中
# /usr/lib/aarch64-linux-gnu/pkgconfig 里的 .pc，导致链接到 2.39 的库，
# 产物就会带上设备（glibc 2.36）不存在的符号。
export PKG_CONFIG_LIBDIR="$SR/usr/lib/${SYS_TRIPLET}/pkgconfig"
for pc in libdrm zlib; do
	f="$SR/usr/lib/${SYS_TRIPLET}/pkgconfig/$pc.pc"
	[ -f "$f" ] && sed -i "s#^prefix=/usr#prefix=$SR/usr#" "$f"
done
echo "    PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
fi   # 结束 SYSROOT_MODE=deb 分支

# ---------------------------------------------------------------- #
# 3. 取 Mesa 源码
# ---------------------------------------------------------------- #
echo "==> [3/6] 获取 Mesa $MESA_TAG"
cd "$BUILD_DIR"
if [ ! -d mesa/.git ]; then
	git clone --depth 1 --branch "$MESA_TAG" \
		https://gitlab.freedesktop.org/mesa/mesa.git mesa
else
	echo "    已有 mesa 源码，跳过 clone"
fi

# 确认 vendored TFLite 头在位（这是 teflon 的前提）
if [ ! -f mesa/include/tensorflow/lite/c/common.h ]; then
	echo "ERROR: Mesa 树内缺 include/tensorflow/lite/c/common.h"
	echo "       该 tag 未 vendored TFLite 头，teflon 无法编译。"
	echo "       请换一个 tag，或改用带 vendored 头的版本。"
	exit 1
fi
echo "    vendored TFLite 头就位 (来自 TF 2.13.0)"

# ---------------------------------------------------------------- #
# 3b. 打 NPU 识别补丁（och）
# ---------------------------------------------------------------- #
# 不打这个补丁，teflon 在真机上**必然段错误**：
#   NPU 类型（ETNA_CORE_NPU）唯一的赋值点在 etna_query_feature_db()，
#   而它只在 drm_version >= 1.4 时才被调用。本设备内核报 1.3.0，
#   但实际支持 PRODUCT_ID/CUSTOMER_ID/ECO_ID 这三个 param（0x1c/0x1d/0x1e，
#   与 Mesa enum etna_param_id 完全一致）—— Mesa 的版本号代理判断错误。
#   结果：hwdb 查找被跳过 → info->type 恒为 ETNA_CORE_GPU
#        → screen->npu 恒为 NULL → etna_ml_get_core_info() 空指针 → 段错误。
#
# 已用崩溃栈 + 反汇编把崩溃点定位到单条指令（etna_ml_subgraph_create+0x44
# 的 `ldr w0, [x0]`），完整推导见 docs/15-npu-detection-rootcause.md。
#
# 补丁脚本自带校验：匹配数不为 1 就**中止构建**，避免静默产出
# 一个没打上补丁的 .so（那会表现为"换版本也没用"，很难查）。
echo "==> [3b/6] 打 etnaviv NPU 识别补丁"
# SCRIPT_DIR 在脚本顶部（任何 cd 之前）已捕获，此处直接用
PATCHER="$SCRIPT_DIR/patch-etnaviv-npu-detect.py"
if [ ! -f "$PATCHER" ]; then
	echo "ERROR: 缺 $PATCHER"
	exit 1
fi
python3 "$PATCHER" mesa/src/etnaviv/drm/etnaviv_gpu.c || {
	echo "ERROR: NPU 识别补丁应用失败 —— 不要跳过，否则产物在真机上会段错误"
	exit 1
}

# ---------------------------------------------------------------- #
# 3c. 打「只接受卷积」运行时开关补丁（och）
# ---------------------------------------------------------------- #
# 实测背景：本 delegate 接管的算子全是 TRANSPOSE/RESHAPE/CONCAT/PAD（布局操作），
# 而 partition_invoke 每次调用、每个分区都 context_create/destroy，
# 单分区固定开销 6–31 ms → 18 个分区就超过 CPU 全部推理时间（302 ms）。
# 置 TEFLON_CONV_ONLY=1 可只接受 Conv2d/DWConv2d，让布局操作留 CPU。
#
# ⚠️ 与 3b 的致命性**故意不同**：3b 的补丁缺失会导致真机段错误（静默产出坏产物），
#    所以必须 fail loudly；而本补丁缺失只是**少一个实验开关**，
#    产物本身仍完全可用（默认行为不变）。
#    因此这里**非致命** —— 否则会给 25.0.7 等结构不同的版本带来假失败。
#    但要**醒目警告**，并且靠真机验证开关是否真的生效（缺补丁时开关无效会被发现）。
echo "==> [3c/6] 打「只接受卷积」运行时开关补丁"
PATCHER2="$SCRIPT_DIR/patch-teflon-conv-only.py"
if [ -f "$PATCHER2" ]; then
	if python3 "$PATCHER2" mesa/src/gallium/frontends/teflon/tfl_device.c; then
		echo "    TEFLON_CONV_ONLY 开关已就绪（默认关闭，行为不变）"
	else
		echo "    ！！警告：本 Mesa 版本结构不同，TEFLON_CONV_ONLY 开关【不可用】"
		echo "    ！！这不影响产物可用性（默认行为不变），但真机上设了开关也不会生效"
	fi
else
	echo "    ！！警告：缺 $PATCHER2，跳过（开关不可用）"
fi

# ---------------------------------------------------------------- #
# 3d. 【诊断专用】calculate_bias_correction 越界取证补丁
# ---------------------------------------------------------------- #
# 崩溃定位：PrepareDelegate 阶段 SIGSEGV，故障地址落在
#   etnaviv_ml_nn_v7.c 的 calculate_bias_correction() 里逐字节读 weights[i]，
#   越界条件 = 循环界(w*h*input_channels) > weight_tensor 实际分配字节数。
# 本补丁上机打印尺寸/指针，并在预测到越界时钳住循环（让进程不崩，把后续问题也暴露）。
#
# ⚠️ 默认关闭：不设 TEFLON_DIAG 时本步整段跳过，**构建流程与产物完全不变**。
# ⚠️ 设了 TEFLON_DIAG=1 的产物**只能用于诊断，不可交付**（会改变越界处行为）。
#
# 用法：TEFLON_DIAG=1 TEFLON_DEBUG_INFO=1 ./build-teflon.sh
# 诊断补丁与修复补丁针对**同一个函数**，同时开没有意义
#（诊断版会钳住循环、修复版会纠正长度，语义上互斥）。
#
# 说明：实测两者**文本上其实能共存**（fix 之后 diag 的锚点
# `int32_t correction = 0;` 依然存在），所以下面这道拦截是**防混淆的安全栏**，
# 不是「技术上必然失败」—— 注释别写成后者，那会误导下一个人。
if [ "${TEFLON_DIAG:-0}" = "1" ] && [ "${TEFLON_FIX_V7:-0}" = "1" ]; then
	echo "ERROR: TEFLON_DIAG 与 TEFLON_FIX_V7 语义互斥，只能二选一"
	exit 1
fi

if [ "${TEFLON_DIAG:-0}" = "1" ]; then
	echo "==> [3d/6] 打【诊断】越界取证补丁（该产物不可交付）"
	PATCHER3="$SCRIPT_DIR/patch-etnaviv-diag-bias.py"
	ML7="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"
	if [ ! -f "$PATCHER3" ]; then
		echo "ERROR: 设了 TEFLON_DIAG=1 却缺 $PATCHER3"
		exit 1
	fi
	if [ ! -f "$ML7" ]; then
		echo "ERROR: 找不到 $ML7 —— Mesa 版本结构不同，诊断补丁无法定位目标"
		exit 1
	fi
	python3 "$PATCHER3" "$ML7" || {
		echo "ERROR: 诊断补丁应用失败 —— 诊断构建无意义，中止"
		exit 1
	}
else
	echo "==> [3d/6] 诊断补丁：默认关闭（设 TEFLON_DIAG=1 启用）"
fi

# ---------------------------------------------------------------- #
# 3e. 【修复】v7 calculate_bias_correction 漏 depthwise 特判
# ---------------------------------------------------------------- #
# 上游 Mesa 缺陷（已静态证明，见 verify/reports/05 与 06）：
#   etnaviv_ml_nn_v7.c 的 calculate_bias_correction() 用 operation->input_channels
#   决定扫描长度，**没有 depthwise 特判**；而同仓库的 calc_weight_quant /
#   requantize_weights / v8 版同名函数**都有**。
#   深度卷积权重形状是 [1,kh,kw,out_c]，每输出通道只有 W*H*1 个权重，
#   于是 v7 按 W*H*in_c 扫描并以此步长 → 第 2 个输出通道起越界 → SIGSEGV，
#   撞到哪页取决于堆布局（表现为非确定性）。
# 本补丁把 v8 的特判逻辑 backport 到 v7，并打印 buggy_bound/fixed_bound 取证。
#
# ⚠️ 默认关闭：不设 TEFLON_FIX_V7 时本步整段跳过，构建流程与产物完全不变。
#
# 用法：TEFLON_FIX_V7=1 TEFLON_DEBUG_INFO=1 ./build-teflon.sh
if [ "${TEFLON_FIX_V7:-0}" = "1" ]; then
	echo "==> [3e/6] 打【修复】v7 calculate_bias_correction depthwise 特判"
	PATCHER4="$SCRIPT_DIR/patch-etnaviv-v7-biasfix.py"
	ML7B="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"
	if [ ! -f "$PATCHER4" ]; then
		echo "ERROR: 设了 TEFLON_FIX_V7=1 却缺 $PATCHER4"
		exit 1
	fi
	if [ ! -f "$ML7B" ]; then
		echo "ERROR: 找不到 $ML7B"
		exit 1
	fi
	python3 "$PATCHER4" "$ML7B" || {
		echo "ERROR: 修复补丁应用失败 —— 不要跳过（上游若已修请直接升级 Mesa）"
		exit 1
	}
else
	echo "==> [3e/6] v7 修复补丁：默认关闭（设 TEFLON_FIX_V7=1 启用）"
fi

# ---------------------------------------------------------------- #
# 4. meson 交叉配置
# ---------------------------------------------------------------- #
echo "==> [4/6] 生成 meson cross file"
# sysroot 内的目录布局两种模式不同：
#   apt 模式：/usr/aarch64-linux-gnu/{include,lib}（无 multiarch 拆分）
#   deb 模式：<SR>/usr/include + multiarch 子目录 + libdrm 头子目录
# 注意：apt 模式下 SR_INC_MULTI / SR_INC_LIBDRM 为空串，
# 不能直接插进 meson 数组（会留下空行导致 "Malformed machine file"），
# 因此用 ${VAR:+...} 这种「非空才展开」的写法。
if [ "$SR_MODE" = "apt" ]; then
	SR_INC="$SR/include"
	SR_LIB="$SR/lib"
	SR_INC_MULTI=""
	# foreign-arch 包（libdrm-dev:arm64 / zlib1g-dev:arm64）的头在
	# /usr/include/<triplet>/ 下，不在 cross sysroot 里，必须单独加，
	# 否则编译报 'xf86drm.h: No such file or directory' / 'zlib.h not found'。
	SR_INC_LIBDRM=""
	if [ -d "/usr/include/${SYS_TRIPLET}/libdrm" ]; then
		SR_INC_LIBDRM="/usr/include/${SYS_TRIPLET}/libdrm"
	fi
	if [ -d "/usr/include/${SYS_TRIPLET}" ]; then
		SR_INC_MULTI="/usr/include/${SYS_TRIPLET}"
	fi
else
	SR_INC="$SR/usr/include"
	SR_LIB="$SR/usr/lib/${SYS_TRIPLET}"
	SR_INC_MULTI="$SR/usr/include/${SYS_TRIPLET}"
	SR_INC_LIBDRM="$SR/usr/include/libdrm"
fi

# 编译器前缀：
#   CI       CROSS_PREFIX=aarch64-linux-gnu- → 用真正的交叉编译器（默认 glibc）
#   musl 宿主 本机本身就是 aarch64，交叉前缀置空 → 直接用 gcc（需 SYSROOT_MODE=deb）
# 注意：置空前缀时 gcc 会把 '-gcc' 拼成 '-gcc'，故此处显式处理空值。
if [ -n "${CROSS_PREFIX}" ]; then
	CC_BIN="${CC_BIN:-${CROSS_PREFIX}gcc}"
	CXX_BIN="${CXX_BIN:-${CROSS_PREFIX}g++}"
	AR_BIN="${AR_BIN:-${CROSS_PREFIX}ar}"
	STRIP_BIN="${STRIP_BIN:-${CROSS_PREFIX}strip}"
else
	CC_BIN="${CC_BIN:-gcc}"
	CXX_BIN="${CXX_BIN:-g++}"
	AR_BIN="${AR_BIN:-ar}"
	STRIP_BIN="${STRIP_BIN:-strip}"
fi
command -v "$CC_BIN" >/dev/null 2>&1 || {
	echo "ERROR: 找不到编译器 $CC_BIN"
	echo "  CI:   需安装 gcc-aarch64-linux-gnu g++-aarch64-linux-gnu"
	echo "  本机: 若宿主本身是 aarch64，用 NATIVE=1 SYSROOT_MODE=deb 运行"
	exit 1
}

# musl 宿主的额外链接修正：
# Alpine gcc 的 specs 隐式 -lc 指向 musl，导致 glibc 的 fortified 符号
# （__fprintf_chk / __snprintf_chk / __printf_chk …）解析不到，
# 链接期报大量 undefined reference。
# 解法：显式把 glibc 的 libc.so.6 / libm.so.6 追加到链接命令尾部。
# CI 用真正的 aarch64-linux-gnu 交叉编译器（默认 glibc），不需要这段。
# ---------------------------------------------------------------- #
# libdrm 处理（方案 A：随包部署 libdrm.so.2）
#
# 宿主 2026-10-15 实测确认：
#   DSM 7.4.1 (DS423) 系统级**完全没有 libdrm**（find / -name 'libdrm*' 全空），
#   所以 libteflon.so 单独拷过去必然 dlopen 失败。
#
# 处置：构建时把 aarch64 的 libdrm.so.2 一并产出，
# 与 libteflon.so 放同一目录，并给 libteflon.so 加 RUNPATH=$ORIGIN，
# 让 dlopen 能自动找到同目录的 libdrm.so.2，无需设置 LD_LIBRARY_PATH。
#
# 只用 libdrm.so.2 一个文件即可 —— 实测 libteflon.so 的 14 个未定义 drm 符号
# 都是核心 API，**不包含 drmGetEntry**，因此不需要 libdrm_etnaviv.so.1。
#
# STATIC_LIBDRM=1 可选静态链入 libdrm.a，但需额外处理 musl 宿主的
# stat64/fstat64 问题（见 5b），默认关闭。
# ---------------------------------------------------------------- #
STATIC_LIBDRM="${STATIC_LIBDRM:-0}"

EXTRA_LINK=""
# NEED_TAIL_LINK=1 表示"普通编译可能失败，需在末尾重链"，
# 只在 musl 宿主 + 静态 libdrm 同时成立时需要（见 5b 步骤）。
NEED_TAIL_LINK=0
# musl 宿主：Alpine gcc specs 隐式 -lc 指向 musl，导致 glibc 的 fortified 符号
# （__fprintf_chk 等）解析不到，必须显式补 glibc 的 libc/libm。
if [ "$NATIVE" = "1" ] && [ "$SYSROOT_MODE" = "deb" ]; then
	if command -v ldd >/dev/null 2>&1 && ldd --version 2>&1 | grep -qi musl; then
		GLIBC_LIB_DIR="$SR/lib/aarch64-linux-gnu"
		if [ -f "$GLIBC_LIB_DIR/libc.so.6" ]; then
			EXTRA_LINK="'$GLIBC_LIB_DIR/libc.so.6', '$GLIBC_LIB_DIR/libm.so.6'"
			NEED_TAIL_LINK=1
			echo "    检测到 musl 宿主：追加 glibc libc/libm 到链接命令"
		fi
	fi
fi

if [ "$STATIC_LIBDRM" = "1" ]; then
	LIBDRM_A="$SR_LIB/libdrm.a"
	if [ -f "$LIBDRM_A" ]; then
		# 关键：不能只把 libdrm.a 加到链接命令——
		# meson 会通过 pkg-config 再加一次 -ldrm，动态库会被重新引入。
		# 正确做法：把 sysroot 里的 libdrm.so 换成指向 libdrm.a 的 ld 链接脚本，
		# 这样无论以 -ldrm 还是全路径链接，都会解析到静态库。
		LIBDRM_SO="$SR_LIB/libdrm.so"
		if [ -e "$LIBDRM_SO" ] || [ -L "$LIBDRM_SO" ]; then
			cp -f "$LIBDRM_SO" "$BUILD_DIR/libdrm.so.bak" 2>/dev/null || true
		fi
		cat > "$LIBDRM_SO" <<LDscript
/* 由 build-teflon.sh 生成：强制 -ldrm 解析到静态库，
   避免 libteflon.so 运行时依赖设备上的 libdrm.so.2。
   注意两点：
   1) GROUP 内必须用**相对** libdrm.a。链接时 ld 会给 .a 加 --sysroot 前缀，
      写绝对路径会变成 <sysroot>/<绝对路径> 而找不到文件。
   2) 不要加 AS_NEEDED(libdrm.so.2)，那会把动态依赖重新加回来。 */
OUTPUT_FORMAT("elf64-littleaarch64")
GROUP ( libdrm.a )
LDscript
		echo "    静态链接 libdrm: libdrm.so → libdrm.a（覆盖 pkg-config 的 -ldrm）"
	else
		echo "    警告: 未找到 $LIBDRM_A，退回动态链接 libdrm"
		echo "         （DSM 上若无 libdrm.so.2，dlopen 会失败）"
	fi
fi

# 转成 meson machine file 里逗号分隔的字符串
EXTRA_LINK="${EXTRA_LINK:+, $EXTRA_LINK}"

# RUNPATH 说明（DSM 7.4.1 实测完全没有 libdrm，libteflon.so 与 libdrm.so.2
# 必须同目录部署）：
#   下面 c_link_args / cpp_link_args 里给的是
#     -Wl,--enable-new-dtags,-rpath,\$ORIGIN
#   $ORIGIN 会被动态链接器解析为"该 .so 自身所在目录"，
#   因此 dlopen 时能找到同目录的 libdrm.so.2，无需设LD_LIBRARY_PATH。
#   用 --enable-new-dtags 生成 RUNPATH 而非 RPATH，语义更干净
#   （RUNPATH 不传递给下游库）。
#   反斜杠转义是为了让 heredoc 把 $ORIGIN 原样写进 cross.ini，
#   再由 meson/ld 传到最终 .so 的动态段（交叉链接时 ld 不展开它）。
#
# 注意：heredoc 内不能出现未转义的 $ORIGIN，否则 set -u 会直接报错。

# ⚠️ 下面的 shell 控制流**必须在 heredoc 之外**。
#    （踩过的坑：曾把 DBG_ARGS 的 if/else/fi 写在 `cat > cross.ini <<EOF … EOF` 内部，
#      结果 shell 的 else/fi 被当成 cross.ini 的内容写进文件，
#      meson 报 "Malformed machine file: [line 28]: 'else'"，
#      而 ninja 那步的失败又被管道/|| 掩盖，整体看起来还是 success ——
#      会拿着一个 cross.ini 损坏的产物上机，得到零信息。）
# TEFLON_DEBUG_INFO=1 时追加 -g3（带调试信息，供 addr2line 定位源码行）。
# 默认不带，保持干净产物体积；修复/诊断构建建议与 TEFLON_FIX_V7=1 一起用。
DBG_ARGS=""
if [ "${TEFLON_DEBUG_INFO:-0}" = "1" ]; then
	DBG_ARGS="-g3"
	echo "    调试信息：开启（-g3）"
else
	echo "    调试信息：关闭（干净产物）"
fi

cat > "$BUILD_DIR/cross.ini" <<EOF
[binaries]
c         = '${CC_BIN}'
cpp       = '${CXX_BIN}'
ar        = '${AR_BIN}'
strip     = '${STRIP_BIN}'
pkg-config = 'pkg-config'

[host_machine]
system     = 'linux'
cpu_family = 'aarch64'
cpu        = 'aarch64'
endian     = 'little'

[properties]
sys_root         = '$SR'
needs_exe_wrapper = true

[built-in options]
# 数组必须写成**单行**：meson 1.11 的 machine file 解析器不接受
# 跨行数组（会在 ']' 处报 Malformed machine file）。
# ${VAR:+...} 保证空值不产生多余空格/空元素（apt 模式无 multiarch 路径）。
# DBG_ARGS 的赋值与 echo 已挪到 heredoc 之外（见上方注释里的踩坑记录）。
c_args = ['--sysroot=$SR', ${DBG_ARGS:+'$DBG_ARGS',} ${SR_INC_MULTI:+'-isystem', '$SR_INC_MULTI',} ${SR_INC_LIBDRM:+'-isystem', '$SR_INC_LIBDRM',} '-isystem', '$SR_INC', '-B', '$SR_LIB/', '-L', '$SR_LIB']
cpp_args = ['--sysroot=$SR', ${DBG_ARGS:+'$DBG_ARGS',} ${SR_INC_MULTI:+'-isystem', '$SR_INC_MULTI',} ${SR_INC_LIBDRM:+'-isystem', '$SR_INC_LIBDRM',} '-isystem', '$SR_INC', '-B', '$SR_LIB/', '-L', '$SR_LIB']
c_link_args = ['--sysroot=$SR', '-Wl,-rpath-link,$SR_LIB', '-Wl,--enable-new-dtags,-rpath,\$ORIGIN'$EXTRA_LINK]
cpp_link_args = ['--sysroot=$SR', '-Wl,-rpath-link,$SR_LIB', '-Wl,--enable-new-dtags,-rpath,\$ORIGIN'$EXTRA_LINK]
EOF

echo "    $BUILD_DIR/cross.ini"

# ---------------------------------------------------------------- #
# 5. meson configure + build
# ---------------------------------------------------------------- #
echo "==> [5/6] meson setup（只构建 teflon + etnaviv）"
cd "$BUILD_DIR/mesa"
rm -rf _build

# 选项说明（**不要硬编码** —— 不同 Mesa 版本选项集不同，下面会自动探测）：
#   -Dteflon=true            启用 TFLite delegate（默认 false，必开）
#   -Dgallium-drivers=etnaviv 只建 etnaviv（choices 里确有此项）
#   -Dllvm=disabled          避免拉 LLVM 依赖，加速且减小体积
#   -Dplatforms=              留空=不建任何窗口平台（该选项不接受 'none'）
#   -Dbuild-tests=false      不建测试
#   -Dshared-glapi=disabled  交叉编译下 glapi 不单独共享
#
# 实测差异：mesa-24.3.0 有 92 个选项、mesa-26.2.5 有 113 个，
# 且 26.2.5 **删除了 osmesa**，传 -Dosmesa=false 会报
#   ERROR: Unknown option: "osmesa".
# 反之 vulkan / shared-egl / gl-dri / dri-tests / install-test-dri-deps
# 在 24.3.0 中不存在，传了同样会报错。
# → 所以下面统一走「探测该版本真实存在的选项，只传存在的」。
#
# -Dexpat=disabled：避免 meson 退回 expat subproject（交叉编译下会因
#   arc4random_buf（glibc 2.36+ 才有）编译失败）。
# -Dvulkan-drivers= / -Dvulkan-layers=：默认 auto 会带入 intel/amd，
#   进而要求宿主装 glslangValidator，纯属无谓。
# -Dplatforms= 必须留空而不是 "none"（choices 里没有 none）。
# ⚠️ 选项集随 Mesa 版本变化，**不能硬编码**。
# 实测：mesa-26.2.5 已删除 osmesa 选项，meson 直接报
#   meson.build:4:0: ERROR: Unknown option: "osmesa".
# 而 mesa-24.3.0 仍有该选项，删掉它又会走别的路。
#
# 所以做法：**探测该版本真实存在的选项**，只传存在的。
# meson 1.1+ 把 meson_options.txt 改名为 meson.options，两个名字都试。
OPT_FILE=""
for f in meson.options meson_options.txt; do
	[ -f "$f" ] && { OPT_FILE="$f"; break; }
done
if [ -z "$OPT_FILE" ]; then
	echo "ERROR: 找不到 meson.options / meson_options.txt，无法探测选项集"
	exit 1
fi
echo "    选项定义文件: $OPT_FILE"

# 提取全部选项名。**必须兼容两种写法**，否则会误判：
#   meson 1.1+ 的 meson.options：option( 换行后才是 'name',   ← 26.2.5
#   旧 meson_options.txt：      option('name', type=...) 同一行  ← 24.3.0
# 注意不能用 grep -E "option\(\s*'name'"：grep 逐行处理，\s* 不跨行，
# 对 26.2.5 会把**所有**选项都误判为不存在（已实测踩过）。
extract_opts() {
	awk '
/^[[:space:]]*option\([[:space:]]*$/ { want=1; next }
/^[[:space:]]*option\(/              { want=0
	    if (match($0, /'"'"'[^'"'"']+'"'"'/)) print substr($0, RSTART+1, RLENGTH-2); next }
	want && match($0, /'"'"'[^'"'"']+'"'"'/) { print substr($0, RSTART+1, RLENGTH-2); want=0 }
	' "$OPT_FILE" | sort -u
}
OPT_NAMES="$(extract_opts)"
echo "    该版本共 $(printf '%s\n' "$OPT_NAMES" | grep -c .) 个选项"

# 期望传的选项及默认值（只是「想要」的值，不保证该版本认识）
WANT_OPTS="teflon=true gallium-drivers=etnaviv llvm=disabled platforms= \
build-tests=false shared-glapi=disabled glx=disabled gles2=disabled \
opengl=false osmesa=false gbm=disabled libunwind=disabled egl=disabled \
expat=disabled vulkan-drivers= vulkan-layers="

MESON_D=""
DROPPED=""
for kv in $WANT_OPTS; do
	name="${kv%%=*}"
	if printf '%s\n' "$OPT_NAMES" | grep -qx "$name"; then
		MESON_D="$MESON_D -D$kv"
	else
		DROPPED="$DROPPED $name"
	fi
done
echo "    传入选项:$MESON_D"
if [ -n "$DROPPED" ]; then
	echo "    本版本不存在的选项（已剔除）:$DROPPED"
fi

# shellcheck disable=SC2086
meson setup _build \
	--cross-file "$BUILD_DIR/cross.ini" \
	$MESON_D

echo "==> 编译中（只编 libteflon.so，避免全量构建）"
# 只构建 teflon 目标：依赖会自动带上 etnaviv/gallium 所需部分，
# 但不会去编 vulkan/radv 等无关驱动。
#
# 注意：musl 宿主 + 静态 libdrm 时，链接步骤可能因 stat64/fstat64 失败
# （musl 无这些符号），此时由 5b 的尾部重链兜底，
# 因此这里不能用 set -e 直接中断。
if ! ninja -C _build src/gallium/targets/teflon/libteflon.so; then
	if [ "$NEED_TAIL_LINK" = "1" ]; then
		echo "    首次链接失败（预期内），转入尾部重链流程"
	else
		echo "ERROR: 编译失败"
		exit 1
	fi
fi

# ---------------------------------------------------------------- #
# 5b. musl 宿主的尾部重链
#
# 背景：静态链入 libdrm.a 后，它会引用 stat64/fstat64 —— 这些符号
# **只存在于 glibc**，musl 没有。而 Alpine gcc 的 specs 隐式 -lc 指向 musl，
# 因此链接期报：
#   libdrm.a(xf86drm.c.o): undefined reference to `fstat64' / `stat64'
#
# c_link_args 里的 glibc libc 救不了：meson 把它放在 --end-group 之前，
# 那时 libdrm.a 的目标文件还没被拉入，符号无从解析。
#
# 解法：把 glibc 的 libc.so.6/libm.so.6 追加到链接命令**最末尾**，
# 让 ld 在已拉入 libdrm.a 目标文件后再解析它们。
# CI 用真正的交叉工具链（默认 glibc），不会走到这一步。
# ---------------------------------------------------------------- #
if [ "$NEED_TAIL_LINK" = "1" ]; then
	SO_PATH="$BUILD_DIR/mesa/_build/src/gallium/targets/teflon/libteflon.so"
	echo "==> 尾部重链（musl 宿主：补 glibc libc 以解析 stat64/fstat64）"
	rm -f "$SO_PATH"
	LINK_CMD=$(ninja -C _build -t commands src/gallium/targets/teflon/libteflon.so 2>/dev/null | tail -1)
	# 用 sed 在命令末尾（--end-group 之后）追加 glibc libc/libm
	LINK_CMD_TAIL=$(printf '%s' "$LINK_CMD" \
		| sed "s#-Wl,--end-group#-Wl,--end-group -Wl,--start-group '$GLIBC_LIB_DIR/libc.so.6' '$GLIBC_LIB_DIR/libm.so.6' -Wl,--end-group#")
	( cd "$BUILD_DIR/mesa/_build" && eval "$LINK_CMD_TAIL" ) 2>&1 \
		| grep -viE "dangerous|best use|linker input file unused" | tail -3 || true
	[ -f "$SO_PATH" ] || { echo "ERROR: 尾部重链失败"; exit 1; }
	echo "    尾部重链完成"
fi

# ---------------------------------------------------------------- #
# 6. 收集产物
# ---------------------------------------------------------------- #
echo "==> [6/6] 收集产物"
SO=$(find "$BUILD_DIR/mesa/_build" -name 'libteflon.so' -type f | head -1)
if [ -z "$SO" ]; then
	echo "ERROR: 未找到 libteflon.so"
	echo "可尝试检查 meson 日志中的 teflon 相关错误："
	echo "  grep -i teflon $BUILD_DIR/mesa/_build/meson-logs/meson-log.txt"
	exit 1
fi

mkdir -p "$BUILD_DIR/out"
cp "$SO" "$BUILD_DIR/out/libteflon.so"
chmod 755 "$BUILD_DIR/out/libteflon.so"

cd "$BUILD_DIR/out"

# 随包部署 libdrm.so.2（方案 A）
# DSM 7.4.1 实测无任何 libdrm，libteflon.so 与它必须同目录。
# 取 sysroot 里那份 aarch64 的 libdrm.so.2 原样带出，保持 SONAME 一致。
LIBDRM_SO2=""
if [ "$STATIC_LIBDRM" != "1" ]; then
	# apt 模式下 libdrm.so.2 可能落在两处，都要找：
	#   foreign-arch 包 → /usr/lib/<triplet>/
	#   cross sysroot   → $SR/lib/
	LIBDRM_SO2=$(find "$SR" /usr/lib/"${SYS_TRIPLET}" \
		-name 'libdrm.so.2*' -type f 2>/dev/null | head -1)
	if [ -n "$LIBDRM_SO2" ]; then
		cp "$LIBDRM_SO2" ./libdrm.so.2
		chmod 755 ./libdrm.so.2
		echo "    随包: libdrm.so.2 (来自 $(basename "$LIBDRM_SO2"))"
	else
		echo "    警告: sysroot 里没找到 libdrm.so.2 —— 产物在设备上会 dlopen 失败"
	fi
fi

# 宿主若是 musl（Alpine），gcc specs 会在 .so 里留下一条
# DT_NEEDED: libc.musl-aarch64.so.1。DSM 是纯 glibc，设备上没有该库，
# dlopen 会直接失败 → 必须剔除。
#
# 注意：CI 用的是真正的 aarch64-linux-gnu 交叉工具链（默认 glibc），
# 不会出现这条依赖，此段是 musl 宿主下的权宜处理，删掉也不影响 CI。
if readelf -d libteflon.so 2>/dev/null | grep -q "libc\.musl"; then
	echo "    检测到 musl 依赖，剔除（宿主为 musl 环境）"
	if command -v patchelf >/dev/null 2>&1; then
		patchelf --remove-needed libc.musl-aarch64.so.1 libteflon.so || true
	else
		echo "    警告：缺 patchelf，无法剔除 musl 依赖；请安装 patchelf"
	fi
fi

# 仅对产物做 strip，保留调试可读性
#
# TEFLON_NO_STRIP=1 时**跳过**，保留 DWARF 调试段。
# 用途：`addr2line -e libteflon.so <地址>` 一行给出源码行，省掉
# 「指令指纹 + 结构体偏移反查 + 同仓库函数对照」这套容易出错的间接推理。
# ⚠️ 默认关闭（干净产物）；带调试段的产物明显变大且**不可交付**。
#
# 历史：第一轮我判断「先不加，等真机验证不过再说」——那一轮判断是对的；
#真机证明首个修复无效后，它就从「可选」变成了硬需求（2026-10-17）。
if [ "${TEFLON_NO_STRIP:-0}" = "1" ]; then
	echo "    strip：跳过（保留调试段，供 addr2line 使用；该产物不可交付）"
elif command -v "$STRIP_BIN" >/dev/null 2>&1; then
	"$STRIP_BIN" --strip-unneeded libteflon.so || true
else
	echo "    警告：找不到 strip 可执行文件，跳过 strip"
fi

# 自检：产物必须是 aarch64 + 纯 glibc 依赖
echo "==> 产物自检"
MACH=$(readelf -h libteflon.so | awk -F: '/Machine/{print $2}' | tr -d ' ')
case "$MACH" in
	*AArch64*|*aarch64*) echo "    架构 OK: $MACH" ;;
	*) echo "ERROR: 架构不对: $MACH"; exit 1 ;;
esac
if readelf -d libteflon.so | grep -q "libc\.musl"; then
	echo "ERROR: 仍残留 musl 依赖，产物无法在 DSM 上加载"; exit 1
fi
readelf -sW libteflon.so | grep -q tflite_plugin_create_delegate \
	&& echo "    delegate ABI OK" \
	|| { echo "ERROR: 缺少 tflite_plugin_create_delegate 导出"; exit 1; }

# libdrm 自检：DSM 无 libdrm，缺了就等于交付了一个打不开的包
if readelf -d libteflon.so | grep -q "libdrm\.so"; then
	if [ -f libdrm.so.2 ]; then
		DM=$(readelf -h libdrm.so.2 2>/dev/null | awk -F: '/Machine/{print $2}' | tr -d ' ')
		case "$DM" in
			*AArch64*|*aarch64*) echo "    随包 libdrm.so.2 OK ($DM)" ;;
			*) echo "ERROR: 随包 libdrm.so.2 架构不对: $DM"; exit 1 ;;
		esac
	else
		echo "ERROR: 产物依赖 libdrm.so.2 但未随包产出"
		echo "       DSM 7.4.1 无 libdrm，这会导致 dlopen 失败。"
		exit 1
	fi
fi
# RUNPATH 自检
if readelf -d libteflon.so | grep -qE 'RUNPATH|RPATH'; then
	echo "    RUNPATH: $(readelf -d libteflon.so | awk -F'[][]' '/RUNPATH|RPATH/{print $2}')"
fi

# 校验和覆盖全部交付文件
sha256sum libteflon.so > SHA256SUMS
[ -f libdrm.so.2 ] && sha256sum libdrm.so.2 >> SHA256SUMS
cat SHA256SUMS

echo
echo "=============================================="
echo " 产物: $BUILD_DIR/out/libteflon.so"
echo " 校验: $(cat SHA256SUMS)"
echo " 信息:"
"${CROSS_PREFIX}readelf" -h libteflon.so 2>/dev/null | grep -E 'Class|Machine|Type' || \
	readelf -h libteflon.so | grep -E 'Class|Machine|Type'
echo " 依赖:"
readelf -d libteflon.so 2>/dev/null | grep NEEDED || true
echo " 导出(应含 tflite_plugin_create_delegate):"
readelf -sW libteflon.so 2>/dev/null | grep -E 'tflite_plugin_(create|destroy)_delegate' || \
	readelf -sW libteflon.so | grep -E 'tflite_plugin_(create|destroy)_delegate' || true
echo " glibc 符号版本:"
readelf -V libteflon.so 2>/dev/null | grep -oE 'GLIBC_2\.[0-9]+' | sort -u | tr '\n' ' '
echo
echo "=============================================="