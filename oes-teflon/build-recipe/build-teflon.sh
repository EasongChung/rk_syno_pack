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
#   TEFLON_WCDIAG=1      在 write_core_interleaved() 写出权重前打印循环变量
#                         （core/kernel/z/superblock/superblocks/kernels_* /
#                          cores_used/out_channel/output_channels/w/h/wt_size），
#                         out_channel >= output_channels 时追加 "*** OOB ***"。
#                         锚点由 tools/gen-wcdiag-patch.py 从真实源码逐字生成。
#                         ⚠️ 不改行为：崩溃行仍会照旧触发，但那一行已进日志
#
#   TEFLON_OCFIX=1       在 write_core_interleaved() 的 out_channel 后加**尾部越界钳位**；
#                         **公式逐字不动**（曾做过「重映射」版，已作废，见下）
#                         根因已由真机 [och-wc] 日志证实：136 组合 0 反例地复现原式，
#                         且出现 out_channel >= output_channels（实测 123 vs 122）→ SIGSEGV。
#                         真正的缺陷是 kernels_per_core = DIV_ROUND_UP(output_channels,
#                         cores_used) **均分**（8*16=128 > 122 产生幽灵槽）；上游 v8 已在
#                         etnaviv_ml_nn_v8.c:729-731 用「余数分给前几个 core」修掉。
#                         原式结构由 v8 :726 / :754 证明**有意**（末尾的 * cores_used
#                         是 superblock 步长；逐 core 推进的是 kernels_in_superblock）。
#                         锚点由 tools/gen-ocfix-patch.py 从真实源码逐字生成，
#                         自带精确计数门 + 9 个反例测试 + 编译期标记
#                         och-patch-v7-outchannel-clamp（供 strings 判定）。
#                         ⚠️ 仍需四级判据第 ④ 级（逐字节比对）才算「能用」——
#                            不崩只说明越界消失，不代表系数排布正确。
#
#   TEFLON_ALIASDIAG=1    【诊断】alias（批量元数据迁移）取证。
#                         起因：ROFIX 把野指针换成 assert
#                           `pipe_buffer_map_range: offset + length <= buffer->width0'
#                           ⇒ tensor 元数据与 resource 实际大小不一致。
#                         可疑：reference_tensor_with_offset(:198) 同时改 size/offset，
#                           recreate_tensor(:226) **只改 size** —— 不对称是条线索；
#                           两者都把「所有共享 old_res 的 tensor」改成同一组值。
#                         调用点 CONCAT/SPLIT/Add-Subtract/BYPASS。
#                         ★ 与 TEFLON_ROFIX=1 **可同时开**（改的是不同函数）。
#                         目标文件 etnaviv_ml.c（3k 步，在 3j 之后）。
#                         ⚠️ 不可交付；上机须设 ETNA_MESA_DEBUG=ml_msgs。
#                         锚点由 tools/gen-aliasdiag-patch.py 从真实源码逐字生成。
#   TEFLON_CREATEDIAG=1   【诊断】create_tensor 分配取证。
#                         起因：ALIASDIAG 把范围收窄到「buffer 分配大小」——
#                           tensor->size=610000 正确，但 pipe_buffer_size(res)=5000，
#                           而**模型里没有任何 5000 字节的张量** ⇒ 5000 是驱动算出来的；
#                           5000 = 2×50×50，610000 = 244×50×50（差 122 倍）。
#                         嫌疑：create_tensor 的早退 `if (res != NULL) return;`
#                           （只看向不看 size；release 关 assert 后会静默早退）。
#                         每条出口都打 pipe_buffer_size(res) = 实际分配大小。
#                         ★ 与 ALIASDIAG / ROFIX 可同时开（注入函数互不相同）。
#                         目标文件 etnaviv_ml.c（3l 步，在 3k 之后）。
#                         ⚠️ 不可交付；上机须设 ETNA_MESA_DEBUG=ml_msgs。
#                         锚点由 tools/gen-creatediag-patch.py 从真实源码逐字生成。
#   TEFLON_ROFIX=1        【修复】read_outputs 映射范围与读取范围对齐。
#                         根因（真机 RODIAG 取证 0078f5b2，三条假说一次性判定）：
#                         H1 idx 越界**排除**；H2 pipe_buffer_size(res)!=tensor->size
#                         **确认**（致命组 bsize=5000 vs tsize=610000，
#                         越界读 605000 字节）；H3 跨 context 资源失效**排除**。
#                         改法：map 参数与 invoke(:908) 同构 (tensor->offset, tensor->size)
#                         + idx 边界检查。不加 min() 截断（会把越界换成静默错误数值）。
#                         目标文件 etnaviv_ml.c（3j 步，在 3i 之后）。
#                         ✅ 可与 TEFLON_RODIAG=1 叠加（自动先 ROFIX 后取证）。
#                         ⚠️ 只针对崩溃；tensor->size 被批量改写的更深问题未处理。
#                         锚点由 tools/gen-rofix-patch.py 从真实源码逐字生成。
#   TEFLON_RODIAG=1       【诊断】read_outputs 越界取证：打印 idx / n_tensors / 槽 / res /
#                         tsize / toff / bsize / src_map，用于区分 H1(idx 越界) /
#                         H2(bsize != tsize) / H3(resource 跨 context 失效)。
#                         目标文件 etnaviv_ml.c（3i 步，在 3h 之后）。
#                         ⚠️ 该产物不可交付；上机必须同时设 ETNA_MESA_DEBUG=ml_msgs，
#                            否则 ML_DBG 不打印（DBG_ENABLED 是运行期判断）。
#                         锚点由 tools/gen-rodiag-patch.py 从真实源码逐字生成。
#   TEFLON_FSGUARD=1      不接管「filter 深度 ≠ 输入通道数」的卷积，回退 CPU。
#                         根因：真机 +0x22fda0 SIGSEGV → addr2line → subsample:424
#                         → 崩溃算子 op#58 filter=[244,3,3,1] 但 input 有 244 通道
#                         （op#153 同型，filter=[488,3,3,1] / input 488 通道）
#                         → map_in 步长按 244 算而张量只有 2196 字节 → out_channel>=1 越界。
#                         改法是**不接管**而非改系数排布：整条 coeff 路径都建立在
#                         「filter 深度 == 输入通道」这个不成立的前提上，改单点只会把
#                         越界换成静默错误数值。
#                         目标文件 etnaviv_ml.c。
#                         锚点由 tools/gen-fsguard-patch.py 从真实源码逐字生成。
#                         预期：不崩 + 接管数下降 + 第 ④ 级逐字节比对应通过。
#
#   TEFLON_PERCHANNEL=1 **放开 weight/bias 的 per-channel（per-axis）量化接管**。
#   TEFLON_NNDUMP=1        【诊断】NN core 描述符**逐字节** dump（create_nn_config 收尾）。
#   TEFLON_KPAT=1          【诊断】NN core kernel_pattern **运行期强制开关**
#   TEFLON_FPARTIAL=1      【诊断】NN core **强制 PARTIAL cache 分支**（运行期 TEFLON_PARTIAL=1）
#                         用途：验证「判别量 = FULL-cache + 大占用」这个新候选。
#                         （TEFLON_KPAT=6|3 选择强制值；不设=原分档逻辑，行为不变）。
#                         用于单变量对照：244 通道 1×1 conv 挂死是否由 pattern 0x6 引起。
#                         ⚠️ 产物**改行为**，**不可交付**。
#                         用途：定位「244 通道 1×1 conv 使 NPU 作业挂死」的坏字段
#                         （真机实测：244→244 挂、122→122 不挂，见 reports/27、28）。
#                         只打印、**不改行为**；产物带取证打印，**不可交付**。
#                         补丁器：build-recipe/patch-etnaviv-perchannel-quant.py
#                         ⚠️ 默认关闭。不设时本步整段跳过，构建流程与产物完全不变。
#                         根因：etna_ml_operation_supported() 对 input/weight/bias/output
#                         四个张量统一要求
#                           tensor_quantization_supported(t) = (t->scales == NULL && …)
#                         ⇒ 任何 per-axis 模型卷积族接管数归零。
#                         但下降路径**早已实现** per-channel
#                         （etnaviv_ml_nn.c：scales[oc] 逐通道反量化 →
#                           calc_quantization() 求单一 scale →
#                           requantize_weights()/requantize_bias() 重量化），
#                         upstream 自己标注 "with a loss of accuracy"。
#                         ★ 只有 weight/bias 的下降路径处理了 per-axis；
#                           input/output 只读**标量** tensor->scale，
#                           放行它们会读到未设置的标量 ⇒ **静默算错**。
#                           故本补丁按张量区分，并自带该不变量的硬断言。
#                         ⚠️ 折叠**有精度损失**，本补丁只保证「能接管、不崩溃」，
#                            **不保证数值达标**；cos/top-1 必须设备实测（门 e）。
#                         ★ **input/output 仍严格** —— 那两个张量的下降路径只读
#                           **标量** tensor->scale，per-axis 张量的标量并未被设置；
#                           放行它们会读到未设置的标量 ⇒ **静默算错**。
#                           故只对 weight/bias 放开，补丁内与
#                           tools/check-perchannel-quant.py 各硬断言这一点。
#                         目标文件 etnaviv_ml.c。
#                         锚点由 tools/gen-perchannel-quant-patch.py 从真实源码逐字生成。
#                         预期：per-channel 模型的卷积接管数恢复 + 不崩；
#                               ③④ 判据**不因本补丁而达成**。
#
#   典型诊断构建：TEFLON_FIX_V7=1 TEFLON_DEBUG_INFO=1 TEFLON_NO_STRIP=1
#   典型修复构建：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 TEFLON_FSGUARD=1
#   典型诊断构建：同上 + TEFLON_RODIAG=1（上机另设 ETNA_MESA_DEBUG=ml_msgs）
#   本轮修复构建：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 TEFLON_FSGUARD=1 TEFLON_ROFIX=1
#   本轮诊断构建：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 TEFLON_FSGUARD=1 \
#                 TEFLON_ROFIX=1 TEFLON_RODIAG=1   ← 取证必须配 ROFIX
#                 TEFLON_ALIASDIAG=1 TEFLON_CREATEDIAG=1（上机另设 ETNA_MESA_DEBUG=ml_msgs）
#   per-channel 构建：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 TEFLON_FSGUARD=1 TEFLON_ROFIX=1
#                 TEFLON_PERCHANNEL=1
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
# ★ BUILD_DIR 允许是**相对**路径（默认 "mesa-build"），而此时 cwd 已经是它。
#   后面凡要按绝对路径写文件（自测日志 tee 等）必须用这个绝对路径，
#   否则相对路径会被拼成 <cwd>/mesa-build/... 而写错地方。
BUILD_DIR_ABS="$(pwd)"
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

# ── 3f. 【诊断】write_core_interleaved 的 out_channel 取证 ──────────────
# addr2line（CI 390…，unstripped + -g3）已把崩溃定位到
#   write_core_interleaved()  etnaviv_ml_nn_v7.c:345
#   wb_stream_write(&wb_stream, weights_map[out_channel][z][x][y]);
# 而 weights_map 是把权重缓冲区强转成 [out_channel][z][x][y] 四维视图。
# **已证实**（真机取证产物，见 3g）：`out_channel` 确实会超过 `output_channels`
#   —— 136 个独立组合 0 反例地复现公式，且出现 out_ch=123 > output_channels=122。
# 本补丁只打印数字、不改行为 —— 打印紧挨崩溃行，且 tf_load_check 日志无缓冲。
#
# ⚠️ 默认关闭。诊断产物不可交付。
if [ "${TEFLON_WCDIAG:-0}" = "1" ]; then
	echo "==> [3f/6] 打【诊断】write_core_interleaved out_channel 取证"
	PATCHER5="$SCRIPT_DIR/patch-etnaviv-write-core-diag.py"
	ML7C="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"
	if [ ! -f "$PATCHER5" ]; then
		echo "ERROR: 设了 TEFLON_WCDIAG=1 却缺 $PATCHER5"
		exit 1
	fi
	if [ ! -f "$ML7C" ]; then
		echo "ERROR: 找不到 $ML7C"
		exit 1
	fi
	python3 "$PATCHER5" "$ML7C" || {
		echo "ERROR: out_channel 取证补丁应用失败 —— 请先跑 tools/gen-wcdiag-patch.py 重新生成锚点"
		exit 1
	}
else
	echo "==> [3f/6] out_channel 取证补丁：默认关闭（设 TEFLON_WCDIAG=1 启用）"
fi

# ── 3g. 【修复】write_core_interleaved 的 out_channel 尾部越界钳位 ──────
# 【机制已由真机数据证实】
#   取证产物（TEFLON_WCDIAG=1，真机 per-tensor 模型）打出 136 个独立组合，
#   原式 core*kernels_in_superblock + kernel
#          + superblock*DIV_ROUND_UP(kernels_per_core,superblocks)*cores_used
#   **0 个反例地复现了全部实测 out_ch**，并确实出现 out_ch >= output_channels
#   （实测 122 / 123，而 output_channels = 122）→ weights_map[out_channel]
#   与 biases[out_channel] 越界读 → SIGSEGV。
#
# ⚠️⚠️ 本补丁**只加钳位，绝不改公式**。理由（来自上游自己的 v8 实现）：
#   etnaviv_ml_nn_v8.c:715-760 的 fill_weights() 是同源实现，它
#     :726  channel_per_superblock[superblock] = superblock * full_superblock * cores_used;
#          —— **原样保留**了 v7 末尾那个 `* cores_used`（曾被误判为「多余」）；
#     :754  channel_per_superblock[superblock] += kernels_in_superblock;
#          —— 逐 core 推进 kernels_in_superblock，**不是** kernels_per_core；
#     :729-731 kernels_per_core = output_channels/cores_used (+1 给前 core<余数)
#          —— v8 把余数分给前几个 core，于是槽位总数恰为 output_channels，
#             **没有幽灵槽**；这才是 v7 缺的修复（v7 用 DIV_ROUND_UP 均分，
#             实测 8*16=128 > 122 才产生越界槽）。
#   所以原式结构是有意的：多出来的槽对应的通道根本不存在（输出会被丢弃），
#   钳位到 output_channels-1 后写的是同一个通道的同样权重 —— 幂等且无害，
#   通道 0..output_channels-1 的权重与上游意图逐字节一致。
#
# 护栏用**钳位**而非 continue：函数内 append_bits(kernels_per_core, 16, ...)
# 已把「每 core 槽位数」写进位流头，跳过会让位流少槽而整体错位。
#
# ⚠️ 只修 write_core_interleaved 一处：同款公式在文件里另有 3 处
#    （write_core_6 ×2、write_core_sequential），属同一缺陷类但**无真机证据**，
#    不把未验证的改动混进交付产物。已记录为待修项。
#
# ⚠️ 默认关闭。不设时本步整段跳过，构建流程与产物完全不变。
# ⚠️ 产物仍须过四级判据第 ④ 级（与 x86 基准逐字节一致）才算「能用」。
#
# 用法：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 ./build-teflon.sh
if [ "${TEFLON_OCFIX:-0}" = "1" ]; then
	echo "==> [3g/6] 打【修复】write_core_interleaved out_channel 尾部越界钳位"
	PATCHER6="$SCRIPT_DIR/patch-etnaviv-outchannel-clamp.py"
	ML7O="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"
	if [ ! -f "$PATCHER6" ]; then
		echo "ERROR: 设了 TEFLON_OCFIX=1 却缺 $PATCHER6"
		exit 1
	fi
	if [ ! -f "$ML7O" ]; then
		echo "ERROR: 找不到 $ML7O"
		exit 1
	fi
	python3 "$PATCHER6" "$ML7O" || {
		echo "ERROR: out_channel 钳位补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出未修复的 .so，崩溃会照旧）。"
		echo "       请跑 tools/gen-ocfix-patch.py 从真实源码重新生成锚点。"
		exit 1
	}
else
	echo "==> [3g/6] out_channel 钳位补丁：默认关闭（设 TEFLON_OCFIX=1 启用）"
fi

# ── 3g2. 【修复】幻影通道 out_channel 钳位（write_core_6 / write_core_sequential） ──
# 【目的：消除 v7 三个写入器共有的幻影通道越界】
#   kernels_per_core = DIV_ROUND_UP(output_channels, cores_used)（v7:193）把通道总数
#   上取整到 cores_used 的倍数 ⇒ 末批多出 m 个幻影通道，被真实执行 ⇒
#   v7:226/391 权重越界读、v7:238/407 偏置越界读、v7:251/426 输出偏移越界写（GPU）。
#   钳位到 output_channels-1 ⇒ 幻影 kernel 变成「重算最后一个合法通道」，三处越界归零。
# ⚠️ write_core_interleaved 的同一处已由 OCFIX（TEFLON_OCFIX）覆盖，本补丁**不动**它
#    ⇒ 两个补丁可共存（锚点不同、函数不同）。
# ⚠️ 与 OCFIX 的区别：OCFIX 只覆盖 interleaved；1×1 conv 走 write_core_6、
#    depthwise 走 write_core_sequential ⇒ **这两条路径此前完全没有保护**。
# ⚠️ 默认关闭。不设时本步整段跳过，构建流程与产物完全不变。
#
# 用法：TEFLON_PHANTOM_CLAMP=1 ./build-teflon.sh
if [ "${TEFLON_PHANTOM_CLAMP:-0}" = "1" ]; then
	echo "==> [3g2/6] 打【修复】幻影通道 out_channel 钳位（write_core_6 + write_core_sequential）"
	PATCHER6C="$SCRIPT_DIR/patch-etnaviv-phantom-clamp.py"
	ML7C="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"
	if [ ! -f "$PATCHER6C" ]; then
		echo "ERROR: 设了 TEFLON_PHANTOM_CLAMP=1 却缺 $PATCHER6C"
		exit 1
	fi
	if [ ! -f "$ML7C" ]; then
		echo "ERROR: 找不到 $ML7C"
		exit 1
	fi
	python3 "$PATCHER6C" "$ML7C" || {
		echo "ERROR: 幻影通道钳位补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出**未修复**的 .so，越界会照旧）。"
		exit 1
	}
else
	echo "==> [3g2/6] 幻影通道 out_channel 钳位：默认关闭（设 TEFLON_PHANTOM_CLAMP=1 启用）"
fi

# ── 3e. 【修复】v7 calculate_bias_correction 漏 depthwise 特判
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

# ── 3h. 【修复】filter 深度 ≠ 输入通道数 的卷积不接管（回退 CPU）─────────────
# 【根因：真机 SIGSEGV + 本地 addr2line + 离线模型审计 四重互证】
#   崩溃地址 +0x22fda0（b9936131 与 f9bf1d51 完全相同）
#     → addr2line → subsample  etnaviv_ml_nn.c:424  out[x][y] = in[in_x][in_y][in_z];
#     → 调用链 etna_ml_lower_convolution → strided_to_normal(stride>1) → reshape → subsample
#   离线审计定位到崩溃分区里的算子：
#     op#58   CONV_2D input=[1,52,52,244] filter=[244,3,3,1] stride=2
#     op#153  CONV_2D input=[1,25,25,488] filter=[488,3,3,1] stride=2
#     **filter 的 in_c（最后一维）= 1，输入张量通道数 = 244 / 488**
#   算术：input_channels 取自输入张量 = 244 → wdims_in[3]=244
#         → map_in = input + out_channel*3*3*244 = input + out_channel*2196
#         而权重张量只有 244*3*3*1 = 2196 字节 → out_channel>=1 必然越界读。
#
# 为什么是「不接管」而不是「改系数排布」：
#   从 coeff 构造（weights_map[input_channels][w][h]）到 strided_to_normal 的整条路径
#   都建立在「filter 深度 == 输入通道」这个**不成立**的前提上，改单点只会把
#   越界换成**静默错误数值**。回退 CPU 是唯一不产生错误数值的做法，
#   也符合 Mesa 上游风格（etna_ml_operation_supported 本就拒绝 dilation 与 per-axis 量化）。
#   判据：非 depthwise 卷积要求 weight_tensor->dims[3] == input_tensors[0]->dims[3]；
#   depthwise 的 filter 布局是 [1,h,w,out_c]，dims[3] 是输出通道，豁免。
#
# 预期效果：不崩 + 接管数下降（那 2 个算子回退 CPU）+ 第 ④ 级逐字节比对应通过。
#
# ⚠️ 默认关闭。不设时本步整段跳过，构建流程与产物完全不变。
# ⚠️ 目标文件是 etnaviv_ml.c（不是 ml_nn*.c）。
#
# 用法：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 TEFLON_FSGUARD=1 ./build-teflon.sh
if [ "${TEFLON_FSGUARD:-0}" = "1" ]; then
	echo "==> [3h/6] 打【修复】filter 深度一致性守卫（不自洽卷积回退 CPU）"
	PATCHER7="$SCRIPT_DIR/patch-etnaviv-filter-shape-guard.py"
	MLFS="mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c"
	if [ ! -f "$PATCHER7" ]; then
		echo "ERROR: 设了 TEFLON_FSGUARD=1 却缺 $PATCHER7"
		exit 1
	fi
	if [ ! -f "$MLFS" ]; then
		echo "ERROR: 找不到 $MLFS"
		exit 1
	fi
	python3 "$PATCHER7" "$MLFS" || {
		echo "ERROR: filter 形状守卫补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出未修复的 .so，崩溃会照旧）。"
		echo "       请跑 tools/gen-fsguard-patch.py 从真实源码重新生成锚点。"
		exit 1
	}
else
	echo "==> [3h/6] filter 形状守卫：默认关闭（设 TEFLON_FSGUARD=1 启用）"
fi

# ── 3h2. 【修复】幻影通道形状门控（out_c %% 8 != 0 的卷积不接管） ─────────
# 【目的：零挂死兜底】Mesa v7 按 ceil(C/8) 分配通道 ⇒ C 不是 8 的倍数时末批多出
#   m = 8*ceil(C/8)-C 个「幻影通道」，被真实执行 ⇒ 权重/偏置/输出三处越界。
#   真机实测 1×1 conv 在 C=244/284 挂死（C=244 只挂 GPU 可 recover，C=284 整机失联）。
#   本门控把这类形状挡在 NPU 之外 ⇒ 该机制不可能再触发；代价是它们回落 CPU。
# 【是兜底不是根治】要拿回加速 ⇒ 模型侧 pad-to-8（tools/pad-to-8.py）或长期移植 v8 分配。
# ⚠️ 默认关闭。不设时本步整段跳过，构建流程与产物完全不变。
#
# 用法：TEFLON_PHANTOM_GATE=1 ./build-teflon.sh
if [ "${TEFLON_PHANTOM_GATE:-0}" = "1" ]; then
	echo "==> [3h2/6] 打【修复】幻影通道形状门控（out_c %% 8 != 0 ⇒ 回落 CPU）"
	PATCHER7G="$SCRIPT_DIR/patch-etnaviv-phantom-gate.py"
	MLPG="mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c"
	if [ ! -f "$PATCHER7G" ]; then
		echo "ERROR: 设了 TEFLON_PHANTOM_GATE=1 却缺 $PATCHER7G"
		exit 1
	fi
	if [ ! -f "$MLPG" ]; then
		echo "ERROR: 找不到 $MLPG"
		exit 1
	fi
	python3 "$PATCHER7G" "$MLPG" || {
		echo "ERROR: 幻影通道门控补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出**未门控**的 .so，挂死会照旧）。"
		exit 1
	}
else
	echo "==> [3h2/6] 幻影通道形状门控：默认关闭（设 TEFLON_PHANTOM_GATE=1 启用）"
fi

# ── 3i. 【诊断】read_outputs 越界取证 ─────────────────────────────────────
# 【目的：区分三条静态层面都能自洽的假说，必须用运行时数据】
#   真机现象（675a1091 / c8e8e658 均复现）：
#     Invoke 阶段 SIGSEGV，回溯 etna_ml_subgraph_read_outputs+0x188
#       ← +0x8564c == partition_invoke（frontends/teflon/tfl_device.c:702）
#     addr2line（CI 与本地复核一致）：0x2307e8 → etnaviv_ml.c:1076
#       （FSGUARD 净增 23 行 ⇒ 干净源码 1053 行 =
#        `((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;`）
#     反汇编：ldrb w0,[x24,w26]（src_map[k]），且 NULL 分支走 __assert_fail
#       ⇒ **src_map 非 NULL 但是野指针**
#
#   H1 output_idxs[i] 越界索引 subgraph->tensors dynarray
#      —— 本地已算：partition 1 输出 tensor=139、dynarray 大小=140，**不越界**
#   H2 pipe_buffer_size(res) != tensor->size
#      —— read_outputs 绕过 tensor->offset 直接 map 整个 resource，
#         循环上界却用 tensor->size
#   H3 resource 跨 context 失效
#      —— subgraph 是 device 级（tfl_device.c:603），context 每次 invoke 新建（:666）
#
#   本补丁打印：i / idx / n_tensors / IN|OOB / 槽指针 / res / tsize / toff
#                + map 后的 bsize 与 src_map
#   自我保护：只读 util_dynarray_element 的槽值，**不调 etna_ml_get_tensor**
#   （它在槽为 NULL 时会 calloc 并写入，idx 越界时等于往野地址写，会掩盖现场）。
#
# ⚠️ 该产物含取证打印，**不可交付**。
# ⚠️ 目标文件 etnaviv_ml.c（与 3h 同一文件，顺序在 3h 之后）。
# ⚠️ 默认关闭。⚠️ **上机必须同时设 ETNA_MESA_DEBUG=ml_msgs**，
#    否则 ML_DBG 不打印（格式串进 .rodata，但 DBG_ENABLED 是运行期判断）。
#
# 用法：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 TEFLON_FSGUARD=1 TEFLON_RODIAG=1
if [ "${TEFLON_RODIAG:-0}" = "1" ]; then
	if [ "${TEFLON_ROFIX:-0}" != "1" ]; then
		echo "ERROR: TEFLON_RODIAG=1 必须同时设 TEFLON_ROFIX=1"
		echo "       取证打印本身不改行为；但【不含 ROFIX 的取证产物】会把"
		echo "       read_outputs 的越界读重新引入 —— 2026-10-20 该形态产物在真机上"
		echo "       作业启动后几秒内整机硬挂、需断电重启（verify/reports/38）。"
		echo "       正确形态：ROFIX 在前、取证叠加在后（本脚本自动排序）。"
		exit 1
	fi
	PATCHER9="$SCRIPT_DIR/patch-etnaviv-readout-diag.py"
	if [ ! -f "$PATCHER9" ]; then
		echo "ERROR: 设了 TEFLON_RODIAG=1 却缺 $PATCHER9"
		exit 1
	fi
	echo "==> [3i/6] read_outputs 取证：已排队（叠加在 3j 的 ROFIX 之后执行）"
else
	echo "==> [3i/6] read_outputs 取证：默认关闭（设 TEFLON_RODIAG=1，且必须同时 TEFLON_ROFIX=1）"
fi

# ── 3j. 【修复】read_outputs 映射范围与读取范围对齐 ───────────────────────
# 【根因：真机运行时数据，RODIAG 取证（0078f5b2），三条假说一次性判定】
#   崩溃点（新产物重取，旧偏移已作废）：
#     etna_ml_subgraph_read_outputs+0x220 → etnaviv_ml.c:1095
#       = `((uint8_t *)(outputs[i]))[k] = src_map[k] - 128;`
#   [och-ro] 逐行：
#     idx=139 n_tensors=141 IN  bsize=940896 off_expected=940896   一致
#     idx=149 n_tensors=152 IN  bsize=610000 off_expected=305000   bsize=2×tsize
#     idx=151 n_tensors=152 IN  bsize=5000   off_expected=610000   ★ 致命
#
#   H1 idx 越界  → 排除（三组全部 IN）
#   H2 bsize≠tsize → **确认**：map 长度用 pipe_buffer_size(res)=5000，
#                     循环上界用 tensor->size=610000 ⇒ 越界读 605000 字节
#   H3 跨 context 资源失效 → 排除（src_map 均为合法 mmap 地址）
#
# 修法：map 参数改成与 invoke（etnaviv_ml.c:908）完全同构 ——
#       (tensor->offset, tensor->size)，并加 idx 边界检查。
#   不加 min() 截断：前两处的教训是「改单点会把越界换成静默错误数值」。
#   resource 真不够大时 map 返回 NULL → assert 明确失败（可观测）。
#
# ✅ 可与 TEFLON_RODIAG=1 叠加：本脚本保证**先 ROFIX、再取证**。
#    （2026-10-20 之前两者互斥，导致「ROFIX=0 的取证产物」被当成低危仪器；
#     该形态在真机上整机硬挂 —— 见 verify/reports/38。）
# ⚠️ 该修复只针对**崩溃**；tsize 被 etna_ml_tensor_offset 批量改写的更深问题未处理。
# ⚠️ 默认关闭。
#
# 用法：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 TEFLON_FSGUARD=1 TEFLON_ROFIX=1
if [ "${TEFLON_ROFIX:-0}" = "1" ]; then
	echo "==> [3j/6] 打【修复】read_outputs 范围对齐 invoke + idx 边界检查"
	PATCHER10="$SCRIPT_DIR/patch-etnaviv-readout-offset-fix.py"
	MLRF="mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c"
	if [ ! -f "$PATCHER10" ]; then
		echo "ERROR: 设了 TEFLON_ROFIX=1 却缺 $PATCHER10"
		exit 1
	fi
	if [ ! -f "$MLRF" ]; then
		echo "ERROR: 找不到 $MLRF"
		exit 1
	fi
	python3 "$PATCHER10" "$MLRF" || {
		echo "ERROR: read_outputs 范围修复补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出未修复的 .so，崩溃会照旧）。"
		echo "       请跑 tools/gen-rofix-patch.py 从真实源码重新生成锚点。"
		exit 1
	}
	# ── 3j-2. 【诊断】取证叠加（必须在 ROFIX 之后）────────────────────────
	# 取证打印只读、不改行为；ROFIX 必须先落地，否则越界读重新出现。
	if [ "${TEFLON_RODIAG:-0}" = "1" ]; then
		echo "==> [3j-2] 打【诊断】read_outputs 越界取证（叠加在 ROFIX 之上）"
		python3 "$PATCHER9" "$MLRF" || {
			echo "ERROR: 取证补丁叠加失败 —— 不要跳过校验"
			echo "       （会静默产出无取证的 .so，白跑一轮真机验证）。"
			exit 1
		}
	fi
else
	echo "==> [3j/6] read_outputs 范围修复：默认关闭（设 TEFLON_ROFIX=1 启用）"
fi

# ── 3k. 【诊断】alias（批量元数据迁移）取证 ───────────────────────────────
# 【为什么需要：ROFIX(6acfef1f) 把野指针换成了可观测的 assert】
#   u_inlines.h:503: pipe_buffer_map_range:
#   Assertion `offset + length <= buffer->width0' failed
#   ⇒ tensor 元数据（offset/size）与 resource 实际大小不一致。
#
#   RODIAG(0078f5b2) 实测三组：
#     idx=139 bsize=940896 off_expected=940896
#     idx=149 bsize=610000 off_expected=305000
#     idx=151 bsize=5000   off_expected=610000
#   三个 off_expected 全是 5000 的整数倍、toff 全 0 —— 静态分析解释不了，需运行时数据。
#
# 【可疑代码】
#   reference_tensor_with_offset (:198-224)  同时改 size 与 offset
#   recreate_tensor              (:226-246)  **只改 size，不改 offset** ← 不对称，是条线索
#   两者都是：把「所有共享 old_res 的 tensor」一律改成同一个 (offset,size)。
#   调用点 :540/:552/:562/:569 = CONCAT / SPLIT / NN且input_count>1(Add/Subtract) / BYPASS
#
# 【打印】ref 的 src/dst/offset/size + **每个被迁移 tensor 的旧 size/offset**（改之前抓）
#         recreate 的 idx/new_size/旧 size/offset
#
# ★ 与 TEFLON_ROFIX=1 **可同时开**：本补丁改的是 reference_tensor_with_offset /
#   recreate_tensor，ROFIX 改的是 read_outputs —— 不同函数，锚点不冲突。
#
# ⚠️ 该产物含取证打印，不可交付。
# ⚠️ 上机必须同时设 ETNA_MESA_DEBUG=ml_msgs（ML_DBG 是运行期判断）。
# ⚠️ 默认关闭。
#
# 用法：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 TEFLON_FSGUARD=1 TEFLON_ROFIX=1 TEFLON_ALIASDIAG=1
if [ "${TEFLON_ALIASDIAG:-0}" = "1" ]; then
	echo "==> [3k/6] 打【诊断】alias 迁移取证（该产物不可交付）"
	PATCHER11="$SCRIPT_DIR/patch-etnaviv-alias-diag.py"
	MLAD="mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c"
	if [ ! -f "$PATCHER11" ]; then
		echo "ERROR: 设了 TEFLON_ALIASDIAG=1 却缺 $PATCHER11"
		exit 1
	fi
	if [ ! -f "$MLAD" ]; then
		echo "ERROR: 找不到 $MLAD"
		exit 1
	fi
	python3 "$PATCHER11" "$MLAD" || {
		echo "ERROR: alias 取证补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出无取证的 .so，白跑一轮真机验证）。"
		echo "       请跑 tools/gen-aliasdiag-patch.py 从真实源码重新生成锚点。"
		exit 1
	}
else
	echo "==> [3k/6] alias 迁移取证：默认关闭（设 TEFLON_ALIASDIAG=1 启用）"
fi

# ── 3l. 【诊断】create_tensor 分配取证 ───────────────────────────────────
# 【为什么需要：ALIASDIAG(64933494) 把范围收窄到了「buffer 分配大小」】
#   已确定：tensor->size = 610000 正确（与模型一致）
#   已确定：pipe_buffer_size(res) = 5000 —— 错的是 buffer 分配，不是元数据
#   已排除（真机数据）：recreate_tensor 0 次调用；批量迁移 13 次值全没变
#   模型里**没有任何 5000 字节的张量** ⇒ 5000 是驱动自己算出来的
#   且 5000 = 2×50×50，而 610000 = 244×50×50（差 122 倍）
#
# 【嫌疑：etna_ml_create_tensor 的早退条件】
#   真实源码（不是我最初以为的 `res != NULL && size == ...`）：
#       if (res != NULL) {
#          assert(size == tensor->size);      // ★ release 构建关掉 assert ⇒ 静默早退
#          return;
#       }
#   只要有 resource 就早退 ⇒ 「先以小 size 建、后以大 size 调用」会**静默保留小 buffer**。
#
# 【打印】enter / EARLY-RETURN（带 "*** size 不一致！***" 标记）/ destroy / created，
#        每条都带 `pipe_buffer_size(res)` —— 实际分配大小，是本轮的关键列。
#
# ★ 与 ALIASDIAG / ROFIX **可同时开**：注入函数互不相同。
#   标记锚点复用 `#include <stdio.h>`（ALIASDIAG 在其后插标记，该行仍唯一，已实测）。
#
# ⚠️ 该产物含取证打印，不可交付。
# ⚠️ 上机必须同时设 ETNA_MESA_DEBUG=ml_msgs。
# ⚠️ 默认关闭。
#
# 用法：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 TEFLON_FSGUARD=1 TEFLON_ROFIX=1 \
#       TEFLON_ALIASDIAG=1 TEFLON_CREATEDIAG=1
if [ "${TEFLON_CREATEDIAG:-0}" = "1" ]; then
	echo "==> [3l/6] 打【诊断】create_tensor 分配取证（该产物不可交付）"
	PATCHER12="$SCRIPT_DIR/patch-etnaviv-create-diag.py"
	MLCD="mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c"
	if [ ! -f "$PATCHER12" ]; then
		echo "ERROR: 设了 TEFLON_CREATEDIAG=1 却缺 $PATCHER12"
		exit 1
	fi
	if [ ! -f "$MLCD" ]; then
		echo "ERROR: 找不到 $MLCD"
		exit 1
	fi
	python3 "$PATCHER12" "$MLCD" || {
		echo "ERROR: create_tensor 取证补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出无取证的 .so，白跑一轮真机验证）。"
		echo "       请跑 tools/gen-creatediag-patch.py 从真实源码重新生成锚点。"
		exit 1
	}
else
	echo "==> [3l/6] create_tensor 取证：默认关闭（设 TEFLON_CREATEDIAG=1 启用）"
fi

# ── 3m. 【修复】放开 weight/bias 的 per-channel（per-axis）量化接管 ──────
# 【为什么需要：per-channel 模型卷积族接管数归零】
#   etna_ml_operation_supported() 的 CONVOLUTION 分支对
#   input/weight/bias/output **四个张量统一**调用
#       tensor_quantization_supported(t) = (t->scales == NULL && t->zero_points == NULL)
#   ⇒ 任何带 per-axis 参数的模型，卷积族全部不接管。
#
# 【但下降路径其实早就实现了 per-channel，只是被这道门挡住】
#   etnaviv_ml_nn.c 里权重按 scales[oc] 逐通道反量化，
#   再用 calc_quantization() 沿所有通道求出**单一** scale，
#   最后 requantize_weights()/requantize_bias() 重量化到该标量。
#   upstream 自己打印："hardware doesn't support it natively, so it will be
#   workarounded by requantizing on the fly, **with a loss of accuracy**"。
#   ⇒ NN multiplier 确实是标量（operation->weight_scale），
#     但驱动已内置「按通道反量化再折叠」的 workaround。
#
# 【★ 为什么不能把那张门无脑翻成 true】
#   那张门对四个张量一律同判，但只有 weight/bias 的下降路径处理了 per-axis：
#     weight : scales[oc] -> calc_quantization -> requantize_weights()   ✅
#     bias   : bias_tensor->scales[oc] -> requantize_bias()              ✅
#     input  : input_tensors[0]->scale（**标量**）                          ❌
#     output : output_tensors[0]->scale（**标量**）                         ❌
#   放行 input/output 会读到**未被设置的标量 scale** ⇒ 静默算错。
#   所以本补丁按张量区分，并自带「替换后 input/output 仍严格」的硬断言。
#
# 【代价（不藏着）：折叠有精度损失，小权重可能被压到 0】
#   本补丁只保证「能接管、不崩溃」，**不保证数值达标**；
#   cos / top-1 必须设备实测（门 e）判定，不能由本补丁推断。
#
# 【与 FSGUARD 的关系】
#   两者都改 etna_ml_operation_supported 附近：FSGUARD 的 OLD 锚点结尾正是
#   `// Dilation and per-axis quantization not yet implemented`，
#   就在本补丁改的四连调用**上面**。
#   selftest-perchannel-quant.py 已验证 **双向顺序都共存**。
#
# ⚠️ 默认关闭。不设时本步整段跳过，构建流程与产物完全不变。
# ⚠️ 目标文件是 etnaviv_ml.c（与 3h/3i/3j/3l 同一文件，顺序在最后）。
#
# 用法：TEFLON_FIX_V7=1 TEFLON_OCFIX=1 TEFLON_FSGUARD=1 TEFLON_ROFIX=1 \
#       TEFLON_PERCHANNEL=1
if [ "${TEFLON_PERCHANNEL_FOLDEARLY:-0}" = "1" ] && [ "${TEFLON_PERCHANNEL:-0}" != "1" ]; then
	echo "ERROR: TEFLON_PERCHANNEL_FOLDEARLY=1 但 TEFLON_PERCHANNEL=0"
	echo "       fold-early 修的是「per-channel 被准入之后」的下降路径；"
	echo "       门没开就没有可修的对象，产物会与不加任何补丁一致 ⇒ 拒绝构建。"
	exit 1
fi
if [ "${TEFLON_PERCHANNEL:-0}" = "1" ]; then
	echo "==> [3m/6] 打【修复】per-channel 量化门（只放开 weight/bias）"
	PATCHER13="$SCRIPT_DIR/patch-etnaviv-perchannel-quant.py"
	MLPC="mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c"
	if [ ! -f "$PATCHER13" ]; then
		echo "ERROR: 设了 TEFLON_PERCHANNEL=1 却缺 $PATCHER13"
		exit 1
	fi
	if [ ! -f "$MLPC" ]; then
		echo "ERROR: 找不到 $MLPC"
		exit 1
	fi
	python3 "$PATCHER13" "$MLPC" || {
		echo "ERROR: per-channel 量化门补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出未放开的 .so，接管数仍是 0，白跑一轮真机验证）。"
		echo "       请跑 tools/gen-perchannel-quant-patch.py 从真实源码重新生成锚点。"
		exit 1
	}
	# 真源码复核：打完补丁后在**真实 Mesa 源码**上硬判三件事。
	# 这一步替代不了 tools/selftest-perchannel-quant.py（那跑在合成 mock 上），
	# 两者互补：mock 证逻辑，真源码证锚点仍然命中。
	if [ -f "$SCRIPT_DIR/../tools/check-perchannel-quant.py" ]; then
		python3 "$SCRIPT_DIR/../tools/check-perchannel-quant.py" "$MLPC" || {
			echo "ERROR: per-channel 补丁在真实源码上的复核未通过"
			exit 1
		}
	fi
# ---------------------------------------------------------------- #
	# 3m-2. per-channel **布局契约**修复（同一个开关，不另开）
	#
	# 【为什么必须在同一个开关里，不能拆成第二个开关】
	#   门禁放开与布局修复都只可能在 per-channel 被准入时才走到；
	#   拆开只会允许「开一个不启另一个」的坏组合 —— 那必然是错的。
	#   单一 TEFLON_PERCHANNEL = 单一语义 = 单一产物来源。
	#
	# 【修什么】（详见 tools/selftest-perchannel-layout-math.py 的镜像）
	#   calc_quantization / requantize_weights 都按 [out][W][H][in] 读权重，
	#   但它们跑在 transpose() 之后，而 transpose() 的输出是 [out][in][W][H]。
	#   depthwise 时 nn.c:239-240 还把 input_channels 压成 1：
	#     - calc_quantization 只覆盖 0.41% 的权重 ⇒ scale 本身就是错的；
	#     - requantize_weights 把缓冲区压成 1/244 ⇒ v7 未写回 in_c ⇒
	#       create_coeffs_v7 按 in_c=244 选 write_core_interleaved ⇒ 越界 ⇒ SIGSEGV。
	#
	# 【非 depthwise 情形经证明无害，见报告 21】
	#   两个索引都是到 [0,out*W*H*in) 的双射且同下标读写，oc 又能从下标还原
	#   ⇒ 置换自逆、数值不变。本补丁在那一支只是把契约写明确。
	# ---------------------------------------------------------------- #
	MLPNN="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn.c"
	MLV7="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn_v7.c"
	PATCHER14="$SCRIPT_DIR/patch-etnaviv-perchannel-layout.py"
	if [ ! -f "$PATCHER14" ]; then
		echo "ERROR: 缺 $PATCHER14"
		exit 1
	fi
	for f in "$MLPNN" "$MLV7"; do
		if [ ! -f "$f" ]; then
			echo "ERROR: 找不到 $f"
			exit 1
		fi
	done
	# ── 3m-2a. 【替代方案】per-channel **折叠提前**（fold-early，报告 55 方案 A）──
	#
	# 与 3m-2b 的布局补丁**互斥**：两者都整函数重写 calc_quantization /
	# requantize_weights，且对 depthwise 的 input_channels 处置相反
	# （layout 补丁压成 1；fold-early 保持 C）。
	# 显式用 TEFLON_PERCHANNEL_FOLDEARLY=1 选择；默认仍走 3m-2b ⇒ 现有交付件不变。
	#
	# 【为什么 fold-early 更对】（报告 55，已过 OCA 评审）
	#   v7 没有 depthwise 模式（expand_depthwise 注释：hardware doesn't support those），
	#   MobileNetV1 的 depthwise 能到 99.70% 走的是**稠密 C×C 展开**。
	#   把 in_c 压成 1 让 C 个输出共享同一输入通道 ⇒ 语义塌陷。
	#   折叠提前后深度方向保持 C、写入器回 interleaved、缓冲尺寸恒等于原始尺寸的派生量
	#   ⇒ 「缩容后仍按 C 读」的越界**结构性消失**（不是靠钳位）。
	if [ "${TEFLON_PERCHANNEL_FOLDEARLY:-0}" = "1" ]; then
		echo "==> [3m/6] per-channel 折叠提前（fold-early，方案 A）"
		PATCHER15="$SCRIPT_DIR/patch-etnaviv-perchannel-foldearly.py"
		if [ ! -f "$PATCHER15" ]; then
			echo "ERROR: 设了 TEFLON_PERCHANNEL_FOLDEARLY=1 却缺 $PATCHER15"
			exit 1
		fi
		# ★ 打补丁**之前**核对（此时失配即为真失配）：三个 body md5 钉死 + 插入锚点唯一。
		python3 "$PATCHER15" "$MLPNN" --verify || {
			echo "ERROR: fold-early 的上游基准失配（源码与报告中钉死的 md5 不同）"
			echo "       请重读 verify/reports/55 的假设，更新补丁与 md5 后重跑。"
			exit 1
		}
		# 结构与数值自测（正控制 19 项 + 负控制/反作弊），**必须排在打补丁之前**：
		# 它内部会 --verify 未打补丁的原文，打完就必然失配。
		if [ -f "$SCRIPT_DIR/../tools/selftest-perchannel-foldearly.py" ]; then
			python3 "$SCRIPT_DIR/../tools/selftest-perchannel-foldearly.py" \
				--src-root "$(cd "$(dirname "$MLPNN")" && pwd)" \
				> "$BUILD_DIR_ABS/selftest-perchannel-foldearly.log" 2>&1
			_rc=$?
			cat "$BUILD_DIR_ABS/selftest-perchannel-foldearly.log"
			if [ "$_rc" -ne 0 ]; then
				echo "ERROR: fold-early 补丁自测未通过（rc=$_rc）"
				exit 1
			fi
		fi
		# ★ fold-early 整函数重写了 requantize_weights ⇒ 3m2 的 PCDOMAIN 锚点
		#   在它上面必然失配（run 39754283314 实测）。因此**域修复并入本补丁**，
		#   由 --pcdomain 选择；语义与 patch-etnaviv-perchannel-domain.py 的 NEW
		#   逐项相同（int32 中间量 / CLAMP / 同一运行期门 / 同一标记字符串）。
		FE_PCDOMAIN=""
		if [ "${TEFLON_PCDOMAIN_PATCH:-0}" = "1" ]; then
			FE_PCDOMAIN="--pcdomain"
			echo "  [foldearly] 域修复并入：--pcdomain（等价于 3m2 PCDOMAIN，语义不变）"
		fi
		python3 "$PATCHER15" "$MLPNN" $FE_PCDOMAIN || {
			echo "ERROR: fold-early 补丁应用失败 —— 不要跳过校验"
			echo "       （会静默产出仍按旧顺序折叠的 .so，depthwise 语义照旧塌陷）。"
			exit 1
		}
		if ! grep -qF 'och-patch-perchannel-foldearly' "$MLPNN"; then
			printf '\n/* och patch: compile-time marker */\n__attribute__((used)) static const char och_fe_marker[] = "och-patch-perchannel-foldearly";\n' >> "$MLPNN"
		fi
		# 互斥证明：另一个补丁的标记必须缺席
		if grep -qF 'och-patch-perchannel-layout' "$MLPNN"; then
			echo "ERROR: fold-early 与 layout 两个补丁同时出现在源码里（互斥被破坏）"
			exit 1
		fi
		echo "==> [3m/6] fold-early：折叠前移 + 原地折叠（尺寸不变）+ depthwise 保持 in_c=C"
	else
	# ★ 一切「核对未打补丁的原文」的检查都必须排在打补丁**之前**。
	#   打完补丁后 calc_quantization / requantize_weights 已被整函数替换，
	#   再去核对「OLD_* 是否命中原始文本」必然失配 ——
	#   这个坑连栽两次：
	#     run 39226436430 —— --verify 排在打补丁之后
	#     run 39227152464 —— 结构自测排在打补丁之后（它内部也跑 --verify）
	#   ⇒ 改成门来查：tools/check-patch-verify-order.py。
	MLNN_DIR="$(cd "$(dirname "$MLPNN")" && pwd)"
	if [ -f "$SCRIPT_DIR/../tools/gen-perchannel-layout-patch.py" ]; then
		python3 "$SCRIPT_DIR/../tools/gen-perchannel-layout-patch.py" --verify \
			--src "$MLPNN" --src-v7 "$MLV7" --out "$PATCHER14" || {
			echo "ERROR: 布局补丁锚点与真实源码不符（尚未打补丁，此时失配即为真失配）"
			exit 1
		}
	fi
	# 结构自测：幂等 / fail-closed / 两种 diag 模式。
	#   它自己会拷一份临时副本去打，所以**不依赖**工作树是否已打补丁；
	#   但它第一步就是 --verify，因此必须排在打补丁之前。
	#   --src-root 必须给**绝对路径**：Mesa 在 $BUILD_DIR 下，不在仓库根。
	if [ -f "$SCRIPT_DIR/../tools/selftest-perchannel-layout.py" ]; then
		python3 "$SCRIPT_DIR/../tools/selftest-perchannel-layout.py" \
			--src-root "$MLNN_DIR" > "$BUILD_DIR_ABS/selftest-perchannel-layout.log" 2>&1
		_rc=$?
		cat "$BUILD_DIR_ABS/selftest-perchannel-layout.log"
		if [ "$_rc" -ne 0 ]; then
			echo "ERROR: per-channel 布局补丁结构自测未通过（rc=$_rc）"
			exit 1
		fi
	fi
	# ★ 一切「核对未打补丁的原文」的检查都必须排在打补丁**之前**。
	#   打完补丁后 calc_quantization / requantize_weights 已被整函数替换，
	#   再去核对「OLD_* 是否命中原始文本」必然失配 ——
	#   这个坑连栽两次：
	#     run 39226436430 —— --verify 排在打补丁之后
	#     run 39227152464 —— 结构自测排在打补丁之后（它内部也跑 --verify）
	#   ⇒ 改成门来查：tools/check-patch-verify-order.py。
	#   （本块在打补丁之前，与上面的 --verify / 结构自测一起 constitutes
	#    「打补丁前的 pristine 原文核对区」。）
	python3 "$PATCHER14" "$MLPNN" --v7 "$MLV7" --diag "${TEFLON_PERCHANNEL_DIAG:-1}" || {
		echo "ERROR: per-channel 布局补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出仍按错误布局读权重的 .so，崩溃照旧，白跑一轮真机验证）。"
		echo "       请跑 tools/gen-perchannel-layout-patch.py 从真实源码重新生成锚点。"
		exit 1
	}
	# ★ 编译期标记：让 strings 门能硬判「这个补丁真的在二进制里」。
	#   少了它，产物侧只能靠 [och-pcl] 那几个格式串，判据不够独立
	#   （run 39229309524 的产物里这个标记就是缺席的）。
	#   用 shell 追加而不是在补丁器的三层嵌套模板里拼字符串 —— 那样转义太易错。
	if ! grep -qF 'och-patch-perchannel-layout' "$MLPNN"; then
		printf '\n/* och patch: compile-time marker */\n__attribute__((used)) static const char och_pcl_marker[] = "och-patch-perchannel-layout";\n' >> "$MLPNN"
	fi

	# 数值自测：把下降路径镜像成 Python 逐元素比对（不依赖编译、不依赖设备）。
	if [ -f "$SCRIPT_DIR/../tools/selftest-perchannel-layout-math.py" ]; then
		python3 "$SCRIPT_DIR/../tools/selftest-perchannel-layout-math.py" \
			> "$BUILD_DIR_ABS/selftest-perchannel-layout-math.log" 2>&1
		_rc=$?
		cat "$BUILD_DIR_ABS/selftest-perchannel-layout-math.log"
		if [ "$_rc" -ne 0 ]; then
			echo "ERROR: per-channel 布局数值自测未通过（rc=$_rc）"
			exit 1
		fi
	fi
	echo "==> [3m/6] 布局契约修复：calc_quantization + requantize_weights + v7 写回"
	echo "==> [3m/6] 诊断打印：TEFLON_PERCHANNEL_DIAG=${TEFLON_PERCHANNEL_DIAG:-1}（0=无诊断候选）"
	fi

# ── 3k7. 【修复】ML 层自建资源的 width0=0（Synology Photos 主题识别模型的 abort）──
# 现象：photos 的 concept 模型（FC 输出 rank-2）在推理期 abort：
#        pipe_buffer_map_range: Assertion `offset + length <= buffer->width0' failed.
# 根因：etna_ml_create_resource() 调 pipe_buffer_create(pscreen, 0, …, size)
#        ⇒ Gallium 把 width 直接写进 res->width0 ⇒ 自建资源 width0=0 ⇒ 任何 map_range 必炸。
#        （未打补丁的上游原版同样 abort ⇒ 与本仓其它改动无关。）
# 修法：width 取 size（width0 的语义就是「可映射的字节范围」）；不改任何数值。
#
#   构建：TEFLON_MLWIDTH_PATCH=1
#
if [ "${TEFLON_MLWIDTH_PATCH:-0}" = "1" ]; then
	echo "==> [3k7/6] 打【修复】ML 自建资源 width0=0"
	PATCHER_MLW="$SCRIPT_DIR/patch-etnaviv-ml-resource-width.py"
	MLMLW="mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c"
	if [ ! -f "$PATCHER_MLW" ]; then
		echo "ERROR: 设了 TEFLON_MLWIDTH_PATCH=1 却缺 $PATCHER_MLW"
		exit 1
	fi
	if [ ! -f "$MLMLW" ]; then
		echo "ERROR: 找不到 $MLMLW"
		exit 1
	fi
	python3 "$PATCHER_MLW" "$MLMLW" || {
		echo "ERROR: ML 资源 width 补丁应用失败 —— 不要跳过校验。"
		exit 1
	}
else
	echo "==> [3k7/6] ML 自建资源 width 修复：默认关闭（设 TEFLON_MLWIDTH_PATCH=1 启用）"
fi

# ── 3k8. 【修复】通用 rank>4 守卫（任何算子都不接管 rank>4 激活）─────────────
# 现象：Synology Photos 主题识别模型（rank-5 折叠形态）推理期 abort：
#        pipe_buffer_map_range: Assertion `offset + length <= buffer->width0' failed.
# 根因：delegate 接管了 RESHAPE（tfl_device.c 支持 kTfLiteBuiltinReshape），
#        而 ML 契约没有「维度个数」字段 ⇒ 驱动算 size 时丢掉 rank-5 的 dims[0]
#        （真实 610,000 被算成 305,000）⇒ 资源/尺寸错 ⇒ abort。
#        已有的 R5GUARD 抓不到，因为它只挂在 CONVOLUTION 分支上。
# 修法：在 etna_ml_operation_supported() 入口做**通用**检查（input/output 全部遍历）。
#       命中 ⇒ 不接管（回落 CPU）。不改任何数值。
#
#   构建：TEFLON_MLR5G_PATCH=1
#
if [ "${TEFLON_MLR5G_PATCH:-0}" = "1" ]; then
	echo "==> [3k8/6] 打【修复】通用 rank>4 守卫"
	PATCHER_R5G="$SCRIPT_DIR/patch-etnaviv-ml-rank5-guard.py"
	MLR5G="mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c"
	if [ ! -f "$PATCHER_R5G" ]; then
		echo "ERROR: 设了 TEFLON_MLR5G_PATCH=1 却缺 $PATCHER_R5G"
		exit 1
	fi
	if [ ! -f "$MLR5G" ]; then
		echo "ERROR: 找不到 $MLR5G"
		exit 1
	fi
	python3 "$PATCHER_R5G" "$MLR5G" || {
		echo "ERROR: 通用 rank>4 守卫补丁应用失败 —— 不要跳过校验。"
		exit 1
	}
else
	echo "==> [3k8/6] 通用 rank>4 守卫：默认关闭（设 TEFLON_MLR5G_PATCH=1 启用）"
fi

# ── 3m2. 【修复】per-channel（per-axis）权重折叠后的域错误 ────────────────
# 背景（报告 46，真机实测）：
#   权重 scale **不均匀**的 int8 模型（C=248）上 NPU 输出**全 255 饱和**，
#   而 CPU 正常。根因：requantize_weights() 的 signed 分支用 int8_t 截断
#   折叠后的 uint8 域值，**又减了一次 0x80**；而 per-channel 路径的
#   coef_zero_point = zp_new（本例 0，**不是 128**）⇒ 硬件按 `字节 − 0` 解释
#   ⇒ 非对角权重 0 → 128、248 项各带 +128 ⇒ 巨额正累加 ⇒ 饱和。
#   离线复算（verify/perchannel-prediction.py）与设备输出**逐字节 100% 吻合**。
#
#   构建：TEFLON_PCDOMAIN_PATCH=1 ；运行：TEFLON_PCDOMAIN=1（不设 = 上游行为）
#
# ★ 必须排在 **perchannel-layout 补丁之后**（3m 段）：那个补丁整函数重写了
#   requantize_weights（map_in[oc][src_ic][w][h] / map_out[oc][ic][w][h]），
#   本补丁的锚点是它的**输出形态** ⇒ 顺序反了会锚点失配（run 39565432373 就这么挂的）。
#
# ⚠️ 这是**改行为**的修复，默认关闭。交付前需另行决策（产物语义变更）。
if [ "${TEFLON_PCDOMAIN_PATCH:-0}" = "1" ] && [ "${TEFLON_PERCHANNEL_FOLDEARLY:-0}" = "1" ]; then
	# fold-early 已把**同一语义**并入自己的 requantize_weights（--pcdomain）。
	# 这里再打一遍既会锚点失配，也会造成「两处实现漂移」隐患 ⇒ 明确跳过并自证。
	if ! grep -qF '[och-pcdomain]' "$MLPNN"; then
		echo "ERROR: 走了 fold-early + PCDOMAIN，但源码里没有 [och-pcdomain] 标记"
		echo "       ⇒ 域修复没并进去，产物会整片饱和。拒绝继续。"
		exit 1
	fi
	echo "==> [3m2/6] per-channel 域修复：已并入 fold-early（--pcdomain），跳过独立补丁"
elif [ "${TEFLON_PCDOMAIN_PATCH:-0}" = "1" ]; then
	echo "==> [3m2/6] 打【修复】per-channel 权重折叠后的域错误"
	PATCHER_PCD="$SCRIPT_DIR/patch-etnaviv-perchannel-domain.py"
	MLPCD="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn.c"
	if [ ! -f "$PATCHER_PCD" ]; then
		echo "ERROR: 设了 TEFLON_PCDOMAIN_PATCH=1 却缺 $PATCHER_PCD"
		exit 1
	fi
	if [ ! -f "$MLPCD" ]; then
		echo "ERROR: 找不到 $MLPCD"
		exit 1
	fi
	python3 "$PATCHER_PCD" "$MLPCD" || {
		echo "ERROR: per-channel 域修复补丁应用失败 —— 不要跳过校验。"
		exit 1
	}
else
	echo "==> [3m2/6] per-channel 域修复：默认关闭（设 TEFLON_PCDOMAIN_PATCH=1 启用）"
fi

	# ---------------------------------------------------------------- #
	# 3n. R5GUARD —— rank-5 激活 guard（och 报告 11）
	#
	# 【为什么需要】
	#   契约里没有「维度个数」字段，delegate 写入最后 4 维；
	#   驱动一律算 dims[1]*dims[2]*dims[3]，恒定丢 dims[0]。
	#   rank-4 激活 dims[0]=1 ⇒ 无害；rank-5 激活 dims[0]=122 ⇒ 致命（size 5000 vs 610000）。
	#   ⇒ 含 rank-5 的子图不接管，回退 CPU，不再 abort。
	#
	# 【predicate】data == NULL && dims[0] != 1（och 给的签名）
	#   ⚠️ 误命中率必须先在真机上验证（P0），本补丁只保证「不再 abort」。
	#
	# 【P0 未做】och 要求先验误命中率。本补丁先入库，等设备侧数据。
	# ---------------------------------------------------------------- #
	if [ "${TEFLON_R5GUARD:-0}" = "1" ]; then
		echo "==> [3n/6] 打【修复】R5GUARD（rank-5 激活 guard，回退 CPU）"
		PATCHER15="$SCRIPT_DIR/patch-etnaviv-r5guard.py"
		ML="mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c"
		if [ ! -f "$PATCHER15" ]; then
			echo "ERROR: 缺 $PATCHER15"
			exit 1
		fi
		python3 "$PATCHER15" "$ML" || {
			echo "ERROR: R5GUARD 补丁应用失败"
			exit 1
		}
		echo "  [r5guard] 已加：含 rank-5 的子图不接管、回退 CPU"
		echo "  [r5guard] ⚠️ 接管数必须真机实测；跌到不可接受时停下来报告"
	else
		echo "==> [3n/6] R5GUARD：默认关闭（设 TEFLON_R5GUARD=1 启用）"
	fi
	echo "==> [3m/6] per-channel 量化门：默认关闭（设 TEFLON_PERCHANNEL=1 启用）"
fi

# ── 3i. 【诊断】NN core 描述符 dump（逐字节取证）──────────────────────
# 背景（真机实测，见 verify/reports/27、28）：
#   挂死最小单元已缩到**单个 1×1 CONV_2D（244→244 通道）**：
#     段 #9 前 3 算子（RESHAPE,TRANSPOSE,CONV）→ 挂（520 ms/次，内核复位 4 次）
#     段 #12 前 4 算子（…,CONV）              → 挂（512 ms/次）
#     对照 段 #4 的 122→122 同型 1×1 conv      → 不挂（14 ms，零复位）
#   驱动侧判挂死的依据是「前端 DMA 指针不再前进」⇒ NPU 连作业都没开始推进
#   ⇒ 极可能是**下发给硬件的描述符本身不合法**。
# 本补丁在 create_nn_config() 填完描述符后，把**整份描述符的原始字节**打出来；
# 与不挂的 122 通道作业**逐字节 diff** 即可锁定是哪个字段越界/错位。
#   struct etna_nn_params 是**位域**结构体 ⇒ 打原始字节（而非逐字段名）才不受改名影响，
#   一次 dump 覆盖全部字段，之后离线解码，不必再改一次补丁、跑一次 CI。
#
# ⚠️ 默认关闭。不设时本步整段跳过，构建流程与产物完全不变。
# ⚠️ 产物带取证打印，**不可交付**。
#
# 用法：TEFLON_NNDUMP=1 ./build-teflon.sh
if [ "${TEFLON_NNDUMP:-0}" = "1" ]; then
	echo "==> [3i/6] 打【诊断】NN core 描述符 dump"
	PATCHER9="$SCRIPT_DIR/patch-etnaviv-nn-desc-dump.py"
	MLNN="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn.c"
	if [ ! -f "$PATCHER9" ]; then
		echo "ERROR: 设了 TEFLON_NNDUMP=1 却缺 $PATCHER9"
		exit 1
	fi
	if [ ! -f "$MLNN" ]; then
		echo "ERROR: 找不到 $MLNN"
		exit 1
	fi
	python3 "$PATCHER9" "$MLNN" || {
		echo "ERROR: NN 描述符 dump 补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出未打补丁的 .so，诊断结论无效）。"
		exit 1
	}
else
	echo "==> [3i/6] NN core 描述符 dump 补丁：默认关闭（设 TEFLON_NNDUMP=1 启用）"
fi

# ── 3j. 【诊断】NN core kernel_pattern 强制开关（改行为，不可交付）─────────
# 背景（报告 28 实测 + 源码复算）：
#   挂死最小单元 = 单个 1×1 CONV_2D（244→244 通道）；122→122 同型**不挂**。
#   复算发现：1×1 conv 的权重一律由 v7 的 write_core_6 写出（= pattern 0x6 的布局），
#   而描述符的 kernel_pattern 却按**输出通道数**分档挑：
#     z≥1024→0x13  z≥512→0x3d  z≥256→0x3e  z≥160→0x6  else→0x3f
#   ⇒ **244 是全模型唯一落进 [160,256) 档的通道数**（24/122/488/976/2048 都不在），
#     也**唯一**挂死 ⇒ 6 个通道数 5 正常 1 挂，完美相关。
# 本补丁把分档选择换成**运行期开关**（默认保持原逻辑，行为完全不变）：
#   TEFLON_KPAT=6  强制 0x6  / 0x7e       / 0x0
#   TEFLON_KPAT=3  强制 0x3f / 0xfffffffe / 0xffffffff
#   不设            走原分档逻辑（默认）
# 目的只有一个：**单变量对照** —— 同一个 122 通道 1×1 conv，只改 pattern 一个字段：
#   仍不挂 ⇒ pattern 不是元凶；立刻挂 ⇒ pattern 0x6 就是元凶。
#
# ⚠️ 默认关闭。不设时本步整段跳过，构建流程与产物完全不变。
# ⚠️ 产物**改行为**，**不可交付**。
#
# 用法：TEFLON_KPAT=1 ./build-teflon.sh
if [ "${TEFLON_KPAT:-0}" = "1" ]; then
	echo "==> [3j/6] 打【诊断】NN core kernel_pattern 强制开关"
	PATCHER10="$SCRIPT_DIR/patch-etnaviv-kernel-pattern.py"
	MLKPAT="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn.c"
	if [ ! -f "$PATCHER10" ]; then
		echo "ERROR: 设了 TEFLON_KPAT=1 却缺 $PATCHER10"
		exit 1
	fi
	if [ ! -f "$MLKPAT" ]; then
		echo "ERROR: 找不到 $MLKPAT"
		exit 1
	fi
	python3 "$PATCHER10" "$MLKPAT" || {
		echo "ERROR: kernel_pattern 补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出未打补丁的 .so，实验结论无效）。"
		exit 1
	}
else
	echo "==> [3j/6] kernel_pattern 强制开关补丁：默认关闭（设 TEFLON_KPAT=1 启用）"
fi

# ── 3k. 【诊断】强制 NN core 走 PARTIAL cache 分支（运行期开关）──────────
# 背景（真机实测 + oca 报告 31 第三轮复核）：
#   挂死最小单元 = 单个 1×1 CONV_2D，244→244 通道；122/488/976/2048 同型都不挂。
#   按 FULL/PARTIAL 判据（nn.c:1188，total = 0x800 + coef + image vs 512KB）分组：
#     122 = 93,184  ⇒ FULL，占 18% ⇒ 不挂
#     244 = 213,376 ⇒ FULL，占 41% ⇒ **挂**      ← 唯一「FULL 且占用大」
#     488 = 540,928 ⇒ PARTIAL         ⇒ 不挂
#   ⇒ 新头号候选：判别量 = **FULL-cache 路径 + 大占用**。
# 本补丁加运行期开关：TEFLON_PARTIAL=1 ⇒ 强制走 PARTIAL 分支（只改描述符，数值不可信）。
#   244 通道的 1×1 conv 若改走 PARTIAL 后**不再挂** ⇒ 该候选成立（根因确证）。
#
# ⚠️ 默认关闭；不设运行期变量时行为与原产物完全一致（可当基线）。
#
# 用法：TEFLON_FPARTIAL=1 ./build-teflon.sh
if [ "${TEFLON_FPARTIAL:-0}" = "1" ]; then
	echo "==> [3k/6] 打【诊断】强制 PARTIAL cache 分支开关"
	PATCHER11="$SCRIPT_DIR/patch-etnaviv-force-partial.py"
	MLFP="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn.c"
	if [ ! -f "$PATCHER11" ]; then
		echo "ERROR: 设了 TEFLON_FPARTIAL=1 却缺 $PATCHER11"
		exit 1
	fi
	if [ ! -f "$MLFP" ]; then
		echo "ERROR: 找不到 $MLFP"
		exit 1
	fi
	python3 "$PATCHER11" "$MLFP" || {
		echo "ERROR: 强制 PARTIAL 补丁应用失败 —— 不要跳过校验"
		exit 1
	}
else
	echo "==> [3k/6] 强制 PARTIAL 分支补丁：默认关闭（设 TEFLON_FPARTIAL=1 启用）"
fi

# ── 3k2. 【修复】per-tensor 权重 int8→uint8 域（+128，运行期门控）──────────────
# 背景（报告 39 + [och-coef] 真机实测）：
#   etna_tensor_zero_point() 的注释写明「signed 张量上传时 +128，
#   **Tensor zero point and weight coefficients have to be adapted**」，
#   但实现只做了两处：输入（etna_ml_subgraph_invoke 的 src[k]+128）与 zero point。
#   **per-tensor 的权重走裸 memcpy，漏了 +128**；
#   per-channel 路径由 requantize_weights() 的 `+ wzp - 0x80` 正确转换。
#
#   实测（C=248，1×1 conv，权重=单位阵）：系数 BO 里是 0x01/0x00（int8 原始值），
#   描述符 coef_zero_point=128 ⇒ 硬件按 字节-128 解释 ⇒ 每个权重变成 -128/-127
#   ⇒ 累加巨额负数 ⇒ ReLU 全钳零。**这正是「改权重为全 1、改偏置为 +20000，
#   NPU 输出逐字节不变」的原因**（三次 hash 都是 0xbb9e90a3c13f4a03）。
#
#   本补丁只在 `is_signed && scales == NULL`（per-tensor + int8）时逐字节 +128，
#   并由环境变量在**运行期**门控，便于同产物 A/B：
#     构建：TEFLON_WDOM_PATCH=1  → 把补丁编进 .so
#     运行：TEFLON_WDOMAIN=1     → 激活转换（不设 = 与上游逐字节一致）
#
# ⚠️ 默认关闭。产物改行为，取证性质，不可直接交付。
if [ "${TEFLON_WDOM_PATCH:-0}" = "1" ]; then
	echo "==> [3k2/6] 打【修复】per-tensor 权重 int8→uint8 域（+128）"
	PATCHER_WDOM="$SCRIPT_DIR/patch-etnaviv-weight-domain.py"
	MLWDOM="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn.c"
	if [ ! -f "$PATCHER_WDOM" ]; then
		echo "ERROR: 设了 TEFLON_WDOM_PATCH=1 却缺 $PATCHER_WDOM"
		exit 1
	fi
	if [ ! -f "$MLWDOM" ]; then
		echo "ERROR: 找不到 $MLWDOM"
		exit 1
	fi
	python3 "$PATCHER_WDOM" "$MLWDOM" || {
		echo "ERROR: 权重域补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出未打补丁的 .so，结论无效）。"
		exit 1
	}
else
	echo "==> [3k2/6] 权重域补丁：默认关闭（设 TEFLON_WDOM_PATCH=1 启用）"
fi

# ── 3k3. 【修复】图输出 NCHW→NHWC 收口（detranspose，运行期门控）────────────
# 背景（报告 39 §5，真机实测）：
#   权重域修好后，NPU 输出与 CPU 按 HWC 读只 1.19% 一致；**按 CHW 重排后
#   76.51% 逐字节一致、其余恰好差 -1、|差|>1 为 0** ⇒ 计算对、布局错。
#
#   源码定位：lower_operations() 尾部本来有 detranspose 收口，条件是
#   `exp_layout != act_layout`；但上面的 etna_ml_copy_layout() 把**图输出**的
#   exp_layout 也继承成了卷积的 NCHW ⇒ 条件恒假 ⇒ detranspose 永不插入。
#   真机日志里 "Adding transpose"/"Adding detranspose" 出现次数均为 0。
#
#   本补丁只把「无消费者的图输出」的 exp_layout 改回 NHWC，让原有逻辑自然触发。
#
#   构建：TEFLON_ODETRANS_PATCH=1 ；运行：TEFLON_ODETRANS=1（不设 = 与上游一致）
#
# ⚠️ 默认关闭。
if [ "${TEFLON_ODETRANS_PATCH:-0}" = "1" ]; then
	echo "==> [3k3/6] 打【修复】图输出 NCHW→NHWC 收口"
	PATCHER_ODT="$SCRIPT_DIR/patch-etnaviv-output-detranspose.py"
	MLODT="mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c"
	if [ ! -f "$PATCHER_ODT" ]; then
		echo "ERROR: 设了 TEFLON_ODETRANS_PATCH=1 却缺 $PATCHER_ODT"
		exit 1
	fi
	if [ ! -f "$MLODT" ]; then
		echo "ERROR: 找不到 $MLODT"
		exit 1
	fi
	python3 "$PATCHER_ODT" "$MLODT" || {
		echo "ERROR: 图输出收口补丁应用失败 —— 不要跳过校验"
		echo "       （会静默产出未打补丁的 .so，结论无效）。"
		exit 1
	}
else
	echo "==> [3k3/6] 图输出收口补丁：默认关闭（设 TEFLON_ODETRANS_PATCH=1 启用）"
fi

# ── 3k4. 【卫生】补齐 etnaviv_ml{,_nn}.c 的显式 std 头文件 ─────────────────
# 这两个文件原本没有 <stdlib.h>（nn.c 连 <stdio.h> 也没有），我们后续补丁首次在
# 其中引入 getenv()/atoi()/fprintf()。当前靠 mesa 头文件链间接包含才能编过 ——
# 属脆弱依赖（上游改一次 include 就退化成 implicit declaration，指针被截断）。
#
# 只加 include，不改任何逻辑。幂等：已存在则跳过。
#
#   构建：TEFLON_STDINC_PATCH=1
#
if [ "${TEFLON_STDINC_PATCH:-0}" = "1" ]; then
	echo "==> [3k4/6] 补【卫生】显式 std 头文件"
	PATCHER_INC="$SCRIPT_DIR/patch-etnaviv-includes.py"
	if [ ! -f "$PATCHER_INC" ]; then
		echo "ERROR: 设了 TEFLON_STDINC_PATCH=1 却缺 $PATCHER_INC"
		exit 1
	fi
	python3 "$PATCHER_INC" \
		"mesa/src/gallium/drivers/etnaviv/etnaviv_ml.c" \
		"mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn.c" || {
		echo "ERROR: include 补齐失败 —— 不要跳过校验。"
		exit 1
	}
else
	echo "==> [3k4/6] 显式 std 头文件：默认关闭（设 TEFLON_STDINC_PATCH=1 启用）"
fi

# ── 3k5. 【探针】NN 描述符 rounding_mode 运行期覆盖（舍入方式实测）──────────
# 背景（报告 42）：两处根因修好后，NPU vs CPU 只剩「恰好差 -1、npu>cpu 恒为 0」
# 的单侧残差 —— 典型舍入方式差异（硬件向 -inf 截断，TFLite 四舍五入）。
#
# rounding_mode 是 2 bit 字段（etnaviv_ml_nn.c:71），上游固定写 0x1 且**没有任何
# 取值含义的注释或枚举** ⇒ 只能真机实测 0/1/2/3。
#
#   构建：TEFLON_ROUNDMODE_PATCH=1 ；运行：TEFLON_ROUNDMODE=0..3（不设 = 上游行为）
#
# ⚠️ 探针性质，默认关闭。它不是修复，是为了拿到「哪个值 = 四舍五入」这一事实。
#
if [ "${TEFLON_ROUNDMODE_PATCH:-0}" = "1" ]; then
	echo "==> [3k5/6] 打【探针】舍入模式运行期覆盖"
	PATCHER_RM="$SCRIPT_DIR/patch-etnaviv-roundmode.py"
	MLRM="mesa/src/gallium/drivers/etnaviv/etnaviv_ml_nn.c"
	if [ ! -f "$PATCHER_RM" ]; then
		echo "ERROR: 设了 TEFLON_ROUNDMODE_PATCH=1 却缺 $PATCHER_RM"
		exit 1
	fi
	if [ ! -f "$MLRM" ]; then
		echo "ERROR: 找不到 $MLRM"
		exit 1
	fi
	python3 "$PATCHER_RM" "$MLRM" || {
		echo "ERROR: 舍入探针补丁应用失败 —— 不要跳过校验。"
		exit 1
	}
else
	echo "==> [3k5/6] 舍入探针：默认关闭（设 TEFLON_ROUNDMODE_PATCH=1 启用）"
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