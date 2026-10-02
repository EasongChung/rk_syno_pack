// SPDX-License-Identifier: GPL-2.0
/*
 * CMA-backed dumb buffer DRM device for the OEC RK3566 box.
 *
 * The board has no display pipeline wired in DTS, so the Rockchip DRM
 * master (rockchip,display-subsystem) never probes and no DRM device with
 * drm_driver.dumb_create exists. Userspace librga (2.x, used by the
 * jellyfin rockchip image) allocates its video buffers through
 * DRM_IOCTL_MODE_CREATE_DUMB and imports the resulting prime fd; the only
 * cards on this machine are panfrost and RKNPU, neither of which
 * implements that ioctl, so librga gets ENOSYS and dereferences a NULL
 * pointer (segfault inside av_hwframe_get_buffer).
 *
 * This driver registers a standalone DRM device that implements only the
 * dumb-buffer entry points on top of the kernel CMA area. It is not a
 * render engine and exposes no modesetting, no crtc and no prime export
 * beyond what drm_gem_cma provides; librga only needs a device whose dumb
 * create/map/ destroy succeed so it can obtain and mmap a buffer.
 */

#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/of_device.h>
#include <linux/dma-mapping.h>
#include <linux/cma.h>

#include <drm/drm_device.h>
#include <drm/drm_drv.h>
#include <drm/drm_file.h>
#include <drm/drm_gem.h>
#include <drm/drm_gem_cma_helper.h>
#include <drm/drm_ioctl.h>
#include <drm/drm_print.h>
#include <drm/drm_prime.h>

#define DRIVER_NAME	"rkrga-dumb"
#define DRIVER_DESC	"CMA dumb buffer device for librga"
#define DRIVER_DATE	"20261001"

static const struct file_operations rga_dumb_fops = {
	.owner		= THIS_MODULE,
	.open		= drm_open,
	.release	= drm_release,
	.unlocked_ioctl	= drm_ioctl,
	.compat_ioctl	= drm_compat_ioctl,
	.poll		= drm_poll,
	.read		= drm_read,
	.llseek		= noop_llseek,
	.mmap		= drm_gem_cma_mmap,
};

static struct drm_driver rga_dumb_driver = {
	.driver_features	= DRIVER_GEM | DRIVER_MODESET | DRIVER_RENDER,
	.fops			= &rga_dumb_fops,
	.name			= DRIVER_NAME,
	.desc			= DRIVER_DESC,
	.date			= DRIVER_DATE,
	.major			= 1,
	.minor			= 0,

	.gem_create_object	= drm_gem_cma_create_object_default_funcs,
	.dumb_create		= drm_gem_cma_dumb_create,
	.dumb_map_offset	= drm_gem_dumb_map_offset,
	.prime_handle_to_fd	= drm_gem_prime_handle_to_fd,
	.prime_fd_to_handle	= drm_gem_prime_fd_to_handle,
	.gem_prime_get_sg_table	= drm_gem_cma_prime_get_sg_table,
	.gem_prime_import_sg_table = drm_gem_cma_prime_import_sg_table,
	.gem_prime_mmap		= drm_gem_prime_mmap,
	.gem_prime_vmap		= drm_gem_cma_prime_vmap,
	.gem_prime_vunmap	= drm_gem_cma_prime_vunmap,
	.gem_free_object_unlocked = drm_gem_cma_free_object,
};

static int rga_dumb_probe(struct platform_device *pdev)
{
	struct drm_device *drm;
	int ret;

	drm = drm_dev_alloc(&rga_dumb_driver, &pdev->dev);
	if (IS_ERR(drm))
		return PTR_ERR(drm);

	ret = drm_dev_register(drm, 0);
	if (ret) {
		drm_dev_put(drm);
		return ret;
	}

	platform_set_drvdata(pdev, drm);
	return 0;
}

static int rga_dumb_remove(struct platform_device *pdev)
{
	struct drm_device *drm = platform_get_drvdata(pdev);

	drm_dev_unregister(drm);
	drm_dev_put(drm);
	return 0;
}

static const struct of_device_id rga_dumb_of_match[] = {
	{ .compatible = "rockchip,oec-rga-dumb" },
	{ /* sentinel */ },
};
MODULE_DEVICE_TABLE(of, rga_dumb_of_match);

static struct platform_driver rga_dumb_platform_driver = {
	.probe	= rga_dumb_probe,
	.remove	= rga_dumb_remove,
	.driver	= {
		.name		= DRIVER_NAME,
		.of_match_table	= rga_dumb_of_match,
	},
};

module_platform_driver(rga_dumb_platform_driver);

MODULE_DESCRIPTION(DRIVER_DESC);
MODULE_LICENSE("GPL v2");
