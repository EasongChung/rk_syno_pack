// SPDX-License-Identifier: GPL-2.0+
/*
 * Generate a Synology DS423+ style serial number and store it in
 * Rockchip vendor storage.
 */

#include <common.h>
#include <command.h>
#include <asm/arch/vendor.h>

#define SYNO_DS423_SN_PREFIX	"22A0VKR"

static char syno_random_letter(void)
{
	static const char letters[] = "ABCDEFGHJKLMNPQRSTUVWXYZ";

	return letters[rand() % (sizeof(letters) - 1)];
}

static char syno_random_value(void)
{
	static const char values[] = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

	return values[rand() % (sizeof(values) - 1)];
}

static void syno_generate_ds423_sn(char *serial, size_t len)
{
	snprintf(serial, len, "%s%c%c%c%c%c%c",
		 SYNO_DS423_SN_PREFIX,
		 syno_random_letter(),
		 syno_random_value(),
		 syno_random_value(),
		 syno_random_value(),
		 syno_random_value(),
		 syno_random_letter());
}

static int do_syno_sn(cmd_tbl_t *cmdtp, int flag, int argc, char *const argv[])
{
	char serial[sizeof("22A0VKRA1234B")];
	int ret;

	if (argc > 2)
		return CMD_RET_USAGE;

	if (argc == 2 && strcmp(argv[1], "print"))
		return CMD_RET_USAGE;

	syno_generate_ds423_sn(serial, sizeof(serial));

	if (argc == 2) {
		printf("%s\n", serial);
		return CMD_RET_SUCCESS;
	}

	ret = vendor_storage_write(SN_ID, serial, sizeof(serial));
	if (ret < 0) {
		printf("Failed to write Synology SN to vendor storage: %d\n",
		       ret);
		return CMD_RET_FAILURE;
	}

	env_set("serial#", serial);
	printf("Synology SN: %s\n", serial);

	return CMD_RET_SUCCESS;
}

U_BOOT_CMD(
	synosn, 2, 0, do_syno_sn,
	"generate DS423+ Synology serial and write it to vendor storage",
	"[print]\n"
	"    - generate and write SN_ID; use 'print' to only display one"
);
