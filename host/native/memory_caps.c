/* Texas process-local MLX limits. No inference arithmetic or system setting changes. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>

static const size_t texas_cap = (size_t)108 * 1024 * 1024 * 1024;
extern int mlx_set_memory_limit(size_t *, size_t);
extern int mlx_set_wired_limit(size_t *, size_t);
typedef struct { void *ctx; } mlx_device_info;
extern int mlx_device_info_get_size(size_t *, mlx_device_info, const char *);

static int texas_memory(size_t *previous, size_t requested) {
    return mlx_set_memory_limit(previous, requested < texas_cap ? requested : texas_cap);
}
static int texas_wired(size_t *previous, size_t requested) {
    return mlx_set_wired_limit(previous, requested < texas_cap ? requested : texas_cap);
}
static int texas_device_size(size_t *value, mlx_device_info info, const char *key) {
    int rc = mlx_device_info_get_size(value, info, key);
    if (rc == 0 && !strcmp(key, "max_recommended_working_set_size") && *value > texas_cap)
        *value = texas_cap;
    return rc;
}
#define INTERPOSE(replacement, original) \
 __attribute__((used)) static struct { const void *replacement; const void *original; } \
 interpose_##original __attribute__((section("__DATA,__interpose"))) = \
 { (const void *)(uintptr_t)&replacement, (const void *)(uintptr_t)&original }
INTERPOSE(texas_memory, mlx_set_memory_limit);
INTERPOSE(texas_wired, mlx_set_wired_limit);
INTERPOSE(texas_device_size, mlx_device_info_get_size);

__attribute__((constructor)) static void texas_initialize(void) {
    size_t previous;
    if (mlx_set_memory_limit(&previous, texas_cap) || mlx_set_wired_limit(&previous, texas_cap)) {
        fputs("Texas: could not enforce the 108 GiB MLX limits\n", stderr);
        _exit(78);
    }
    fputs("[texas] process-local MLX engine/wired ceilings: 108 GiB\n", stderr);
}
