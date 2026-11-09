#include <stdint.h>
#include <sys/statvfs.h>
int sushi_available_disk_bytes(const char *path, uint64_t *out) {
    struct statvfs s;
    if (statvfs(path, &s) != 0) return -1;
    uint64_t block = s.f_frsize ? s.f_frsize : s.f_bsize;
    *out = block && s.f_bavail > UINT64_MAX / block ? UINT64_MAX : s.f_bavail * block;
    return 0;
}
