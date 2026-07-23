/* =========================================================================
 * ds4_tp_discover.c - RDMA peer discovery utility.
 * =========================================================================
 *
 * Standalone binary that discovers Thunderbolt RDMA interfaces on the local
 * machine and writes a JSON peer-config fragment for ds4/ds4-server mesh
 * tensor parallelism.
 *
 * Usage:
 *   ds4-tp-discover --output-file node0.json \
 *     --role coordinator --listen-port 9000 --control-host 192.168.1.100 \
 *     --world-size 4 --rank 0
 *
 *   ds4-tp-discover --output-file node1.json \
 *     --role worker --coordinator-host 192.168.1.100:9000 \
 *     --control-host 192.168.1.101 --world-size 4 --rank 1
 *
 * Each node runs independently; the user assembles the fragments into a
 * complete config file that feeds --tp-config-file.
 *
 * Link-time dependencies: none (dlopen librdma.dylib at runtime).
 */

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <netdb.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------------------
 * Minimal verbs types (enough for discovery; no send/recv needed).
 * --------------------------------------------------------------------- */

enum { IBV_PORT_ACTIVE = 4 };

struct ibv_device;
struct ibv_context;
struct ibv_pd;
struct ibv_cq;
struct ibv_qp;
struct ibv_mr;

typedef struct {
    uint64_t subnet_prefix;
    uint64_t interface_id;
    uint8_t  raw[16];
} ibv_gid;

typedef struct {
    uint8_t  state;
    uint8_t  link_layer;
    uint8_t  reserved[2];
    int      gid_tbl_len;
    uint16_t lid;
    uint16_t sm_lid;
    uint32_t lmc;
    uint8_t  max_vl;
    uint8_t  active_mtu;
    uint8_t  reserved2[32];
} ibv_port_attr;

/* ------------------------------------------------------------------------
 * Function pointer table (subset for discovery only).
 * --------------------------------------------------------------------- */

typedef struct {
    void *handle;
    struct ibv_device **(*get_device_list)(int *);
    void (*free_device_list)(struct ibv_device **);
    const char *(*get_device_name)(struct ibv_device *);
    struct ibv_context *(*open_device)(struct ibv_device *);
    int (*close_device)(struct ibv_context *);
    int (*query_port)(struct ibv_context *, uint8_t, ibv_port_attr *);
    int (*query_gid)(struct ibv_context *, uint8_t, int, ibv_gid *);
} discover_api;

/* ------------------------------------------------------------------------
 * API loading (replicates tp_rdma_load_api pattern from ds4_tp.c).
 * --------------------------------------------------------------------- */

static int discover_load_api(discover_api *api) {
    if (api->handle) return 1;
    void *h = dlopen("/usr/lib/librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
#define DSYM(field, name) \
    do { api->field = (__typeof__(api->field))dlsym(h, name); \
         if (!api->field) { dlclose(h); return 0; } } while (0)
    DSYM(get_device_list,  "ibv_get_device_list");
    DSYM(free_device_list, "ibv_free_device_list");
    DSYM(get_device_name,  "ibv_get_device_name");
    DSYM(open_device,      "ibv_open_device");
    DSYM(close_device,     "ibv_close_device");
    DSYM(query_port,       "ibv_query_port");
    DSYM(query_gid,        "ibv_query_gid");
#undef DSYM
    api->handle = h;
    return 1;
}

/* ------------------------------------------------------------------------
 * GID IPv4 extraction: IPv4-mapped GID = ::ffff:a.b.c.d
 * Raw bytes 0..9 = 0x00, bytes 10..11 = 0xFFFF, bytes 12..15 = IPv4.
 * --------------------------------------------------------------------- */

static int discover_extract_ipv4(const uint8_t raw[16], char *out, size_t outlen) {
    uint64_t hi;
    uint16_t mid, v4tag;
    memcpy(&hi,    &raw[0],  8);
    memcpy(&mid,   &raw[8],  2);
    memcpy(&v4tag, &raw[10], 2);
    if (hi != 0 || mid != 0 || v4tag != 0xffff) return 0;
    struct in_addr a;
    memcpy(&a, &raw[12], 4);
    const char *s = inet_ntoa(a);
    if (!s) return 0;
    snprintf(out, outlen, "%s", s);
    return 1;
}

/* ------------------------------------------------------------------------
 * Discover RDMA interfaces: enumerate devices, find active ports,
 * extract IPv4 addresses from IPv4-mapped GIDs.
 * --------------------------------------------------------------------- */

typedef struct {
    char device[64];
    char ip[64];
    int gid_index;
} discover_iface;

#define DISCOVER_MAX_IFACES 8

static int discover_ifaces(discover_api *api, discover_iface *ifaces,
                            int max_ifaces, const char *want_device) {
    int count = 0;
    int num = 0;
    struct ibv_device **devs = api->get_device_list(&num);
    if (!devs) return 0;

    for (int i = 0; i < num && count < max_ifaces; i++) {
        const char *name = api->get_device_name(devs[i]);
        if (want_device && strcmp(want_device, name) != 0) continue;

        struct ibv_context *ctx = api->open_device(devs[i]);
        if (!ctx) continue;

        ibv_port_attr pa;
        if (api->query_port(ctx, 1, &pa) != 0 ||
            pa.state != IBV_PORT_ACTIVE) {
            api->close_device(ctx);
            continue;
        }

        /* Scan GID table for IPv4-mapped entries. */
        int found = 0;
        for (int g = 0; g < pa.gid_tbl_len && !found; g++) {
            ibv_gid tmp;
            if (api->query_gid(ctx, 1, g, &tmp) != 0) continue;
            char ip[64];
            if (discover_extract_ipv4(tmp.raw, ip, sizeof(ip))) {
                snprintf(ifaces[count].device, sizeof(ifaces[count].device),
                         "%s", name);
                snprintf(ifaces[count].ip, sizeof(ifaces[count].ip),
                         "%s", ip);
                ifaces[count].gid_index = g;
                count++;
                found = 1;
            }
        }
        api->close_device(ctx);
    }
    api->free_device_list(devs);
    return count;
}

/* ------------------------------------------------------------------------
 * JSON output helpers.
 * --------------------------------------------------------------------- */

static void json_string(FILE *fp, const char *s) {
    fprintf(fp, "\"");
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') fprintf(fp, "\\%c", *s);
        else fprintf(fp, "%c", *s);
    }
    fprintf(fp, "\"");
}

static void json_int(FILE *fp, int v) {
    fprintf(fp, "%d", v);
}

/* ------------------------------------------------------------------------
 * Main: parse options, discover, write JSON.
 * --------------------------------------------------------------------- */

static void usage(FILE *fp) {
    fprintf(fp,
        "ds4-tp-discover - RDMA peer discovery for tensor-parallel mesh\n"
        "Usage: ds4-tp-discover [options]\n"
        "  --output-file <path>       Write JSON fragment to this file.\n"
        "  --role coordinator|worker   This node's role.\n"
        "  --listen-port <N>           Coordinator listen port.\n"
        "  --control-host <ip>         This node's control-plane IP.\n"
        "  --world-size <2..6>         Total ranks in the mesh.\n"
        "  --rank <0..5>               This node's rank.\n"
        "  --coordinator-host <ip>:<port>  Worker dial address.\n"
        "  --rdma-device <name>        Restrict to a specific verbs device.\n"
        "  -h, --help                  This message.\n");
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-tp-discover: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-tp-discover: invalid %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

int main(int argc, char **argv) {
    const char *output_file = NULL;
    const char *role = NULL;
    const char *control_host = NULL;
    const char *coordinator_host = NULL;
    const char *rdma_device = NULL;
    int listen_port = 0;
    int world_size = 0;
    int rank = -1;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout); return 0;
        } else if (!strcmp(arg, "--output-file")) {
            output_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--role")) {
            role = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--listen-port")) {
            listen_port = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--control-host")) {
            control_host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--world-size")) {
            world_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--rank")) {
            rank = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--coordinator-host")) {
            coordinator_host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--rdma-device")) {
            rdma_device = need_arg(&i, argc, argv, arg);
        } else {
            fprintf(stderr, "ds4-tp-discover: unknown option: %s\n", arg);
            usage(stderr); return 2;
        }
    }

    /* Load verbs API. */
    discover_api api = {0};
    if (!discover_load_api(&api)) {
        fprintf(stderr, "ds4-tp-discover: cannot load librdma.dylib "
                "(no Thunderbolt RDMA support on this machine)\n");
        return 1;
    }

    /* Discover interfaces. */
    discover_iface ifaces[DISCOVER_MAX_IFACES];
    int n = discover_ifaces(&api, ifaces, DISCOVER_MAX_IFACES, rdma_device);
    if (n == 0) {
        fprintf(stderr, "ds4-tp-discover: no active RDMA interfaces found\n");
        if (api.handle) dlclose(api.handle);
        return 1;
    }

    /* Write JSON. */
    FILE *fp = output_file ? fopen(output_file, "w") : stdout;
    if (!fp) {
        fprintf(stderr, "ds4-tp-discover: cannot open %s: %s\n",
                output_file, strerror(errno));
        if (api.handle) dlclose(api.handle);
        return 1;
    }

    fprintf(fp, "{\n");
    if (rank >= 0) {
        fprintf(fp, "  ");
        json_string(fp, "rank");
        fprintf(fp, ": ");
        json_int(fp, rank);
        fprintf(fp, ",\n");
    }
    if (role) {
        fprintf(fp, "  ");
        json_string(fp, "role");
        fprintf(fp, ": ");
        json_string(fp, role);
        fprintf(fp, ",\n");
    }
    if (control_host) {
        fprintf(fp, "  ");
        json_string(fp, "control_host");
        fprintf(fp, ": ");
        json_string(fp, control_host);
        fprintf(fp, ",\n");
    }
    if (listen_port > 0) {
        fprintf(fp, "  ");
        json_string(fp, "control_port");
        fprintf(fp, ": ");
        json_int(fp, listen_port);
        fprintf(fp, ",\n");
    }
    if (coordinator_host) {
        fprintf(fp, "  ");
        json_string(fp, "coordinator_host");
        fprintf(fp, ": ");
        json_string(fp, coordinator_host);
        fprintf(fp, ",\n");
    }
    if (world_size > 0) {
        fprintf(fp, "  ");
        json_string(fp, "world_size");
        fprintf(fp, ": ");
        json_int(fp, world_size);
        fprintf(fp, ",\n");
    }
    fprintf(fp, "  ");
    json_string(fp, "rdma_interfaces");
    fprintf(fp, ": [\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "    {");
        json_string(fp, "device");
        fprintf(fp, ": ");
        json_string(fp, ifaces[i].device);
        fprintf(fp, ", ");
        json_string(fp, "ip");
        fprintf(fp, ": ");
        json_string(fp, ifaces[i].ip);
        fprintf(fp, ", ");
        json_string(fp, "gid_index");
        fprintf(fp, ": ");
        json_int(fp, ifaces[i].gid_index);
        fprintf(fp, "}%s\n", i + 1 < n ? "," : "");
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");

    if (output_file) fclose(fp);
    if (api.handle) dlclose(api.handle);

    fprintf(stderr, "ds4-tp-discover: found %d active RDMA interface(s)\n", n);
    return 0;
}
