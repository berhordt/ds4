/* Tensor-parallel transport and lockstep protocol.  See ds4_tp.h and
 * misc/METAL_TENSOR_PARALLELISM.md for the design.
 *
 * Wire notes: both ranks are identical Apple Silicon machines by
 * definition, so the wire format is host little-endian; the hello magic
 * doubles as a byte-order check.  The control socket is a plain blocking
 * TCP stream carrying framed commands.  Gate traffic goes over RDMA
 * (Thunderbolt UC queue pair, two-sided send/recv — see the driver quirks
 * note at ds4_tp_rdma) or over a dedicated full-duplex TCP socket at 16KB
 * per direction as the fallback. */

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <stdarg.h>
#include <sys/uio.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ds4_tp.h"

#if defined(__APPLE__) && defined(__has_include)
#if __has_include(<infiniband/verbs.h>)
#include <infiniband/verbs.h>
#include <dlfcn.h>
#define DS4_TP_HAVE_VERBS 1
#endif
#endif

#define DS4_TP_MAGIC UINT32_C(0x44533454) /* "DS4T" */
#define DS4_TP_BATCH_MAGIC UINT32_C(0x44533442) /* "DS4B" */
#define DS4_TP_PROTOCOL_VERSION 8u  /* mesh: hello world/rank, RDMA_MODE */

/* Default gate timeout is generous: the first gate after a sync waits for
 * the peer's whole (possibly cold page cache) prefill. */
#define DS4_TP_DEFAULT_TIMEOUT_SEC 300

typedef struct {
    uint32_t magic;
    uint32_t type;
    uint32_t bytes;
} ds4_tp_frame_header;

typedef struct {
    uint32_t magic;      /* also detects byte-order mismatch */
    uint32_t version;
    uint32_t role;
    uint32_t rdma_ok;    /* this side has a usable verbs device */
    uint64_t gguf_bytes;
    uint32_t model_id;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t quant_bits;
    uint32_t ctx_size;
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
    uint32_t world;             /* mesh world size (2 for the classic pair) */
    uint32_t rank;              /* this node's rank in the mesh */
    uint32_t pad;
} ds4_tp_hello_fixed;

typedef struct {
    uint64_t slab_base;
    uint32_t rkey;
    uint32_t qpn;
    uint32_t psn;
    uint32_t mtu;
    uint16_t lid;
    uint8_t gid[16];
    uint8_t link_layer;
} ds4_tp_rdma_info;

/* TCP gate frames carry a small header so a desynchronized pair fails loudly
 * instead of silently mixing partials. */
typedef struct {
    uint32_t magic;
    uint16_t layer;
    uint16_t gate;
    uint64_t seq;
} ds4_tp_gate_header;

#ifdef DS4_TP_HAVE_VERBS
/* librdma is loaded at runtime so builds and machines without the RDMA
 * stack (or with it disabled) fall back to TCP with no link-time cost.
 * ibv_post_send()/ibv_poll_cq() are header inlines over context->ops, so
 * only the setup entry points need dlsym. */
typedef struct {
    void *handle;
    struct ibv_device **(*get_device_list)(int *);
    void (*free_device_list)(struct ibv_device **);
    const char *(*get_device_name)(struct ibv_device *);
    struct ibv_context *(*open_device)(struct ibv_device *);
    int (*close_device)(struct ibv_context *);
    int (*query_device)(struct ibv_context *, struct ibv_device_attr *);
    int (*query_port)(struct ibv_context *, uint8_t, struct ibv_port_attr *);
    int (*query_gid)(struct ibv_context *, uint8_t, int, union ibv_gid *);
    struct ibv_pd *(*alloc_pd)(struct ibv_context *);
    int (*dealloc_pd)(struct ibv_pd *);
    struct ibv_mr *(*reg_mr)(struct ibv_pd *, void *, size_t, int);
    int (*dereg_mr)(struct ibv_mr *);
    struct ibv_cq *(*create_cq)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int);
    int (*destroy_cq)(struct ibv_cq *);
    struct ibv_qp *(*create_qp)(struct ibv_pd *, struct ibv_qp_init_attr *);
    int (*destroy_qp)(struct ibv_qp *);
    int (*modify_qp)(struct ibv_qp *, struct ibv_qp_attr *, int);
} ds4_tp_verbs_api;

/* AppleThunderboltRDMA quirks (validated with scratchpad probes,
 * 2026-07-06): only UC queue pairs exist (RC/UD: ENOTSUP); RDMA WRITE work
 * requests are accepted but never execute, so the data plane is two-sided
 * SEND/RECV like Apple's own JACCL; messages above 16KB are not delivered;
 * RTR requires GRH addressing with the IPv4-mapped GID that appears only
 * once the Thunderbolt member interface has an IPv4 address of its own.
 * UC delivery is in-order and the gate sequence is globally deterministic
 * (86 gates per token, fixed order). After any initial bulk prefill, decode
 * keeps a receive window posted by sequence number: recv for seq s lands in
 * the slab in-slot (s-1) % slots and its completion IS the arrival signal. */
#define DS4_TP_RDMA_MAX_MSG 16384
#define DS4_TP_RDMA_RECV_WINDOW 16
#define DS4_TP_RDMA_BULK_SLOTS 64
#define DS4_TP_RDMA_BULK_WR_TAG (UINT64_C(1) << 63)

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    struct ibv_mr *mr;
    struct ibv_port_attr port;
    union ibv_gid gid;
    int gid_index;
    uint32_t max_inline;
    ds4_tp_rdma_info peer;
    uint32_t send_outstanding;  /* signaled sends not yet reaped */
    uint64_t recv_done;         /* highest gate seq whose recv completed */
    uint64_t last_gate_seq;     /* last real decode receive consumed */
    bool recv_window_active;    /* decode recvs are queued ahead */
    pthread_mutex_t post_lock;
} ds4_tp_rdma_link;
#endif

struct ds4_tp {
    ds4_tp_options opt;
    int rank;                   /* 0 leader, 1..world-1 workers */
    int world;                  /* mesh size (2 for the classic pair) */
    ds4_tp_topology topo;       /* resolved mesh descriptor (world==0: none) */
    /* Control sockets, indexed by peer rank.  The leader holds one fd per
     * worker (control_fd[m], m=1..world-1); a worker holds control_fd[0]
     * to the leader. */
    int control_fd[DS4_TP_MAX_WORLD];
    /* Full-mesh data sockets, indexed by peer rank: data_fd[m] is the
     * gate/verify link to rank m (m != rank). */
    int data_fd[DS4_TP_MAX_WORLD];
    bool rdma_active;
    uint32_t peer_rdma_ok[DS4_TP_MAX_WORLD]; /* workers' rdma_ok from hello */
    uint32_t peer_ctx;
    uint32_t n_layer;
    uint32_t n_embd;
    uint64_t vec_bytes;
    uint32_t n_slots;
    /* Decode gate schedule (see ds4_tp_identity). */
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
    uint8_t *slab;
    uint64_t slab_bytes;
    /* Slab regions, see ds4_tp.h layout comment.  in_peer_bytes is the
     * per-peer in stride ((world-1) in vectors per slot). */
    uint64_t out_off;
    uint64_t in_off;
    uint64_t in_peer_bytes;
    uint64_t combined_off;      /* canonical sum for world>2 (CPU exchange) */
    uint64_t in_flags_off;
    uint64_t token_off;
    uint64_t out_flags_off;     /* local staging for RDMA flag writes */
    uint64_t gpu_flags_off;     /* GPU-written gate-ready flags (u32/slot) */
    uint64_t batch_out_off;     /* [layer][row] verify-block local partials */
    uint64_t batch_in_off;      /* [layer][row] verify-block peer partials */
    uint64_t batch_combined_off;/* canonical batch sum for world>2 */
    uint64_t timeout_sec;
    atomic_bool failed;
#ifdef DS4_TP_HAVE_VERBS
    ds4_tp_verbs_api rdma_api;          /* verbs entry points, loaded once */
    ds4_tp_rdma_link rdma[DS4_TP_MAX_WORLD];   /* one QP per peer link */
#endif
};

/* ------------------------------------------------------------------------
 * Small socket helpers (same conventions as ds4_distributed.c).
 * --------------------------------------------------------------------- */

static double tp_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void tp_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || !errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

/* ---------------------------------------------------------------------
 * Mesh topology descriptor.
 * ----------------------------------------------------------------- */

int ds4_tp_topology_load(const char *path, ds4_tp_topology *topo,
                         char *err, size_t errlen) {
    memset(topo, 0, sizeof(*topo));
    FILE *fp = fopen(path, "r");
    if (!fp) {
        tp_set_err(err, errlen, "tp topology: %s: %s", path, strerror(errno));
        return 0;
    }
    char line[512];
    int world = -1;
    bool seen[DS4_TP_MAX_WORLD] = {false};
    int lineno = 0;
    while (fgets(line, sizeof(line), fp)) {
        lineno++;
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (!*s || *s == '#') continue;
        char *tok = strtok(s, " \t\r\n");
        if (!tok) continue;
        if (!strcmp(tok, "world")) {
            tok = strtok(NULL, " \t\r\n");
            if (!tok) goto malformed;
            world = atoi(tok);
            if (world < 2 || world > DS4_TP_MAX_WORLD) {
                tp_set_err(err, errlen,
                           "tp topology: %s:%d: world must be 2..%d",
                           path, lineno, DS4_TP_MAX_WORLD);
                goto fail;
            }
            topo->world = world;
        } else if (!strcmp(tok, "node")) {
            tok = strtok(NULL, " \t\r\n");
            if (!tok) goto malformed;
            int node = atoi(tok);
            if (world < 0 || node < 0 || node >= world) {
                tp_set_err(err, errlen, "tp topology: %s:%d: bad node %s",
                           path, lineno, tok);
                goto fail;
            }
            if (seen[node]) {
                tp_set_err(err, errlen,
                           "tp topology: %s:%d: duplicate node %d",
                           path, lineno, node);
                goto fail;
            }
            seen[node] = true;
            for (int i = 0; i < DS4_TP_LINKS(world); i++) {
                char *host = strtok(NULL, " \t\r\n");
                char *port = strtok(NULL, " \t\r\n");
                if (!host || !port) goto malformed;
                char *end = NULL;
                errno = 0;
                long p = strtol(port, &end, 10);
                if (errno != 0 || end == port || *end || p <= 0 || p > 65535) {
                    tp_set_err(err, errlen,
                               "tp topology: %s:%d: bad link port %s",
                               path, lineno, port);
                    goto fail;
                }
                topo->node[node].host[i] = strdup(host);
                topo->node[node].port[i] = (int)p;
            }
        } else {
            goto malformed;
        }
    }
    if (world < 0) goto malformed;
    for (int i = 0; i < world; i++) {
        if (!seen[i]) {
            tp_set_err(err, errlen, "tp topology: %s: missing node %d",
                       path, i);
            goto fail;
        }
    }
    fclose(fp);
    return 1;
malformed:
    tp_set_err(err, errlen, "tp topology: %s:%d: malformed line", path, lineno);
fail:
    fclose(fp);
    ds4_tp_topology_free(topo);
    return 0;
}

void ds4_tp_topology_free(ds4_tp_topology *topo) {
    if (!topo) return;
    for (int n = 0; n < DS4_TP_MAX_WORLD; n++) {
        for (int i = 0; i < DS4_TP_MAX_WORLD; i++) {
            free(topo->node[n].host[i]);
            topo->node[n].host[i] = NULL;
        }
    }
    memset(topo, 0, sizeof(*topo));
}

static int tp_write_full(int fd, const void *buf, size_t len) {
    const char *p = buf;
    while (len) {
#ifdef MSG_NOSIGNAL
        ssize_t w = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t w = send(fd, p, len, 0);
#endif
        if (w < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (w == 0) return 0;
        p += w;
        len -= (size_t)w;
    }
    return 1;
}

static int tp_read_full(int fd, void *buf, size_t len) {
    char *p = buf;
    while (len) {
        ssize_t r = read(fd, p, len);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (r == 0) return 0;
        p += r;
        len -= (size_t)r;
    }
    return 1;
}

static void tp_socket_tune(int fd) {
    int one = 1;
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    /* Gate exchanges are latency-critical 16KB messages; large socket
     * buffers only matter for the TCP fallback's pipelining. */
    int sz = 4 * 1024 * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
}

#ifdef DS4_TP_HAVE_VERBS
/* UC queue pairs do not report a dead remote reliably.  The control socket
 * does, so sample it while polling an RDMA completion and abort before the
 * Metal command-buffer watchdog fires. */
static int tp_peer_closed(const ds4_tp *tp, int peer) {
    char byte;
    const int fd = tp->data_fd[peer];
    if (fd < 0) return 0;
    const ssize_t n = recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0) return 1;
    if (n > 0) return 0;
    return errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR;
}
#endif

static int tp_listen(const char *host, int port, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    int rc = getaddrinfo(host && host[0] ? host : NULL, portbuf, &hints, &res);
    if (rc != 0) {
        tp_set_err(err, errlen, "tp listen resolve %s:%d: %s", host, port, gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, 2) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) tp_set_err(err, errlen, "tp listen %s:%d: %s", host, port, strerror(errno));
    return fd;
}

static int tp_dial(const char *host, int port, double timeout_sec, char *err, size_t errlen) {
    char portbuf[16];
    snprintf(portbuf, sizeof(portbuf), "%d", port);
    double deadline = tp_now_sec() + timeout_sec;
    int last_errno = 0;
    uint32_t attempts = 0;
    do {
        struct addrinfo hints = {0}, *res = NULL;
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        int gai = getaddrinfo(host, portbuf, &hints, &res);
        if (gai == 0) {
            for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
                int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
                if (fd < 0) continue;
                if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
                    freeaddrinfo(res);
                    return fd;
                }
                last_errno = errno;
                close(fd);
            }
            freeaddrinfo(res);
        }
        /* Retrying is normal while the peer loads its model; still say why
         * every ~10s so a wrong address or a policy block is visible. */
        if (attempts++ % 50 == 0) {
            fprintf(stderr, "ds4-tp: connecting to %s:%d ... (%s)\n", host, port,
                    gai != 0 ? gai_strerror(gai) :
                    last_errno ? strerror(last_errno) : "no address worked");
        }
        usleep(200 * 1000);
    } while (tp_now_sec() < deadline);
    tp_set_err(err, errlen, "tp connect %s:%d: %s", host, port,
               last_errno ? strerror(last_errno) : "unreachable");
    return -1;
}

static int tp_send_frame(int fd, uint32_t type, const void *payload, uint32_t bytes) {
    ds4_tp_frame_header h = { DS4_TP_MAGIC, type, bytes };
    if (!tp_write_full(fd, &h, sizeof(h))) return 0;
    if (bytes && !tp_write_full(fd, payload, bytes)) return 0;
    return 1;
}

static int tp_read_frame_header(int fd, uint32_t *type, uint32_t *bytes) {
    ds4_tp_frame_header h;
    if (!tp_read_full(fd, &h, sizeof(h))) return 0;
    if (h.magic != DS4_TP_MAGIC) return 0;
    *type = h.type;
    *bytes = h.bytes;
    return 1;
}

/* ------------------------------------------------------------------------
 * Options and CLI.
 * --------------------------------------------------------------------- */

bool ds4_tp_enabled(const ds4_tp_options *opt) {
    return opt && opt->role != DS4_TP_NONE;
}

void ds4_tp_usage(FILE *fp) {
    fprintf(fp,
        "Tensor parallelism (two identical machines):\n"
        "  --tensor-parallel           Use --role/--listen/--coordinator for a 50/50 TP pair.\n"
        "  --transport <auto|rdma|tcp> Gate transport (default auto).\n"
        "  --rdma-device <name>        Select a verbs device such as rdma_en1.\n"
        "  --rdma-gid-index <n>        Select the local verbs GID index.\n"
        "  --tensor-parallel-token-prefill\n"
        "                              GLM diagnostic: prefill one token at a time.\n"
        "  --debug-hash <n>            Cross-check hidden state every n tokens.\n"
        "Mesh (fully connected N-node TP over RDMA):\n"
        "  --tp-topology <file>        Mesh descriptor: world size + each node's 3 link\n"
        "                              addresses. Replaces --listen/--coordinator.\n"
        "  --tp-rank <n>               This node's rank (0 leader, 1..world-1 workers).\n");
}

int ds4_tp_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_tp_options *opt,
        char *err,
        size_t errlen)
{
    int i = *index;
    if (!strcmp(arg, "--tensor-parallel")) {
        opt->requested = true;
    } else if (!strcmp(arg, "--transport")) {
        if (i + 1 >= argc) goto missing;
        const char *v = argv[++i];
        if (!strcmp(v, "auto")) opt->transport = DS4_TP_TRANSPORT_AUTO;
        else if (!strcmp(v, "rdma")) opt->transport = DS4_TP_TRANSPORT_RDMA;
        else if (!strcmp(v, "tcp")) opt->transport = DS4_TP_TRANSPORT_TCP;
        else {
            tp_set_err(err, errlen, "invalid %s value: %s", arg, v);
            return DS4_TP_CLI_ERROR;
        }
    } else if (!strcmp(arg, "--rdma-device")) {
        if (i + 1 >= argc) goto missing;
        opt->rdma_device = argv[++i];
    } else if (!strcmp(arg, "--rdma-gid-index")) {
        if (i + 1 >= argc) goto missing;
        char *end = NULL;
        errno = 0;
        long value = strtol(argv[++i], &end, 10);
        if (errno != 0 || !end || *end != '\0' || value < 0 || value > INT_MAX) {
            tp_set_err(err, errlen, "invalid --rdma-gid-index %s", argv[i]);
            return DS4_TP_CLI_ERROR;
        }
        opt->rdma_gid_index = (int)value;
        opt->rdma_gid_index_set = true;
    } else if (!strcmp(arg, "--tensor-parallel-token-prefill")) {
        opt->glm_token_prefill = true;
    } else if (!strcmp(arg, "--debug-hash")) {
        if (i + 1 >= argc) goto missing;
        opt->debug_hash = atoi(argv[++i]);
    } else if (!strcmp(arg, "--tp-topology")) {
        if (i + 1 >= argc) goto missing;
        opt->topology_path = argv[++i];
    } else if (!strcmp(arg, "--tp-rank")) {
        if (i + 1 >= argc) goto missing;
        char *end = NULL;
        errno = 0;
        long value = strtol(argv[++i], &end, 10);
        if (errno != 0 || !end || *end != '\0' || value < 0 ||
            value >= DS4_TP_MAX_WORLD) {
            tp_set_err(err, errlen,
                       "invalid --tp-rank %s (0..%d)", argv[i], DS4_TP_MAX_WORLD - 1);
            return DS4_TP_CLI_ERROR;
        }
        opt->rank = (int)value;
        opt->rank_set = true;
    } else {
        return DS4_TP_CLI_NOT_MATCHED;
    }
    *index = i;
    return DS4_TP_CLI_MATCHED;
missing:
    tp_set_err(err, errlen, "%s requires an argument", arg);
    return DS4_TP_CLI_ERROR;
}

int ds4_tp_adopt_distributed_options(
        ds4_tp_options *tp,
        ds4_distributed_options *dist,
        char *err,
        size_t errlen)
{
    if (!tp || !dist || !tp->requested) return 1;
    if (tp->role != DS4_TP_NONE) {
        tp_set_err(err, errlen,
                   "--tensor-parallel selects its role through --role");
        return 0;
    }
    if (dist->role == DS4_DISTRIBUTED_NONE) {
        tp_set_err(err, errlen,
                   "--tensor-parallel requires --role coordinator or --role worker");
        return 0;
    }
    if (dist->layers.set) {
        tp_set_err(err, errlen,
                   "tensor parallelism does not use distributed layer slices; omit --layers");
        return 0;
    }
    if (dist->prefill_chunk || dist->prefill_window || dist->activation_bits ||
        dist->replay_check || dist->debug) {
        tp_set_err(err, errlen,
                   "--dist-* and distributed debug options cannot be used with --tensor-parallel");
        return 0;
    }

    if (dist->role == DS4_DISTRIBUTED_COORDINATOR) {
        if (!tp->topology_path) {
            if (!dist->listen_host || dist->listen_port <= 0) {
                tp_set_err(err, errlen,
                           "--role coordinator --tensor-parallel requires --listen HOST PORT "
                           "or --tp-topology FILE");
                return 0;
            }
            if (dist->coordinator_host || dist->coordinator_port) {
                tp_set_err(err, errlen,
                           "--role coordinator must not use --coordinator");
                return 0;
            }
            tp->listen_host = dist->listen_host;
            tp->listen_port = dist->listen_port;
        }
        tp->role = DS4_TP_LEADER;
        if (tp->topology_path) tp->rank = 0;
    } else if (dist->role == DS4_DISTRIBUTED_WORKER) {
        if (!tp->topology_path) {
            if (!dist->coordinator_host || dist->coordinator_port <= 0) {
                tp_set_err(err, errlen,
                           "--role worker --tensor-parallel requires --coordinator HOST PORT "
                           "or --tp-topology FILE --tp-rank N");
                return 0;
            }
            if (dist->listen_host || dist->listen_port) {
                tp_set_err(err, errlen,
                           "--role worker --tensor-parallel must not use --listen");
                return 0;
            }
            tp->leader_host = dist->coordinator_host;
            tp->leader_port = dist->coordinator_port;
        } else if (!tp->rank_set) {
            tp_set_err(err, errlen,
                       "--role worker --tensor-parallel --tp-topology requires --tp-rank N");
            return 0;
        }
        tp->role = DS4_TP_WORKER;
    } else {
        tp_set_err(err, errlen, "invalid tensor-parallel role");
        return 0;
    }

    memset(dist, 0, sizeof(*dist));
    return 1;
}

int ds4_tp_validate_engine_options(
        const ds4_engine_options *opt,
        char *err,
        size_t errlen)
{
    if (!ds4_tp_enabled(&opt->tp)) {
        if (opt->tp.requested || opt->tp.transport != DS4_TP_TRANSPORT_AUTO ||
            opt->tp.rdma_device || opt->tp.rdma_gid_index_set ||
            opt->tp.glm_token_prefill || opt->tp.debug_hash != 0) {
            tp_set_err(err, errlen,
                       "tensor-parallel options require --tensor-parallel and --role");
            return 0;
        }
        return 1;
    }
    if (opt->backend != DS4_BACKEND_METAL) {
        tp_set_err(err, errlen, "tensor parallelism requires the Metal backend");
        return 0;
    }
    if (opt->ssd_streaming) {
        tp_set_err(err, errlen, "tensor parallelism requires resident weights (no --ssd-streaming)");
        return 0;
    }
    if (opt->distributed.role != DS4_DISTRIBUTED_NONE) {
        tp_set_err(err, errlen, "tensor parallelism and --role distributed modes are exclusive");
        return 0;
    }
    /* Speculative drafting (DSpark/MTP) is allowed on the leader: the
     * verify block is mirrored to the worker via DS4_TP_FRAME_VERIFY and
     * the legacy MTP path falls back to per-token decode under TP. */
    if (opt->load_slice) {
        tp_set_err(err, errlen, "tensor parallelism does not use distributed layer slices");
        return 0;
    }
    /* Mesh: validate the rank against the topology world early. */
    if (opt->tp.topology_path) {
        ds4_tp_topology topo;
        if (!ds4_tp_topology_load(opt->tp.topology_path, &topo, err, errlen))
            return 0;
        const int world = topo.world;
        ds4_tp_topology_free(&topo);
        if (opt->tp.rank < 0 || opt->tp.rank >= world) {
            tp_set_err(err, errlen,
                       "--tp-rank %d is out of range for a world-%d mesh",
                       opt->tp.rank, world);
            return 0;
        }
        if (opt->tp.role == DS4_TP_LEADER && opt->tp.rank != 0) {
            tp_set_err(err, errlen,
                       "--role coordinator must be rank 0; --tp-rank %d given",
                       opt->tp.rank);
            return 0;
        }
        if (opt->tp.role == DS4_TP_WORKER && opt->tp.rank == 0) {
            tp_set_err(err, errlen,
                       "--role worker cannot be rank 0 (the leader owns rank 0)");
            return 0;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Slab layout.
 * --------------------------------------------------------------------- */

uint64_t ds4_tp_slab_bytes(uint32_t n_layer, uint32_t n_embd, uint32_t world) {
    uint64_t vec = (uint64_t)n_embd * sizeof(float);
    uint64_t slots = (uint64_t)n_layer * DS4_TP_GATES_PER_LAYER;
    uint64_t peers = world > 1 ? (uint64_t)(world - 1) : 0;
    return slots * vec * (1 + peers + 1) +   /* out + in(peers) + combined */
           slots * peers * 8 +               /* per-peer in flags */
           slots * 8 +                       /* out flag staging */
           16 +                              /* token slot */
           slots * 4 +                       /* GPU-written gate-ready flags */
           (uint64_t)n_layer * DS4_TP_BATCH_MAX_ROWS * vec * (1 + peers + 1);
}

static void tp_slab_layout(ds4_tp *tp) {
    uint64_t vec = tp->vec_bytes;
    uint64_t slots = tp->n_slots;
    uint64_t peers = tp->world > 1 ? (uint64_t)(tp->world - 1) : 0;
    tp->out_off = 0;
    tp->in_off = slots * vec;
    tp->in_peer_bytes = peers * vec;
    tp->combined_off = tp->in_off + slots * peers * vec;
    tp->in_flags_off = tp->combined_off + slots * vec;
    tp->token_off = tp->in_flags_off + slots * peers * 8;
    tp->out_flags_off = tp->token_off + 16;
    tp->gpu_flags_off = tp->out_flags_off + slots * 8;
    tp->batch_out_off = tp->gpu_flags_off + slots * 4;
    tp->batch_in_off = tp->batch_out_off +
                       (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * vec;
    tp->batch_combined_off = tp->batch_in_off +
                       (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * peers * vec;
    tp->slab_bytes = tp->batch_combined_off +
                     (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * vec;
}

uint64_t ds4_tp_slab_gpu_flags_offset(const ds4_tp *tp) {
    return tp->gpu_flags_off;
}

static uint32_t tp_slot(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    (void)tp;
    return layer * DS4_TP_GATES_PER_LAYER + gate;
}

uint64_t ds4_tp_slab_out_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    return tp->out_off + (uint64_t)tp_slot(tp, layer, gate) * tp->vec_bytes;
}

uint64_t ds4_tp_slab_in_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    /* Base in-slot (world=2: the single peer partial).  The mesh exchange
     * uses tp_slab_in_peer_offset() below to address each peer's slot. */
    return tp->in_off + (uint64_t)tp_slot(tp, layer, gate) * tp->in_peer_bytes;
}

uint64_t ds4_tp_slab_combined_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate) {
    return tp->combined_off + (uint64_t)tp_slot(tp, layer, gate) * tp->vec_bytes;
}

/* Peer-labeled in-slot index: peers are ordered by rank (0..world-1
 * excluding self), so slot index = peer<rank ? peer : peer-1. */
static uint64_t tp_in_peer_index(const ds4_tp *tp, int peer) {
    return peer < tp->rank ? (uint64_t)peer : (uint64_t)(peer - 1);
}

/* Peer-labeled in-slot: peer partials land at index (peer<rank ? peer :
 * peer-1) inside the slot's (world-1) in-vector block, so the canonical
 * rank-order sum can walk slots in rank order. */
static uint64_t tp_slab_in_peer_offset(const ds4_tp *tp, uint32_t layer,
                                       uint32_t gate, int peer) {
    return tp->in_off + (uint64_t)tp_slot(tp, layer, gate) * tp->in_peer_bytes +
           tp_in_peer_index(tp, peer) * tp->vec_bytes;
}

uint64_t ds4_tp_slab_batch_out_offset(const ds4_tp *tp, uint32_t layer) {
    return tp->batch_out_off +
           (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
}

uint64_t ds4_tp_slab_batch_in_offset(const ds4_tp *tp, uint32_t layer) {
    /* Base batch-in (world=2: the single peer partial). */
    return tp->batch_in_off +
           (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
}

uint64_t ds4_tp_slab_batch_combined_offset(const ds4_tp *tp, uint32_t layer) {
    return tp->batch_combined_off +
           (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
}

static uint64_t tp_slab_batch_in_peer_offset(const ds4_tp *tp, uint32_t layer,
                                             int peer) {
    uint64_t idx = peer < tp->rank ? (uint64_t)peer : (uint64_t)(peer - 1);
    return tp->batch_in_off +
           (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->in_peer_bytes +
           idx * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
}

/* Canonical rank-order combine for world>2: sum partial[0] + partial[1] +
 * ... + partial[world-1] in rank order into the combined slot, where
 * partial[i] is the local out vector when i==rank and the peer's labeled
 * in vector otherwise.  The identical expression on every rank keeps the
 * hidden states bit-exact across the mesh. */
static void tp_combine(ds4_tp *tp, uint32_t layer, uint32_t gate) {
    const uint32_t slot = layer * DS4_TP_GATES_PER_LAYER + gate;
    float *dst = (float *)(tp->slab + tp->combined_off +
                           (uint64_t)slot * tp->vec_bytes);
    const float *out = (const float *)(tp->slab + tp->out_off +
                                       (uint64_t)slot * tp->vec_bytes);
    const uint64_t words = tp->vec_bytes / sizeof(float);
    const float *first = tp->rank == 0 ? out :
        (const float *)(tp->slab +
            tp_slab_in_peer_offset(tp, layer, gate, 0));
    memcpy(dst, first, tp->vec_bytes);
    for (int i = 1; i < tp->world; i++) {
        if (i == tp->rank) continue;
        const float *src = (const float *)(tp->slab +
            tp_slab_in_peer_offset(tp, layer, gate, i));
        for (uint64_t k = 0; k < words; k++) dst[k] += src[k];
    }
}

/* Verify-block batch combine: canonical rank-order sum of the `rows` row
 * partials (out + per-peer in regions) into batch_combined. */
static void tp_batch_combine(ds4_tp *tp, uint32_t layer, uint32_t rows) {
    const uint64_t words = tp->vec_bytes / sizeof(float);
    const float *out = (const float *)(tp->slab +
        tp->batch_out_off + (uint64_t)layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes);
    float *dst = (float *)(tp->slab + ds4_tp_slab_batch_combined_offset(tp, layer));
    const float *first = tp->rank == 0 ? out :
        (const float *)(tp->slab +
            tp_slab_batch_in_peer_offset(tp, layer, 0));
    memcpy(dst, first, (uint64_t)rows * tp->vec_bytes);
    for (int i = 1; i < tp->world; i++) {
        if (i == tp->rank) continue;
        const float *src = (const float *)(tp->slab +
            tp_slab_batch_in_peer_offset(tp, layer, i));
        for (uint64_t r = 0; r < rows; r++) {
            const float *sr = src + r * words;
            float *dr = dst + r * words;
            for (uint64_t k = 0; k < words; k++) dr[k] += sr[k];
        }
    }
}

/* ------------------------------------------------------------------------
 * RDMA path.
 * --------------------------------------------------------------------- */

#ifdef DS4_TP_HAVE_VERBS

static int tp_rdma_load_api(ds4_tp_verbs_api *api) {
    if (api->handle) return 1;
    void *h = dlopen("/usr/lib/librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("librdma.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
#define TP_SYM(field, name) \
    do { \
        api->field = (__typeof__(api->field))dlsym(h, name); \
        if (!api->field) { dlclose(h); return 0; } \
    } while (0)
    TP_SYM(get_device_list, "ibv_get_device_list");
    TP_SYM(free_device_list, "ibv_free_device_list");
    TP_SYM(get_device_name, "ibv_get_device_name");
    TP_SYM(open_device, "ibv_open_device");
    TP_SYM(close_device, "ibv_close_device");
    TP_SYM(query_device, "ibv_query_device");
    TP_SYM(query_port, "ibv_query_port");
    TP_SYM(query_gid, "ibv_query_gid");
    TP_SYM(alloc_pd, "ibv_alloc_pd");
    TP_SYM(dealloc_pd, "ibv_dealloc_pd");
    TP_SYM(reg_mr, "ibv_reg_mr");
    TP_SYM(dereg_mr, "ibv_dereg_mr");
    TP_SYM(create_cq, "ibv_create_cq");
    TP_SYM(destroy_cq, "ibv_destroy_cq");
    TP_SYM(create_qp, "ibv_create_qp");
    TP_SYM(destroy_qp, "ibv_destroy_qp");
    TP_SYM(modify_qp, "ibv_modify_qp");
#undef TP_SYM
    api->handle = h;
    return 1;
}

/* Probe only: does this machine expose a verbs device right now? */
static int tp_rdma_probe(ds4_tp_verbs_api *api) {
    if (!tp_rdma_load_api(api)) return 0;
    int num = 0;
    struct ibv_device **devs = api->get_device_list(&num);
    if (!devs) return 0;
    api->free_device_list(devs);
    return num > 0;
}

static int tp_rdma_open_link(ds4_tp *tp, int peer, char *err, size_t errlen) {
    ds4_tp_rdma_link *r = &tp->rdma[peer];
    ds4_tp_verbs_api *api = &tp->rdma_api;
    int num = 0;
    struct ibv_device **devs = api->get_device_list(&num);
    if (!devs || num == 0) {
        tp_set_err(err, errlen, "tp rdma: no verbs devices");
        if (devs) api->free_device_list(devs);
        return 0;
    }
    /* One verbs device per Thunderbolt port (rdma_enN).  The mesh has one
     * link per peer; match the IPv4 in the device's mapped GID to the
     * link's local address so each QP rides the right cable.  The classic
     * pair (world==2, no topology) keeps the old active-device pick. */
    const char *want_name = tp->opt.rdma_device;
    const int link = tp_link_to(tp->rank, peer, tp->world);
    const char *link_host = tp->topo.world ?
        tp->topo.node[tp->rank].host[link] : NULL;
    struct in_addr link_ip;
    const bool have_ip = link_host &&
        inet_pton(AF_INET, link_host, &link_ip) == 1;
    char states[256] = "";
    for (int i = 0; i < num && !r->ctx; i++) {
        const char *name = api->get_device_name(devs[i]);
        if (want_name && strcmp(want_name, name) != 0) continue;
        struct ibv_context *ctx = api->open_device(devs[i]);
        if (!ctx) continue;
        struct ibv_port_attr pa;
        if (api->query_port(ctx, 1, &pa) == 0 &&
            (pa.state == IBV_PORT_ACTIVE || want_name)) {
            /* The driver only connects through the IPv4-mapped GID
             * (::ffff:a.b.c.d), which exists only when the Thunderbolt
             * member interface carries its own IPv4. */
            int gid_idx = -1;
            for (int g = 0; g < pa.gid_tbl_len; g++) {
                union ibv_gid tmp;
                if (api->query_gid(ctx, 1, g, &tmp) != 0) continue;
                uint64_t hi;
                uint16_t mid, v4tag;
                memcpy(&hi, &tmp.raw[0], 8);
                memcpy(&mid, &tmp.raw[8], 2);
                memcpy(&v4tag, &tmp.raw[10], 2);
                if (hi != 0 || mid != 0 || v4tag != 0xffff) continue;
                if (!have_ip) {
                    gid_idx = g;
                    break;
                }
                struct in_addr gid_ip;
                memcpy(&gid_ip, &tmp.raw[12], 4);
                if (gid_ip.s_addr == link_ip.s_addr) {
                    gid_idx = g;
                    break;
                }
            }
            if (gid_idx >= 0) {
                r->ctx = ctx;
                r->port = pa;
                r->gid_index = gid_idx;
                fprintf(stderr, "ds4-tp: rdma link %d device %s (port state %d)\n",
                        peer, name, (int)pa.state);
                break;
            }
        }
        size_t off = strlen(states);
        snprintf(states + off, sizeof(states) - off, "%s%s=%d",
                 off ? ", " : "", name, (int)pa.state);
        api->close_device(ctx);
    }
    api->free_device_list(devs);
    if (!r->ctx) {
        tp_set_err(err, errlen,
                   "tp rdma: no device with an active port%s (%s); is the peer up "
                   "and rdma_ctl enabled on both machines?",
                   have_ip ? " whose IPv4-mapped GID matches the link address" : "",
                   states);
        return 0;
    }
    if (!tp->opt.rdma_gid_index_set) {
        if (api->query_gid(r->ctx, 1, r->gid_index, &r->gid) != 0) {
            tp_set_err(err, errlen, "tp rdma: query_gid(%d): %s",
                       r->gid_index, strerror(errno));
            return 0;
        }
    } else {
        r->gid_index = tp->opt.rdma_gid_index;
        if (api->query_gid(r->ctx, 1, r->gid_index, &r->gid) != 0) {
            tp_set_err(err, errlen, "tp rdma: query_gid(%d): %s",
                       r->gid_index, strerror(errno));
            return 0;
        }
    }
    r->pd = api->alloc_pd(r->ctx);
    if (!r->pd) {
        tp_set_err(err, errlen, "tp rdma: alloc_pd failed");
        return 0;
    }
    r->cq = api->create_cq(r->ctx, 512, NULL, NULL, 0);
    if (!r->cq) {
        tp_set_err(err, errlen, "tp rdma: create_cq failed");
        return 0;
    }
    struct ibv_qp_init_attr qia = {0};
    qia.send_cq = r->cq;
    qia.recv_cq = r->cq;
    qia.qp_type = IBV_QPT_UC;
    qia.cap.max_send_wr = 256;
    qia.cap.max_recv_wr = 64;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    qia.cap.max_inline_data = 0;
    r->qp = api->create_qp(r->pd, &qia);
    if (!r->qp) {
        tp_set_err(err, errlen, "tp rdma: create_qp(UC): %s", strerror(errno));
        return 0;
    }
    r->max_inline = qia.cap.max_inline_data;

    pthread_mutex_init(&r->post_lock, NULL);
    return 1;
}

static int tp_rdma_post_gate_recv(ds4_tp *tp, int peer, uint64_t seq);

/* Register the slab on one link's device and bring the UC QP to RTS, then
 * barrier with the peer over the pair's data socket (the mesh exchanges
 * one RDMA_INFO per link; the classic pair has exactly one). */
static int tp_rdma_info_exchange(ds4_tp *tp, int peer, char *err, size_t errlen) {
    ds4_tp_rdma_link *r = &tp->rdma[peer];
    ds4_tp_verbs_api *api = &tp->rdma_api;
    r->mr = api->reg_mr(r->pd, tp->slab, tp->slab_bytes,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                        IBV_ACCESS_REMOTE_WRITE);
    if (!r->mr) {
        tp_set_err(err, errlen, "tp rdma: reg_mr(%llu bytes): %s",
                   (unsigned long long)tp->slab_bytes, strerror(errno));
        return 0;
    }
    ds4_tp_rdma_info mine = {0};
    mine.slab_base = (uint64_t)(uintptr_t)tp->slab;
    mine.rkey = r->mr->rkey;
    mine.qpn = r->qp->qp_num;
    mine.psn = (uint32_t)(getpid() ^ (uintptr_t)tp ^ (uintptr_t)r) & 0xffffff;
    mine.mtu = (uint32_t)r->port.active_mtu;
    mine.lid = r->port.lid;
    memcpy(mine.gid, r->gid.raw, 16);
    mine.link_layer = r->port.link_layer;
    if (!tp_send_frame(tp->data_fd[peer], DS4_TP_FRAME_RDMA_INFO,
                       &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp rdma: info send failed");
        return 0;
    }
    uint32_t type = 0, bytes = 0;
    if (!tp_read_frame_header(tp->data_fd[peer], &type, &bytes) ||
        type != DS4_TP_FRAME_RDMA_INFO || bytes != sizeof(r->peer) ||
        !tp_read_full(tp->data_fd[peer], &r->peer, sizeof(r->peer))) {
        tp_set_err(err, errlen, "tp rdma: info recv failed");
        return 0;
    }

    /* INIT -> RTR -> RTS with the exact recipe the driver accepts (same as
     * JACCL): MTU 1024 and GRH via the IPv4-mapped GID. */
    struct ibv_qp_attr a = {0};
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = 1;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                        IBV_ACCESS_REMOTE_WRITE;
    if (api->modify_qp(r->qp, &a,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify INIT: %s", strerror(errno));
        return 0;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = IBV_MTU_1024;
    a.dest_qp_num = r->peer.qpn;
    a.rq_psn = r->peer.psn;
    a.ah_attr.dlid = (uint16_t)r->peer.lid;
    a.ah_attr.port_num = 1;
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, r->peer.gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)r->gid_index;
    a.ah_attr.grh.hop_limit = 1;
    if (api->modify_qp(r->qp, &a,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTR: %s", strerror(errno));
        return 0;
    }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = mine.psn;
    if (api->modify_qp(r->qp, &a, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) {
        tp_set_err(err, errlen, "tp rdma: modify RTS: %s", strerror(errno));
        return 0;
    }
    if (tp->vec_bytes > 2ull * DS4_TP_RDMA_MAX_MSG) {
        tp_set_err(err, errlen,
                   "tp rdma: gate vector %llu bytes exceeds twice the driver's "
                   "%u message limit",
                   (unsigned long long)tp->vec_bytes, DS4_TP_RDMA_MAX_MSG);
        return 0;
    }
    if (tp->vec_bytes > DS4_TP_RDMA_MAX_MSG) {
        fprintf(stderr,
                "ds4-tp: rdma gate vectors ride as 2 chunked messages "
                "(%llu bytes > %u limit)\n",
                (unsigned long long)tp->vec_bytes, DS4_TP_RDMA_MAX_MSG);
    }
    /* Leave the receive queue empty for an initial bulk prefill.  The first
     * decode gate arms the normal lookahead window after prefill finishes. */
    if (!tp_send_frame(tp->data_fd[peer], DS4_TP_FRAME_RDMA_READY, NULL, 0)) {
        tp_set_err(err, errlen, "tp rdma: ready send failed");
        return 0;
    }
    uint32_t rtype = 0, rbytes = 0;
    if (!tp_read_frame_header(tp->data_fd[peer], &rtype, &rbytes) ||
        rtype != DS4_TP_FRAME_RDMA_READY || rbytes != 0) {
        tp_set_err(err, errlen, "tp rdma: ready barrier failed");
        return 0;
    }
    return 1;
}

static int tp_rdma_register_and_exchange(ds4_tp *tp, char *err, size_t errlen) {
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        if (!tp_rdma_info_exchange(tp, m, err, errlen)) return 0;
    }
    return 1;
}

/* ibv_wc_status_str lives in librdma; resolve lazily to keep the dlopen-only
 * linkage discipline. */
static const char *tp_wc_status_str(int status) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "wc status %d", status);
    return buf;
}

/* Slab slot a given gate seq lands in.  DS4 fires every slot in order
 * (identity mapping); GLM's schedule from the hello skips dense layers
 * and the ATTN slots. */
static uint32_t tp_gate_slot(const ds4_tp *tp, uint64_t seq) {
    if (tp->gates_per_token == 0)
        return (uint32_t)((seq - 1) % tp->n_slots);
    return tp->gate_slot_start +
           (uint32_t)((seq - 1) % tp->gates_per_token) * tp->gate_slot_step;
}

/* Reap completions for one link: send CQEs free send-queue slots, recv
 * CQEs advance the arrival watermark (UC is in-order, so gate seq recv
 * completions arrive monotonically).  Returns 0 on any completion error. */
static int tp_rdma_drain_cq(ds4_tp *tp, int peer) {
    ds4_tp_rdma_link *r = &tp->rdma[peer];
    struct ibv_wc wc[16];
    int n = ibv_poll_cq(r->cq, 16, wc);
    if (n < 0) return 0;
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            fprintf(stderr, "ds4-tp: rdma completion error (link %d): %s (wr_id %llu)\n",
                    peer, tp_wc_status_str(wc[i].status),
                    (unsigned long long)wc[i].wr_id);
            return 0;
        }
        if (wc[i].opcode & IBV_WC_RECV) {
            if (wc[i].wr_id > r->recv_done) r->recv_done = wc[i].wr_id;
        } else if (r->send_outstanding > 0) {
            r->send_outstanding--;
        }
    }
    return 1;
}

/* Arm the receive for gate seq on one link: UC delivery order pairs the
 * peer's seq'th send with our seq'th posted recv, landing it in that
 * peer's labeled in-slot. */
static int tp_rdma_post_gate_recv(ds4_tp *tp, int peer, uint64_t seq) {
    ds4_tp_rdma_link *r = &tp->rdma[peer];
    const uint32_t slot = tp_gate_slot(tp, seq);
    const uintptr_t base =
        (uintptr_t)(tp->slab + tp->in_off +
                    (uint64_t)slot * tp->in_peer_bytes +
                    tp_in_peer_index(tp, peer) * tp->vec_bytes);
    /* Vectors above the driver's 16KB message cap ride as two chunks
     * landing contiguously in the slot. UC delivery is in-order and both
     * sides post/send strictly in seq order, so the k'th send always
     * matches the k'th recv; only the FINAL chunk carries the seq as
     * wr_id, so the arrival watermark advances when the slot is whole. */
    uint64_t off = 0;
    while (off < tp->vec_bytes) {
        const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
            DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
        const int last = off + len == tp->vec_bytes;
        struct ibv_sge sge;
        struct ibv_recv_wr wr, *bad = NULL;
        memset(&wr, 0, sizeof(wr));
        sge.addr = base + off;
        sge.length = (uint32_t)len;
        sge.lkey = r->mr->lkey;
        wr.wr_id = last ? seq : 0;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        if (ibv_post_recv(r->qp, &wr, &bad) != 0) {
            fprintf(stderr, "ds4-tp: rdma post_recv(link %d seq %llu off %llu): %s\n",
                    peer, (unsigned long long)seq, (unsigned long long)off,
                    strerror(errno));
            return 0;
        }
        off += len;
    }
    return 1;
}

/* Canonical rank-order combine for world>2: sum partial[0] + partial[1] +
 * ... + partial[world-1] in rank order into the combined slot, where
 * partial[i] is the local out vector when i==rank and the peer's labeled
 * in vector otherwise.  The identical expression on every rank keeps the
 * hidden states bit-exact across the mesh. */
/* One decode gate: broadcast/gather over every link.  Each rank sends its
 * partial to all peers, posts receives from all peers, waits for every
 * recv completion, then (world>2) folds the canonical rank-order sum into
 * the combined slot for the GPU combine. */
static int tp_rdma_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
    const uint32_t slot = layer * DS4_TP_GATES_PER_LAYER + gate;
    if (getenv("DS4_TP_GATE_TRACE")) {
        fprintf(stderr, "ds4-tp: gate trace l=%u g=%u seq=%llu want_slot=%u\n",
                layer, gate, (unsigned long long)seq, tp_gate_slot(tp, seq));
    }
    if (slot != tp_gate_slot(tp, seq)) {
        fprintf(stderr, "ds4-tp: gate order broke: layer %u gate %u vs seq %llu\n",
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    const uintptr_t send_base =
        (uintptr_t)(tp->slab + tp->out_off + (uint64_t)slot * tp->vec_bytes);
    int ok = 1;
    /* Arm each link's lookahead window and post this gate's send. */
    for (int m = 0; ok && m < tp->world; m++) {
        if (m == tp->rank) continue;
        ds4_tp_rdma_link *r = &tp->rdma[m];
        pthread_mutex_lock(&r->post_lock);
        if (!r->recv_window_active) {
            for (uint64_t s = seq; ok && s < seq + DS4_TP_RDMA_RECV_WINDOW; s++)
                ok = tp_rdma_post_gate_recv(tp, m, s);
            if (ok) r->recv_window_active = true;
        }
        for (uint64_t off = 0; ok && off < tp->vec_bytes; ) {
            const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
                DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
            struct ibv_sge sge;
            struct ibv_send_wr wr, *bad = NULL;
            memset(&wr, 0, sizeof(wr));
            sge.addr = send_base + off;
            sge.length = (uint32_t)len;
            sge.lkey = r->mr->lkey;
            wr.wr_id = seq;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_SEND;
            wr.send_flags = IBV_SEND_SIGNALED;
            ok = ibv_post_send(r->qp, &wr, &bad) == 0;
            if (!ok) {
                fprintf(stderr, "ds4-tp: rdma post_send(link %d): %s\n",
                        m, strerror(errno));
            } else {
                r->send_outstanding++;
            }
            off += len;
        }
        pthread_mutex_unlock(&r->post_lock);
    }

    double deadline = 0.0;
    uint32_t peer_poll = 0;
    while (ok) {
        bool all_done = true;
        for (int m = 0; m < tp->world; m++) {
            if (m == tp->rank) continue;
            if (tp->rdma[m].recv_done < seq) { all_done = false; break; }
        }
        if (all_done) break;
        for (int m = 0; ok && m < tp->world; m++) {
            if (m == tp->rank) continue;
            if (!tp_rdma_drain_cq(tp, m)) ok = 0;
        }
        if (ok && (peer_poll++ & 0x3fffu) == 0) {
            for (int m = 0; ok && m < tp->world; m++) {
                if (m == tp->rank) continue;
                if (tp_peer_closed(tp, m)) {
                    fprintf(stderr, "ds4-tp: peer %d disconnected during RDMA gate\n", m);
                    ok = 0;
                }
            }
        }
        if (deadline == 0.0) deadline = tp_now_sec() + (double)tp->timeout_sec;
        else if (tp_now_sec() > deadline) {
            fprintf(stderr, "ds4-tp: timeout waiting gate seq %llu\n",
                    (unsigned long long)seq);
            ok = 0;
        }
    }
    for (int m = 0; ok && m < tp->world; m++) {
        if (m == tp->rank) continue;
        ds4_tp_rdma_link *r = &tp->rdma[m];
        pthread_mutex_lock(&r->post_lock);
        ok = tp_rdma_post_gate_recv(tp, m, seq + DS4_TP_RDMA_RECV_WINDOW);
        if (ok) r->last_gate_seq = seq;
        pthread_mutex_unlock(&r->post_lock);
    }
    if (ok && tp->world > 2) tp_combine(tp, layer, gate);
    return ok;
}

static int tp_rdma_big_gate_capable(const ds4_tp *tp, int peer) {
    const uint64_t stage_bytes =
        (uint64_t)DS4_TP_RDMA_BULK_SLOTS * DS4_TP_RDMA_MAX_MSG;
    const uint64_t batch_region_bytes =
        (uint64_t)tp->n_layer * DS4_TP_BATCH_MAX_ROWS * tp->vec_bytes;
    return tp->rdma[peer].qp && tp->rdma[peer].mr &&
           batch_region_bytes >= stage_bytes;
}

/* Decode keeps a lookahead window of receives on the latency QP. Before a
 * later prompt can reuse that QP for bulk rows, consume those receives with
 * dummy sends on both ends of the link. The TCP big-gate header exchange is
 * the barrier that guarantees both sides have reached this transition. */
static int tp_rdma_drain_decode_window(ds4_tp *tp, int peer) {
    ds4_tp_rdma_link *r = &tp->rdma[peer];
    if (!r->recv_window_active) return 1;

    const uint32_t chunks_per_gate =
        (uint32_t)((tp->vec_bytes + DS4_TP_RDMA_MAX_MSG - 1u) /
                   DS4_TP_RDMA_MAX_MSG);
    const uint32_t nwr = DS4_TP_RDMA_RECV_WINDOW * chunks_per_gate;
    struct ibv_sge sge[DS4_TP_RDMA_RECV_WINDOW * 2u];
    struct ibv_send_wr wr[DS4_TP_RDMA_RECV_WINDOW * 2u];
    memset(wr, 0, sizeof(wr));
    uint8_t *scratch = tp->slab + tp->batch_out_off;
    uint32_t wi = 0;
    for (uint32_t gate = 0; gate < DS4_TP_RDMA_RECV_WINDOW; gate++) {
        for (uint64_t off = 0; off < tp->vec_bytes; ) {
            const uint64_t len = tp->vec_bytes - off > DS4_TP_RDMA_MAX_MSG ?
                DS4_TP_RDMA_MAX_MSG : tp->vec_bytes - off;
            sge[wi] = (struct ibv_sge) {
                .addr = (uintptr_t)(scratch + off),
                .length = (uint32_t)len,
                .lkey = r->mr->lkey,
            };
            wr[wi].wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)wi + 1u);
            wr[wi].sg_list = &sge[wi];
            wr[wi].num_sge = 1;
            wr[wi].opcode = IBV_WR_SEND;
            wr[wi].send_flags = wi + 1u == nwr ? IBV_SEND_SIGNALED : 0;
            if (wi > 0) wr[wi - 1u].next = &wr[wi];
            wi++;
            off += len;
        }
    }

    pthread_mutex_lock(&r->post_lock);
    struct ibv_send_wr *bad = NULL;
    if (ibv_post_send(r->qp, wr, &bad) != 0) {
        fprintf(stderr, "ds4-tp: rdma receive-window drain post failed: %s\n",
                strerror(errno));
        pthread_mutex_unlock(&r->post_lock);
        return 0;
    }

    uint32_t recv_done = 0;
    int send_done = 0;
    const double deadline = tp_now_sec() + (double)tp->timeout_sec;
    uint32_t peer_poll = 0;
    while (recv_done < nwr || !send_done) {
        struct ibv_wc wc[DS4_TP_RDMA_RECV_WINDOW * 2u + 1u];
        int n = ibv_poll_cq(r->cq,
                           (int)(DS4_TP_RDMA_RECV_WINDOW * 2u + 1u), wc);
        if (n < 0) {
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
        for (int i = 0; i < n; i++) {
            if (wc[i].status != IBV_WC_SUCCESS) {
                fprintf(stderr, "ds4-tp: rdma receive-window drain: %s\n",
                        tp_wc_status_str(wc[i].status));
                pthread_mutex_unlock(&r->post_lock);
                return 0;
            }
            if (wc[i].opcode & IBV_WC_RECV) {
                recv_done++;
            } else if (wc[i].wr_id & DS4_TP_RDMA_BULK_WR_TAG) {
                send_done = 1;
            } else if (r->send_outstanding > 0) {
                r->send_outstanding--;
            }
        }
        if ((peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp, peer)) {
            fprintf(stderr,
                    "ds4-tp: peer %d disconnected while draining RDMA receives\n",
                    peer);
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
        if (tp_now_sec() > deadline) {
            fprintf(stderr,
                    "ds4-tp: timeout draining RDMA receive window (%u/%u)\n",
                    recv_done, nwr);
            pthread_mutex_unlock(&r->post_lock);
            return 0;
        }
    }
    r->recv_done = r->last_gate_seq;
    r->recv_window_active = false;
    pthread_mutex_unlock(&r->post_lock);
    return 1;
}

/* Large prefill row swaps share the latency QP.  No future decode receives
 * are queued, so each round can post its 1 MiB receive window before sending
 * the matching 16 KiB messages.  Verify scratch provides already-registered
 * staging memory and is idle during normal prefill.  Per-link: sends `out`
 * to peer and receives the peer's `in` into `in` (world>2 callers provide a
 * per-peer buffer of (world-1)*bytes). */
static int tp_rdma_big_gate_exchange(ds4_tp *tp, int peer,
                                     const void *out,
                                     void *in,
                                     uint64_t bytes) {
    ds4_tp_rdma_link *r = &tp->rdma[peer];
    if (!tp_rdma_big_gate_capable(tp, peer) || r->recv_window_active) return 0;

    /* Payloads already inside the registered slab (verify batches) can ride
     * directly. Ordinary prefill tensors use the idle verify regions as
     * registered staging because their standalone MTLBuffers are not in the
     * NIC memory region. */
    const uintptr_t slab_lo = (uintptr_t)tp->slab;
    const uintptr_t slab_hi = slab_lo + tp->slab_bytes;
    const uintptr_t out_lo = (uintptr_t)out;
    const uintptr_t in_lo = (uintptr_t)in;
    const bool direct =
        out_lo >= slab_lo && out_lo <= slab_hi && bytes <= slab_hi - out_lo &&
        in_lo >= slab_lo && in_lo <= slab_hi && bytes <= slab_hi - in_lo;
    uint8_t *stage_send = tp->slab + tp->batch_out_off;
    uint8_t *stage_recv = tp->slab + tp->batch_in_off;
    uint64_t off = 0;
    while (off < bytes) {
        const uint64_t remaining = bytes - off;
        uint32_t chunks = (uint32_t)((remaining + DS4_TP_RDMA_MAX_MSG - 1u) /
                                     DS4_TP_RDMA_MAX_MSG);
        if (chunks > DS4_TP_RDMA_BULK_SLOTS)
            chunks = DS4_TP_RDMA_BULK_SLOTS;

        uint32_t lens[DS4_TP_RDMA_BULK_SLOTS];
        uint64_t chunk_off[DS4_TP_RDMA_BULK_SLOTS];
        uint64_t round_bytes = 0;
        for (uint32_t i = 0; i < chunks; i++) {
            const uint64_t left = remaining - round_bytes;
            lens[i] = (uint32_t)(left > DS4_TP_RDMA_MAX_MSG ?
                                 DS4_TP_RDMA_MAX_MSG : left);
            chunk_off[i] = direct ? round_bytes :
                (uint64_t)i * DS4_TP_RDMA_MAX_MSG;
            if (!direct) {
                memcpy(stage_send + chunk_off[i],
                       (const uint8_t *)out + off + round_bytes, lens[i]);
            }
            round_bytes += lens[i];
        }

        struct ibv_sge recv_sge[DS4_TP_RDMA_BULK_SLOTS];
        struct ibv_recv_wr recv_wr[DS4_TP_RDMA_BULK_SLOTS];
        memset(recv_wr, 0, sizeof(recv_wr));
        for (uint32_t i = 0; i < chunks; i++) {
            recv_sge[i] = (struct ibv_sge) {
                .addr = direct ? in_lo + off + chunk_off[i] :
                                 (uintptr_t)(stage_recv + chunk_off[i]),
                .length = lens[i],
                .lkey = r->mr->lkey,
            };
            recv_wr[i].wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)i + 1u);
            recv_wr[i].sg_list = &recv_sge[i];
            recv_wr[i].num_sge = 1;
            recv_wr[i].next = i + 1u < chunks ? &recv_wr[i + 1u] : NULL;
        }
        struct ibv_recv_wr *bad_recv = NULL;
        if (ibv_post_recv(r->qp, recv_wr, &bad_recv) != 0) {
            fprintf(stderr, "ds4-tp: bulk rdma post_recv: %s\n",
                    strerror(errno));
            return 0;
        }
        atomic_thread_fence(memory_order_release);
        struct ibv_sge send_sge[DS4_TP_RDMA_BULK_SLOTS];
        struct ibv_send_wr send_wr[DS4_TP_RDMA_BULK_SLOTS];
        memset(send_wr, 0, sizeof(send_wr));
        for (uint32_t i = 0; i < chunks; i++) {
            send_sge[i] = (struct ibv_sge) {
                .addr = direct ? out_lo + off + chunk_off[i] :
                                 (uintptr_t)(stage_send + chunk_off[i]),
                .length = lens[i],
                .lkey = r->mr->lkey,
            };
            send_wr[i].wr_id = DS4_TP_RDMA_BULK_WR_TAG | ((uint64_t)i + 1u);
            send_wr[i].sg_list = &send_sge[i];
            send_wr[i].num_sge = 1;
            send_wr[i].opcode = IBV_WR_SEND;
            send_wr[i].send_flags = i + 1u == chunks ? IBV_SEND_SIGNALED : 0;
            send_wr[i].next = i + 1u < chunks ? &send_wr[i + 1u] : NULL;
        }
        struct ibv_send_wr *bad_send = NULL;
        if (ibv_post_send(r->qp, send_wr, &bad_send) != 0) {
            fprintf(stderr, "ds4-tp: bulk rdma post_send: %s\n",
                    strerror(errno));
            return 0;
        }

        uint32_t recv_done = 0;
        int send_done = 0;
        const double deadline = tp_now_sec() + (double)tp->timeout_sec;
        uint32_t peer_poll = 0;
        while (recv_done < chunks || !send_done) {
            struct ibv_wc wc[DS4_TP_RDMA_BULK_SLOTS + 1u];
            int n = ibv_poll_cq(r->cq,
                               (int)(DS4_TP_RDMA_BULK_SLOTS + 1u), wc);
            if (n < 0) return 0;
            for (int i = 0; i < n; i++) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    fprintf(stderr,
                            "ds4-tp: bulk rdma completion error: %s\n",
                            tp_wc_status_str(wc[i].status));
                    return 0;
                }
                if ((wc[i].wr_id & DS4_TP_RDMA_BULK_WR_TAG) == 0) {
                    /* A final latency-QP send completion can remain queued
                     * when a later prompt starts a bulk gate. */
                    if (wc[i].opcode & IBV_WC_RECV) {
                        if (wc[i].wr_id > r->recv_done)
                            r->recv_done = wc[i].wr_id;
                    } else if (r->send_outstanding > 0) {
                        r->send_outstanding--;
                    }
                    continue;
                }
                if (wc[i].opcode & IBV_WC_RECV) recv_done++;
                else send_done = 1;
            }
            if ((peer_poll++ & 0x3fffu) == 0 && tp_peer_closed(tp, peer)) {
                fprintf(stderr,
                        "ds4-tp: peer %d disconnected during bulk RDMA gate\n",
                        peer);
                return 0;
            }
            if (tp_now_sec() > deadline) {
                fprintf(stderr,
                        "ds4-tp: timeout waiting for bulk RDMA round "
                        "(%u/%u recvs, send=%d)\n",
                        recv_done, chunks, send_done);
                return 0;
            }
        }
        atomic_thread_fence(memory_order_acquire);
        if (!direct) {
            round_bytes = 0;
            for (uint32_t i = 0; i < chunks; i++) {
                memcpy((uint8_t *)in + off + round_bytes,
                       stage_recv + chunk_off[i], lens[i]);
                round_bytes += lens[i];
            }
        }
        off += round_bytes;
    }
    return 1;
}

static void tp_rdma_close(ds4_tp *tp) {
    ds4_tp_verbs_api *api = &tp->rdma_api;
    for (int m = 0; m < DS4_TP_MAX_WORLD; m++) {
        ds4_tp_rdma_link *r = &tp->rdma[m];
        if (r->qp) api->destroy_qp(r->qp);
        if (r->mr) api->dereg_mr(r->mr);
        if (r->cq) api->destroy_cq(r->cq);
        if (r->pd) api->dealloc_pd(r->pd);
        if (r->ctx) api->close_device(r->ctx);
        r->qp = NULL; r->mr = NULL; r->cq = NULL; r->pd = NULL; r->ctx = NULL;
    }
}

#endif /* DS4_TP_HAVE_VERBS */

/* ------------------------------------------------------------------------
 * Bring-up.
 * --------------------------------------------------------------------- */

static int tp_hello_exchange(ds4_tp *tp, int peer, const ds4_tp_identity *id,
                             int rdma_ok, char *err, size_t errlen) {
    ds4_tp_hello_fixed mine = {
        .magic = DS4_TP_MAGIC,
        .version = DS4_TP_PROTOCOL_VERSION,
        .role = (uint32_t)tp->opt.role,
        .rdma_ok = (uint32_t)rdma_ok,
        .gguf_bytes = id->gguf_bytes,
        .model_id = id->model_id,
        .n_layer = id->n_layer,
        .n_embd = id->n_embd,
        .n_vocab = id->n_vocab,
        .quant_bits = id->quant_bits,
        .ctx_size = id->ctx_size,
        .gate_slot_start = id->gate_slot_start,
        .gate_slot_step = id->gate_slot_step,
        .gates_per_token = id->gates_per_token,
        .world = (uint32_t)tp->world,
        .rank = (uint32_t)tp->rank,
    };
    ds4_tp_hello_fixed theirs;
    const int fd = tp->control_fd[peer];
    if (!tp_write_full(fd, &mine, sizeof(mine)) ||
        !tp_read_full(fd, &theirs, sizeof(theirs))) {
        tp_set_err(err, errlen, "tp hello exchange failed");
        return 0;
    }
    if (theirs.magic != DS4_TP_MAGIC) {
        tp_set_err(err, errlen, "tp hello: bad magic (mixed byte order or wrong peer?)");
        return 0;
    }
    if (theirs.version != DS4_TP_PROTOCOL_VERSION) {
        tp_set_err(err, errlen, "tp hello: protocol version %u != %u",
                   theirs.version, DS4_TP_PROTOCOL_VERSION);
        return 0;
    }
    if (theirs.world != (uint32_t)tp->world) {
        tp_set_err(err, errlen, "tp hello: world %u != %u",
                   theirs.world, tp->world);
        return 0;
    }
    if (theirs.rank == (uint32_t)tp->rank) {
        tp_set_err(err, errlen, "tp hello: both sides claim rank %u", tp->rank);
        return 0;
    }
    if (tp->rank == 0 && theirs.rank != (uint32_t)peer) {
        tp_set_err(err, errlen, "tp hello: worker on link to rank %d claims rank %u",
                   peer, theirs.rank);
        return 0;
    }
    if (theirs.role == mine.role) {
        tp_set_err(err, errlen, "tp hello: both sides claim role %u", mine.role);
        return 0;
    }
    if (theirs.gguf_bytes != mine.gguf_bytes || theirs.model_id != mine.model_id ||
        theirs.n_layer != mine.n_layer || theirs.n_embd != mine.n_embd ||
        theirs.n_vocab != mine.n_vocab || theirs.quant_bits != mine.quant_bits ||
        theirs.gate_slot_start != mine.gate_slot_start ||
        theirs.gate_slot_step != mine.gate_slot_step ||
        theirs.gates_per_token != mine.gates_per_token) {
        tp_set_err(err, errlen,
                   "tp hello: model mismatch (peer gguf=%llu id=%u layers=%u embd=%u "
                   "vocab=%u qbits=%u)",
                   (unsigned long long)theirs.gguf_bytes, theirs.model_id,
                   theirs.n_layer, theirs.n_embd, theirs.n_vocab, theirs.quant_bits);
        return 0;
    }
    tp->peer_ctx = theirs.ctx_size;
    tp->n_layer = id->n_layer;
    tp->n_embd = id->n_embd;
    tp->vec_bytes = (uint64_t)id->n_embd * sizeof(float);
    tp->n_slots = id->n_layer * DS4_TP_GATES_PER_LAYER;
    tp->gate_slot_start = id->gate_slot_start;
    tp->gate_slot_step = id->gate_slot_step;
    tp->gates_per_token = id->gates_per_token;
    tp_slab_layout(tp);
    if (tp->rank == 0) {
        /* Leader collects each worker's rdma_ok; the mesh-wide decision is
         * broadcast after the hello barrier. */
        tp->peer_rdma_ok[peer] = theirs.rdma_ok;
    } else {
        /* Worker: the leader broadcasts the transport decision. */
        uint32_t mode = 0;
        uint32_t type = 0, bytes = 0;
        if (!tp_read_frame_header(fd, &type, &bytes) ||
            type != DS4_TP_FRAME_RDMA_MODE || bytes != sizeof(mode) ||
            !tp_read_full(fd, &mode, sizeof(mode))) {
            tp_set_err(err, errlen, "tp rdma: mode barrier failed");
            return 0;
        }
        tp->rdma_active = mode != 0;
        if (tp->opt.transport == DS4_TP_TRANSPORT_RDMA && !tp->rdma_active) {
            tp_set_err(err, errlen,
                       "tp: --transport rdma but a peer has no active device");
            return 0;
        }
    }
    return 1;
}

int ds4_tp_create(
        ds4_tp **out,
        const ds4_tp_options *opt,
        const ds4_tp_identity *id,
        char *err,
        size_t errlen)
{
    *out = NULL;
    ds4_tp *tp = calloc(1, sizeof(*tp));
    if (!tp) {
        tp_set_err(err, errlen, "tp: out of memory");
        return 0;
    }
    tp->opt = *opt;
    tp->timeout_sec = DS4_TP_DEFAULT_TIMEOUT_SEC;
    const char *tmo = getenv("DS4_TP_TIMEOUT_SEC");
    if (tmo) tp->timeout_sec = (uint64_t)atoi(tmo);
    for (int i = 0; i < DS4_TP_MAX_WORLD; i++) {
        tp->control_fd[i] = -1;
        tp->data_fd[i] = -1;
    }

    /* Resolve the mesh: topology file, or the classic 2-node options. */
    if (opt->topology_path) {
        if (!ds4_tp_topology_load(opt->topology_path, &tp->topo, err, errlen))
            goto fail;
        tp->world = tp->topo.world;
        if (!opt->rank_set) {
            tp_set_err(err, errlen, "--tp-topology requires --tp-rank");
            goto fail;
        }
        tp->rank = opt->rank;
    } else {
        tp->world = 2;
        tp->rank = opt->role == DS4_TP_LEADER ? 0 : 1;
        tp->topo.world = 2;
        tp->topo.node[0].host[0] =
            strdup(opt->listen_host && opt->listen_host[0]
                   ? opt->listen_host : "0.0.0.0");
        tp->topo.node[0].port[0] = opt->listen_port > 0 ? opt->listen_port : 9000;
    }
    if (tp->world < 2 || tp->world > DS4_TP_MAX_WORLD ||
        tp->rank < 0 || tp->rank >= tp->world) {
        tp_set_err(err, errlen, "tp: invalid mesh rank %d / world %d",
                   tp->rank, tp->world);
        goto fail;
    }

    int rdma_ok = 0;
#ifdef DS4_TP_HAVE_VERBS
    if (opt->transport != DS4_TP_TRANSPORT_TCP &&
        (uint64_t)id->n_embd * sizeof(float) <= 2ull * DS4_TP_RDMA_MAX_MSG)
        rdma_ok = tp_rdma_probe(&tp->rdma_api);
#endif

    int leader_listener[DS4_TP_MAX_WORLD];
    for (int i = 0; i < DS4_TP_MAX_WORLD; i++) leader_listener[i] = -1;

    /* Phase A: control plane.  The leader listens on each link to a worker
     * (one listener per link stays open for the data accept too); every
     * worker dials node 0 on its link to the leader. */
    if (tp->rank == 0) {
        for (int m = 1; m < tp->world; m++) {
            const int link = tp_link_to(0, m, tp->world);
            const char *host = tp->topo.node[0].host[link];
            const int port = tp->topo.node[0].port[link];
            leader_listener[m] = tp_listen(host, port, err, errlen);
            if (leader_listener[m] < 0) goto fail;
            fprintf(stderr, "ds4-tp: waiting for worker %d on %s:%d ...\n",
                    m, host, port);
            tp->control_fd[m] = accept(leader_listener[m], NULL, NULL);
            if (tp->control_fd[m] < 0) {
                tp_set_err(err, errlen, "tp accept: %s", strerror(errno));
                goto fail;
            }
            tp_socket_tune(tp->control_fd[m]);
        }
    } else {
        const int leader_link = tp_link_to(0, tp->rank, tp->world);
        tp->control_fd[0] = tp_dial(tp->topo.node[0].host[leader_link],
                                    tp->topo.node[0].port[leader_link],
                                    (double)tp->timeout_sec, err, errlen);
        if (tp->control_fd[0] < 0) goto fail;
        tp_socket_tune(tp->control_fd[0]);
    }

    /* Phase B: hello barrier.  The leader exchanges identities with every
     * worker, decides the mesh-wide transport, and broadcasts the mode. */
    if (tp->rank == 0) {
        for (int m = 1; m < tp->world; m++) {
            if (!tp_hello_exchange(tp, m, id, rdma_ok, err, errlen)) goto fail;
        }
        int want_rdma = tp->opt.transport != DS4_TP_TRANSPORT_TCP;
        bool all_ok = rdma_ok != 0;
        for (int m = 1; m < tp->world; m++) {
            if (!tp->peer_rdma_ok[m]) all_ok = false;
        }
        tp->rdma_active = want_rdma && all_ok;
        if (tp->opt.transport == DS4_TP_TRANSPORT_RDMA && !tp->rdma_active) {
            tp_set_err(err, errlen,
                       "tp: --transport rdma but a peer has no active device");
            goto fail;
        }
        uint32_t mode = tp->rdma_active ? 1u : 0u;
        for (int m = 1; m < tp->world; m++) {
            if (!tp_send_frame(tp->control_fd[m], DS4_TP_FRAME_RDMA_MODE,
                               &mode, sizeof(mode))) goto fail;
        }
    } else {
        if (!tp_hello_exchange(tp, 0, id, rdma_ok, err, errlen)) goto fail;
    }

    /* Phase C: full-mesh data sockets (gate traffic).  Each node has one
     * link per peer; the lower-rank side of a pair listens, the higher
     * rank dials the lower's link address. */
    if (tp->rank == 0) {
        for (int m = 1; m < tp->world; m++) {
            tp->data_fd[m] = accept(leader_listener[m], NULL, NULL);
            if (tp->data_fd[m] < 0) {
                tp_set_err(err, errlen, "tp data accept: %s", strerror(errno));
                goto fail;
            }
            tp_socket_tune(tp->data_fd[m]);
        }
    } else {
        const int leader_link = tp_link_to(0, tp->rank, tp->world);
        tp->data_fd[0] = tp_dial(tp->topo.node[0].host[leader_link],
                                 tp->topo.node[0].port[leader_link],
                                 (double)tp->timeout_sec, err, errlen);
        if (tp->data_fd[0] < 0) goto fail;
        tp_socket_tune(tp->data_fd[0]);
        for (int m = 1; m < tp->world; m++) {
            if (m == tp->rank) continue;
            if (m < tp->rank) {
                const int link_m = tp_link_to(m, tp->rank, tp->world);
                tp->data_fd[m] = tp_dial(tp->topo.node[m].host[link_m],
                                         tp->topo.node[m].port[link_m],
                                         (double)tp->timeout_sec, err, errlen);
                if (tp->data_fd[m] < 0) goto fail;
                tp_socket_tune(tp->data_fd[m]);
            } else {
                const int link = tp_link_to(tp->rank, m, tp->world);
                int lfd = tp_listen(tp->topo.node[tp->rank].host[link],
                                    tp->topo.node[tp->rank].port[link],
                                    err, errlen);
                if (lfd < 0) goto fail;
                tp->data_fd[m] = accept(lfd, NULL, NULL);
                close(lfd);
                if (tp->data_fd[m] < 0) {
                    tp_set_err(err, errlen, "tp data accept: %s", strerror(errno));
                    goto fail;
                }
                tp_socket_tune(tp->data_fd[m]);
            }
        }
    }
    for (int i = 0; i < DS4_TP_MAX_WORLD; i++) {
        if (leader_listener[i] >= 0) close(leader_listener[i]);
    }

    /* Phase D: RDMA QPs, one per link. */
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) {
        for (int m = 0; m < tp->world; m++) {
            if (m == tp->rank) continue;
            if (!tp_rdma_open_link(tp, m, err, errlen)) goto fail;
        }
    }
#endif

    fprintf(stderr, "ds4-tp: rank %d/%d connected, transport=%s\n",
            tp->rank, tp->world, tp->rdma_active ? "rdma" : "tcp");
    *out = tp;
    return 1;
fail:
    for (int i = 0; i < DS4_TP_MAX_WORLD; i++) {
        if (leader_listener[i] >= 0) close(leader_listener[i]);
    }
    ds4_tp_free(tp);
    return 0;
}

int ds4_tp_attach_slab(ds4_tp *tp, void *base, char *err, size_t errlen) {
    tp->slab = base;
    memset(tp->slab + tp->in_flags_off, 0,
           (uint64_t)tp->n_slots * (uint64_t)(tp->world > 1 ? tp->world - 1 : 0) * 8);
    memset(tp->slab + tp->token_off, 0, 16);
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) return tp_rdma_register_and_exchange(tp, err, errlen);
#endif
    (void)err; (void)errlen;
    return 1;
}

void ds4_tp_free(ds4_tp *tp) {
    if (!tp) return;
#ifdef DS4_TP_HAVE_VERBS
    tp_rdma_close(tp);
#endif
    for (int i = 0; i < DS4_TP_MAX_WORLD; i++) {
        if (tp->control_fd[i] >= 0) close(tp->control_fd[i]);
        if (tp->data_fd[i] >= 0) close(tp->data_fd[i]);
    }
    ds4_tp_topology_free(&tp->topo);
    free(tp);
}

int ds4_tp_rank(const ds4_tp *tp) { return tp->rank; }
int ds4_tp_world(const ds4_tp *tp) { return tp ? tp->world : 0; }
bool ds4_tp_is_rdma(const ds4_tp *tp) { return tp->rdma_active; }
uint32_t ds4_tp_peer_ctx(const ds4_tp *tp) { return tp->peer_ctx; }
bool ds4_tp_failed(const ds4_tp *tp) {
    return tp && atomic_load_explicit(&tp->failed, memory_order_acquire);
}
void ds4_tp_mark_failed(ds4_tp *tp) {
    if (tp) atomic_store_explicit(&tp->failed, true, memory_order_release);
}

/* ------------------------------------------------------------------------
 * Gate exchange.
 * --------------------------------------------------------------------- */

/* TCP decode-gate per-link send/recv.  Header and payload go out in one
 * writev so NODELAY does not split them into two segments. */
static int tp_tcp_gate_send(ds4_tp *tp, int peer, uint32_t layer,
                            uint32_t gate, uint64_t seq) {
    ds4_tp_gate_header h = { DS4_TP_MAGIC, (uint16_t)layer, (uint16_t)gate, seq };
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { tp->slab + ds4_tp_slab_out_offset(tp, layer, gate), tp->vec_bytes },
    };
    size_t want = sizeof(h) + tp->vec_bytes;
    ssize_t w = writev(tp->data_fd[peer], iov, 2);
    if (w < 0 || (size_t)w != want) {
        if (w < 0) return 0;
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            if (!tp_write_full(tp->data_fd[peer], (char *)&h + done,
                               sizeof(h) - done)) return 0;
            done = sizeof(h);
        }
        uint64_t payload_done = done - sizeof(h);
        if (!tp_write_full(tp->data_fd[peer],
                           tp->slab + ds4_tp_slab_out_offset(tp, layer, gate) +
                               payload_done,
                           tp->vec_bytes - payload_done))
            return 0;
    }
    return 1;
}

static int tp_tcp_gate_recv(ds4_tp *tp, int peer, uint32_t layer,
                            uint32_t gate, uint64_t seq) {
    ds4_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd[peer], &ph, sizeof(ph))) return 0;
    if (ph.magic != DS4_TP_MAGIC || ph.layer != layer ||
        ph.gate != gate || ph.seq != seq) {
        fprintf(stderr,
                "ds4-tp: gate desync: got l=%u g=%u seq=%llu, want l=%u g=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, gate, (unsigned long long)seq);
        return 0;
    }
    return tp_read_full(tp->data_fd[peer],
                        tp->slab + tp_slab_in_peer_offset(tp, layer, gate, peer),
                        tp->vec_bytes);
}

int ds4_tp_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq) {
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) return tp_rdma_gate_exchange(tp, layer, gate, seq);
#endif
    /* TCP: broadcast/gather over every link.  Each link is a full-duplex
     * socket, so the symmetric write-then-read cannot deadlock (16KB per
     * direction, socket buffers absorb). */
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        if (!tp_tcp_gate_send(tp, m, layer, gate, seq)) return 0;
    }
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        if (!tp_tcp_gate_recv(tp, m, layer, gate, seq)) return 0;
    }
    if (tp->world > 2) tp_combine(tp, layer, gate);
    return 1;
}

static int tp_tcp_batch_send(ds4_tp *tp, int peer, uint32_t layer,
                             uint32_t rows, uint64_t seq) {
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC, (uint16_t)layer,
                             (uint16_t)rows, seq };
    const uint64_t bytes = (uint64_t)rows * tp->vec_bytes;
    struct iovec iov[2] = {
        { &h, sizeof(h) },
        { tp->slab + ds4_tp_slab_batch_out_offset(tp, layer), bytes },
    };
    size_t want = sizeof(h) + bytes;
    ssize_t w = writev(tp->data_fd[peer], iov, 2);
    if (w < 0) return 0;
    if ((size_t)w != want) {
        size_t done = (size_t)w;
        if (done < sizeof(h)) {
            if (!tp_write_full(tp->data_fd[peer], (char *)&h + done,
                               sizeof(h) - done)) return 0;
            done = sizeof(h);
        }
        uint64_t payload_done = done - sizeof(h);
        if (!tp_write_full(tp->data_fd[peer],
                           tp->slab + ds4_tp_slab_batch_out_offset(tp, layer) +
                               payload_done,
                           bytes - payload_done))
            return 0;
    }
    return 1;
}

static int tp_tcp_batch_recv(ds4_tp *tp, int peer, uint32_t layer,
                             uint32_t rows, uint64_t seq) {
    ds4_tp_gate_header ph;
    if (!tp_read_full(tp->data_fd[peer], &ph, sizeof(ph))) return 0;
    if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
        ph.gate != rows || ph.seq != seq) {
        fprintf(stderr,
                "ds4-tp: batch gate desync: got l=%u rows=%u seq=%llu, "
                "want l=%u rows=%u seq=%llu\n",
                ph.layer, ph.gate, (unsigned long long)ph.seq,
                layer, rows, (unsigned long long)seq);
        return 0;
    }
    return tp_read_full(tp->data_fd[peer],
                        tp->slab + tp_slab_batch_in_peer_offset(tp, layer, peer),
                        (uint64_t)rows * tp->vec_bytes);
}

/* Verify-block batch gate: broadcast/gather all block rows per layer.  The
 * payload lives in the registered slab, so RDMA sends it directly; TCP uses
 * the symmetric write-then-read fallback on each link. */
int ds4_tp_batch_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t rows,
                               uint64_t seq) {
    if (rows == 0 || rows > DS4_TP_BATCH_MAX_ROWS) return 0;
    const uint64_t bytes = (uint64_t)rows * tp->vec_bytes;
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC, (uint16_t)layer,
                             (uint16_t)rows, seq };
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) {
        /* Per-link header barrier, decode-window drain, and bulk exchange. */
        for (int m = 0; m < tp->world; m++) {
            if (m == tp->rank) continue;
            if (!tp_write_full(tp->data_fd[m], &h, sizeof(h))) return 0;
            ds4_tp_gate_header ph;
            if (!tp_read_full(tp->data_fd[m], &ph, sizeof(ph))) return 0;
            if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
                ph.gate != rows || ph.seq != seq) {
                fprintf(stderr,
                        "ds4-tp: batch gate desync: got l=%u rows=%u seq=%llu, "
                        "want l=%u rows=%u seq=%llu\n",
                        ph.layer, ph.gate, (unsigned long long)ph.seq,
                        layer, rows, (unsigned long long)seq);
                return 0;
            }
            if (!tp_rdma_drain_decode_window(tp, m)) return 0;
        }
        for (int m = 0; m < tp->world; m++) {
            if (m == tp->rank) continue;
            uint8_t *in_peer = tp->slab +
                tp_slab_batch_in_peer_offset(tp, layer, m);
            if (!tp_rdma_big_gate_exchange(tp, m,
                    tp->slab + ds4_tp_slab_batch_out_offset(tp, layer),
                    in_peer, bytes)) return 0;
        }
        if (tp->world > 2) tp_batch_combine(tp, layer, rows);
        return 1;
    }
#endif
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        if (!tp_tcp_batch_send(tp, m, layer, rows, seq)) return 0;
    }
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        if (!tp_tcp_batch_recv(tp, m, layer, rows, seq)) return 0;
    }
    if (tp->world > 2) tp_batch_combine(tp, layer, rows);
    return 1;
}

/* Prefill batch gate: RDMA uses the pipelined registered-slab path above.
 * The fallback alternates 2MB TCP write/read rounds in the same order, so
 * neither side can fill its send buffer while the peer is also only writing
 * (the 4MB socket buffers absorb one round).  Mesh: broadcast/gather over
 * every link; for world>2 the caller provides a per-peer in buffer of
 * (world-1)*bytes and the graph assembles/combines from peer regions. */
#define DS4_TP_BIG_CHUNK (2ull * 1024ull * 1024ull)

/* Canonical rank-order sum of the big-gate partials into the caller's `in`
 * (world>2 only; the per-peer regions in `in` hold the raw peer data and
 * `out` is this rank's partial).  The identical expression on every rank
 * keeps the hidden states bit-exact. */
static void tp_big_combine(ds4_tp *tp, const void *out, void *in,
                           uint64_t bytes) {
    const uint64_t words = bytes / sizeof(float);
    const float *o = (const float *)out;
    float *dst = (float *)in;
    const float *first = tp->rank == 0 ? o :
        (const float *)((uint8_t *)in +
                        (uint64_t)tp_in_peer_index(tp, 0) * bytes);
    memcpy(dst, first, bytes);
    for (int i = 1; i < tp->world; i++) {
        if (i == tp->rank) continue;
        const float *src = (const float *)((uint8_t *)in +
            (uint64_t)tp_in_peer_index(tp, i) * bytes);
        for (uint64_t k = 0; k < words; k++) dst[k] += src[k];
    }
}

static int tp_tcp_big_exchange(ds4_tp *tp, int peer,
                               const void *out, void *in, uint64_t bytes) {
    uint64_t off = 0;
    while (off < bytes) {
        const uint64_t n = bytes - off > DS4_TP_BIG_CHUNK ?
                           DS4_TP_BIG_CHUNK : bytes - off;
        if (!tp_write_full(tp->data_fd[peer], (const char *)out + off, n))
            return 0;
        if (!tp_read_full(tp->data_fd[peer], (char *)in + off, n)) return 0;
        off += n;
    }
    return 1;
}

int ds4_tp_big_gate_exchange(ds4_tp *tp, uint32_t layer, uint64_t seq,
                             const void *out, void *in, uint64_t bytes) {
    if (!out || !in || bytes == 0) return 0;
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC, (uint16_t)layer, 0xB16u, seq };
#ifdef DS4_TP_HAVE_VERBS
    if (tp->rdma_active) {
        /* Per-link header barrier and drain, then the bulk exchange. */
        for (int m = 0; m < tp->world; m++) {
            if (m == tp->rank) continue;
            if (!tp_write_full(tp->data_fd[m], &h, sizeof(h))) return 0;
            ds4_tp_gate_header ph;
            if (!tp_read_full(tp->data_fd[m], &ph, sizeof(ph))) return 0;
            if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
                ph.gate != 0xB16u || ph.seq != seq) {
                fprintf(stderr,
                        "ds4-tp: big gate desync: got l=%u tag=%x seq=%llu, want l=%u seq=%llu\n",
                        ph.layer, ph.gate, (unsigned long long)ph.seq,
                        layer, (unsigned long long)seq);
                return 0;
            }
            if (!tp_rdma_drain_decode_window(tp, m)) return 0;
        }
        for (int m = 0; m < tp->world; m++) {
            if (m == tp->rank) continue;
            uint8_t *in_peer = (uint8_t *)in +
                (tp->world > 2 ? (uint64_t)tp_in_peer_index(tp, m) * bytes : 0);
            if (!tp_rdma_big_gate_exchange(tp, m, out, in_peer, bytes))
                return 0;
        }
        if (getenv("DS4_GLM_TP_DEBUG")) {
            const float *o = (const float *)out;
            const float *i0 = (const float *)in;
            fprintf(stderr,
                    "ds4-tp: big gate l=%u seq=%llu out[0..3]=%g %g %g %g in[0..3]=%g %g %g %g\n",
                    layer, (unsigned long long)seq,
                    o[0], o[1], o[2], o[3], i0[0], i0[1], i0[2], i0[3]);
        }
        if (tp->world > 2) tp_big_combine(tp, out, in, bytes);
        return 1;
    }
#endif
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        if (!tp_write_full(tp->data_fd[m], &h, sizeof(h))) return 0;
        ds4_tp_gate_header ph;
        if (!tp_read_full(tp->data_fd[m], &ph, sizeof(ph))) return 0;
        if (ph.magic != DS4_TP_BATCH_MAGIC || ph.layer != layer ||
            ph.gate != 0xB16u || ph.seq != seq) {
            fprintf(stderr,
                    "ds4-tp: big gate desync: got l=%u tag=%x seq=%llu, want l=%u seq=%llu\n",
                    ph.layer, ph.gate, (unsigned long long)ph.seq,
                    layer, (unsigned long long)seq);
            return 0;
        }
    }
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        uint8_t *in_peer = (uint8_t *)in +
            (tp->world > 2 ? (uint64_t)tp_in_peer_index(tp, m) * bytes : 0);
        if (!tp_tcp_big_exchange(tp, m, out, in_peer, bytes)) return 0;
    }
    if (getenv("DS4_GLM_TP_DEBUG")) {
        const float *o = (const float *)out;
        const float *i0 = (const float *)in;
        fprintf(stderr,
                "ds4-tp: big gate l=%u seq=%llu out[0..3]=%g %g %g %g in[0..3]=%g %g %g %g\n",
                layer, (unsigned long long)seq,
                o[0], o[1], o[2], o[3], i0[0], i0[1], i0[2], i0[3]);
    }
    if (tp->world > 2) tp_big_combine(tp, out, in, bytes);
    return 1;
}

/* ------------------------------------------------------------------------
 * Lockstep control plane.
 * --------------------------------------------------------------------- */

typedef struct {
    uint64_t session_id;
    uint32_t count;
    uint32_t reserved;
} ds4_tp_token_command_header;

typedef struct {
    uint64_t session_id;
    int32_t value;
    uint32_t reserved;
} ds4_tp_value_command;

typedef struct {
    uint64_t session_id;
    uint64_t seq;
    int32_t token;
    uint32_t reserved;
} ds4_tp_eval_command;

typedef struct {
    uint32_t count;
    uint32_t reserved;
} ds4_tp_batch_command_header;

typedef struct {
    uint64_t prefill_session_id;
    uint32_t prompt_count;
    uint32_t item_count;
} ds4_tp_mixed_command_header;

typedef struct {
    uint64_t session_id;
    int32_t status;
    uint32_t reserved;
} ds4_tp_command_ack;

/* Control fan-out: the leader broadcasts a frame to every worker, a worker
 * sends to its single leader socket (control_fd[0]). */
static int tp_send_control(ds4_tp *tp, uint32_t type,
                           const void *payload, uint32_t bytes) {
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        if (tp->control_fd[m] < 0) continue;
        if (!tp_send_frame(tp->control_fd[m], type, payload, bytes)) return 0;
    }
    return 1;
}

static int tp_send_token_command(ds4_tp *tp, uint32_t type,
                                 uint64_t session_id, const int *tokens,
                                 uint32_t count) {
    const uint64_t bytes64 = sizeof(ds4_tp_token_command_header) +
                             (uint64_t)count * sizeof(int32_t);
    if (!tp || (!tokens && count != 0) || bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = malloc(bytes ? bytes : 1u);
    if (!payload) return 0;
    ds4_tp_token_command_header h = { session_id, count, 0 };
    memcpy(payload, &h, sizeof(h));
    int32_t *wire_tokens = (int32_t *)(payload + sizeof(h));
    for (uint32_t i = 0; i < count; i++) wire_tokens[i] = (int32_t)tokens[i];
    const int ok = tp_send_control(tp, type, payload, bytes);
    free(payload);
    return ok;
}

int ds4_tp_send_session_create(ds4_tp *tp, uint64_t session_id, int ctx_size) {
    ds4_tp_value_command msg = { session_id, (int32_t)ctx_size, 0 };
    return tp_send_control(tp, DS4_TP_FRAME_SESSION_CREATE,
                           &msg, sizeof(msg));
}

int ds4_tp_send_session_destroy(ds4_tp *tp, uint64_t session_id) {
    return tp_send_control(tp, DS4_TP_FRAME_SESSION_DESTROY,
                           &session_id, sizeof(session_id));
}

int ds4_tp_send_sync(ds4_tp *tp, uint64_t session_id,
                     const int *tokens, uint32_t n_tokens) {
    return tp_send_token_command(tp, DS4_TP_FRAME_SYNC, session_id,
                                 tokens, n_tokens);
}

int ds4_tp_send_eval(ds4_tp *tp, uint64_t session_id,
                     uint64_t seq, int token) {
    ds4_tp_eval_command msg = { session_id, seq, (int32_t)token, 0 };
    return tp_send_control(tp, DS4_TP_FRAME_EVAL, &msg, sizeof(msg));
}

int ds4_tp_send_rewind(ds4_tp *tp, uint64_t session_id, int pos) {
    ds4_tp_value_command msg = { session_id, (int32_t)pos, 0 };
    return tp_send_control(tp, DS4_TP_FRAME_REWIND,
                           &msg, sizeof(msg));
}

int ds4_tp_send_invalidate(ds4_tp *tp, uint64_t session_id) {
    return tp_send_control(tp, DS4_TP_FRAME_INVALIDATE,
                           &session_id, sizeof(session_id));
}

int ds4_tp_send_eval_batch(ds4_tp *tp, const ds4_tp_batch_item *items,
                           uint32_t count) {
    const uint64_t bytes64 = sizeof(ds4_tp_batch_command_header) +
                             (uint64_t)count * sizeof(*items);
    if (!tp || !items || count == 0 || bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = malloc(bytes);
    if (!payload) return 0;
    ds4_tp_batch_command_header h = { count, 0 };
    memcpy(payload, &h, sizeof(h));
    memcpy(payload + sizeof(h), items, (size_t)count * sizeof(*items));
    const int ok = tp_send_control(tp, DS4_TP_FRAME_EVAL_BATCH,
                                   payload, bytes);
    free(payload);
    return ok;
}

int ds4_tp_send_mixed_batch(ds4_tp *tp, uint64_t prefill_session_id,
                            const int *prompt, uint32_t prompt_count,
                            const ds4_tp_batch_item *items,
                            uint32_t count) {
    const uint64_t prompt_bytes = (uint64_t)prompt_count * sizeof(int32_t);
    const uint64_t item_bytes = (uint64_t)count * sizeof(*items);
    const uint64_t bytes64 = sizeof(ds4_tp_mixed_command_header) +
                             prompt_bytes + item_bytes;
    if (!tp || !prompt || prompt_count == 0 || !items || count == 0 ||
        bytes64 > UINT32_MAX) return 0;
    const uint32_t bytes = (uint32_t)bytes64;
    uint8_t *payload = malloc(bytes);
    if (!payload) return 0;
    ds4_tp_mixed_command_header h = {
        prefill_session_id, prompt_count, count
    };
    memcpy(payload, &h, sizeof(h));
    int32_t *wire_tokens = (int32_t *)(payload + sizeof(h));
    for (uint32_t i = 0; i < prompt_count; i++) {
        wire_tokens[i] = (int32_t)prompt[i];
    }
    memcpy(payload + sizeof(h) + prompt_bytes, items, (size_t)item_bytes);
    const int ok = tp_send_control(tp, DS4_TP_FRAME_MIXED_BATCH,
                                   payload, bytes);
    free(payload);
    return ok;
}

int ds4_tp_send_command_ack(ds4_tp *tp, uint64_t session_id, int status) {
    ds4_tp_command_ack ack = { session_id, (int32_t)status, 0 };
    return tp_send_control(tp, DS4_TP_FRAME_COMMAND_ACK,
                           &ack, sizeof(ack));
}

int ds4_tp_wait_command_ack(ds4_tp *tp, uint64_t session_id,
                            const char *operation, char *err, size_t errlen) {
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        if (tp->control_fd[m] < 0) continue;
        uint32_t type = 0, bytes = 0;
        ds4_tp_command_ack ack;
        if (!tp_read_frame_header(tp->control_fd[m], &type, &bytes) ||
            type != DS4_TP_FRAME_COMMAND_ACK || bytes != sizeof(ack) ||
            !tp_read_full(tp->control_fd[m], &ack, sizeof(ack))) {
            ds4_tp_mark_failed(tp);
            tp_set_err(err, errlen, "tp: worker %d failed during %s",
                       m, operation ? operation : "command");
            return 0;
        }
        if (ack.session_id != session_id || ack.status != 0) {
            tp_set_err(err, errlen,
                       "tp: worker %d %s failed (session %llu, status %d)",
                       m, operation ? operation : "command",
                       (unsigned long long)ack.session_id, (int)ack.status);
            return 0;
        }
    }
    return 1;
}

int ds4_tp_send_stop(ds4_tp *tp) {
    return tp_send_control(tp, DS4_TP_FRAME_STOP, NULL, 0);
}

void ds4_tp_command_free(ds4_tp_command *command) {
    if (!command) return;
    free(command->tokens);
    free(command->items);
    memset(command, 0, sizeof(*command));
    command->type = DS4_TP_FRAME_ERROR;
}

static int tp_command_decode_tokens(ds4_tp_command *command,
                                    const uint8_t *payload,
                                    uint32_t bytes,
                                    char *err, size_t errlen) {
    if (bytes < sizeof(ds4_tp_token_command_header)) return 0;
    ds4_tp_token_command_header h;
    memcpy(&h, payload, sizeof(h));
    const uint64_t want = sizeof(h) + (uint64_t)h.count * sizeof(int32_t);
    if (want != bytes) return 0;
    int *tokens = malloc(h.count ? (size_t)h.count * sizeof(*tokens) : 1u);
    if (!tokens) {
        tp_set_err(err, errlen, "tp: command token allocation failed");
        return -1;
    }
    const int32_t *wire_tokens = (const int32_t *)(payload + sizeof(h));
    for (uint32_t i = 0; i < h.count; i++) tokens[i] = wire_tokens[i];
    command->session_id = h.session_id;
    command->tokens = tokens;
    command->n_tokens = h.count;
    return 1;
}

int ds4_tp_recv_command(ds4_tp *tp, ds4_tp_command *command,
                        char *err, size_t errlen) {
    memset(command, 0, sizeof(*command));
    command->type = DS4_TP_FRAME_ERROR;
    uint32_t ftype = 0, bytes = 0;
    const int fd = tp->control_fd[0];   /* workers read from the leader */
    if (!tp_read_frame_header(fd, &ftype, &bytes)) {
        tp_set_err(err, errlen, "tp: control channel closed");
        return 0;
    }
    uint8_t *payload = NULL;
    if (bytes != 0) {
        payload = malloc(bytes);
        if (!payload || !tp_read_full(fd, payload, bytes)) {
            free(payload);
            tp_set_err(err, errlen, "tp: truncated command frame");
            return 0;
        }
    }
    int ok = 1;
    switch (ftype) {
    case DS4_TP_FRAME_SYNC:
    case DS4_TP_FRAME_VERIFY:
        ok = tp_command_decode_tokens(command, payload, bytes, err, errlen);
        break;
    case DS4_TP_FRAME_SESSION_CREATE:
    case DS4_TP_FRAME_REWIND: {
        ds4_tp_value_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->value = msg.value;
        break;
    }
    case DS4_TP_FRAME_SESSION_DESTROY:
    case DS4_TP_FRAME_INVALIDATE:
        if (bytes != sizeof(command->session_id)) { ok = 0; break; }
        memcpy(&command->session_id, payload, sizeof(command->session_id));
        break;
    case DS4_TP_FRAME_EVAL: {
        ds4_tp_eval_command msg;
        if (bytes != sizeof(msg)) { ok = 0; break; }
        memcpy(&msg, payload, sizeof(msg));
        command->session_id = msg.session_id;
        command->seq = msg.seq;
        command->value = msg.token;
        break;
    }
    case DS4_TP_FRAME_EVAL_BATCH: {
        ds4_tp_batch_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t want = sizeof(h) +
                              (uint64_t)h.count * sizeof(ds4_tp_batch_item);
        if (h.count == 0 || want != bytes) { ok = 0; break; }
        command->items = malloc((size_t)h.count * sizeof(*command->items));
        if (!command->items) { ok = -1; break; }
        memcpy(command->items, payload + sizeof(h),
               (size_t)h.count * sizeof(*command->items));
        command->n_items = h.count;
        break;
    }
    case DS4_TP_FRAME_MIXED_BATCH: {
        ds4_tp_mixed_command_header h;
        if (bytes < sizeof(h)) { ok = 0; break; }
        memcpy(&h, payload, sizeof(h));
        const uint64_t token_bytes =
            (uint64_t)h.prompt_count * sizeof(int32_t);
        const uint64_t item_bytes =
            (uint64_t)h.item_count * sizeof(ds4_tp_batch_item);
        const uint64_t want = sizeof(h) + token_bytes + item_bytes;
        if (h.prompt_count == 0 || h.item_count == 0 || want != bytes) {
            ok = 0;
            break;
        }
        command->tokens = malloc((size_t)h.prompt_count *
                                 sizeof(*command->tokens));
        command->items = malloc((size_t)h.item_count *
                                sizeof(*command->items));
        if (!command->tokens || !command->items) { ok = -1; break; }
        const int32_t *wire_tokens =
            (const int32_t *)(payload + sizeof(h));
        for (uint32_t i = 0; i < h.prompt_count; i++) {
            command->tokens[i] = wire_tokens[i];
        }
        memcpy(command->items, payload + sizeof(h) + token_bytes,
               (size_t)item_bytes);
        command->session_id = h.prefill_session_id;
        command->n_tokens = h.prompt_count;
        command->n_items = h.item_count;
        break;
    }
    case DS4_TP_FRAME_STOP:
        if (bytes != 0) ok = 0;
        break;
    default:
        ok = 0;
        break;
    }
    free(payload);
    if (ok <= 0) {
        ds4_tp_command_free(command);
        if (ok == 0) {
            tp_set_err(err, errlen, "tp: invalid command frame type %u (%u bytes)",
                       ftype, bytes);
        } else if (!err || !err[0]) {
            tp_set_err(err, errlen, "tp: command allocation failed");
        }
        return 0;
    }
    command->type = (ds4_tp_frame_type)ftype;
    return 1;
}

int ds4_tp_send_logits(ds4_tp *tp, const float *chunk, uint32_t count) {
    /* Worker ships its vocab chunk to the leader (rank 0). */
    return tp_send_control(tp, DS4_TP_FRAME_LOGITS,
                           chunk, count * sizeof(float));
}

int ds4_tp_recv_logits(ds4_tp *tp, float *dst, uint32_t count) {
    /* Leader receives each worker's chunk into dst + worker_rank * count;
     * dst is the base of the leader's full logits buffer (its own chunk is
     * already computed locally at offset 0). */
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        if (tp->control_fd[m] < 0) continue;
        uint32_t type = 0, bytes = 0;
        if (!tp_read_frame_header(tp->control_fd[m], &type, &bytes) ||
            type != DS4_TP_FRAME_LOGITS || bytes != count * sizeof(float)) {
            fprintf(stderr, "ds4-tp: bad logits frame (link %d type %u bytes %u)\n",
                    m, type, bytes);
            return 0;
        }
        if (!tp_read_full(tp->control_fd[m], dst + (uint64_t)m * count, bytes))
            return 0;
    }
    return 1;
}

int ds4_tp_send_verify(ds4_tp *tp, uint64_t session_id,
                       const int *drafts, uint32_t n) {
    return tp_send_token_command(tp, DS4_TP_FRAME_VERIFY, session_id,
                                 drafts, n);
}

int ds4_tp_send_verify_commit(ds4_tp *tp, int32_t full_accept, int32_t replay_n) {
    struct { int32_t full; int32_t replay; } msg = { full_accept, replay_n };
    return tp_send_control(tp, DS4_TP_FRAME_VERIFY_COMMIT,
                           &msg, sizeof(msg));
}

int ds4_tp_recv_verify_commit(ds4_tp *tp, int32_t *full_accept, int32_t *replay_n) {
    uint32_t type = 0, bytes = 0;
    struct { int32_t full; int32_t replay; } msg;
    const int fd = tp->control_fd[0];
    if (!tp_read_frame_header(fd, &type, &bytes) ||
        type != DS4_TP_FRAME_VERIFY_COMMIT || bytes != sizeof(msg) ||
        !tp_read_full(fd, &msg, sizeof(msg))) {
        fprintf(stderr, "ds4-tp: bad verify-commit frame (type %u bytes %u)\n",
                type, bytes);
        return 0;
    }
    *full_accept = msg.full;
    *replay_n = msg.replay;
    return 1;
}

int ds4_tp_hash_check(ds4_tp *tp, uint64_t seq, uint64_t hash, char *err, size_t errlen) {
    struct { uint64_t seq; uint64_t hash; } mine = { seq, hash }, theirs;
    if (!tp_send_control(tp, DS4_TP_FRAME_HASH, &mine, sizeof(mine))) {
        tp_set_err(err, errlen, "tp: hash send failed");
        return 0;
    }
    for (int m = 0; m < tp->world; m++) {
        if (m == tp->rank) continue;
        if (tp->control_fd[m] < 0) continue;
        uint32_t type = 0, bytes = 0;
        if (!tp_read_frame_header(tp->control_fd[m], &type, &bytes) ||
            type != DS4_TP_FRAME_HASH || bytes != sizeof(theirs) ||
            !tp_read_full(tp->control_fd[m], &theirs, sizeof(theirs))) {
            tp_set_err(err, errlen, "tp: hash recv failed");
            return 0;
        }
        if (theirs.seq != seq || theirs.hash != hash) {
            tp_set_err(err, errlen,
                       "tp: LOCKSTEP DIVERGENCE at seq %llu (rank %d): local %016llx peer %016llx",
                       (unsigned long long)seq, m,
                       (unsigned long long)hash, (unsigned long long)theirs.hash);
            return -1;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------------
 * Worker main loop.
 * --------------------------------------------------------------------- */

typedef struct {
    uint64_t id;
    ds4_session *session;
} ds4_tp_worker_session;

typedef struct {
    ds4_tp_worker_session *v;
    uint32_t len;
    uint32_t cap;
} ds4_tp_worker_sessions;

static int tp_worker_session_index(const ds4_tp_worker_sessions *sessions,
                                   uint64_t id) {
    if (!sessions || id == 0) return -1;
    for (uint32_t i = 0; i < sessions->len; i++) {
        if (sessions->v[i].id == id) return (int)i;
    }
    return -1;
}

static ds4_session *tp_worker_session_find(
        const ds4_tp_worker_sessions *sessions, uint64_t id) {
    const int index = tp_worker_session_index(sessions, id);
    return index >= 0 ? sessions->v[index].session : NULL;
}

static int tp_worker_session_add(ds4_tp_worker_sessions *sessions,
                                 uint64_t id, ds4_session *session) {
    if (!sessions || !session || id == 0 ||
        tp_worker_session_index(sessions, id) >= 0) return 0;
    if (sessions->len == sessions->cap) {
        uint32_t cap = sessions->cap ? sessions->cap * 2u : 8u;
        ds4_tp_worker_session *v =
            realloc(sessions->v, (size_t)cap * sizeof(*v));
        if (!v) return 0;
        sessions->v = v;
        sessions->cap = cap;
    }
    sessions->v[sessions->len++] = (ds4_tp_worker_session){ id, session };
    return 1;
}

static void tp_worker_session_remove(ds4_tp_worker_sessions *sessions,
                                     uint32_t index) {
    if (!sessions || index >= sessions->len) return;
    ds4_session_free(sessions->v[index].session);
    if (index + 1u < sessions->len) {
        memmove(&sessions->v[index], &sessions->v[index + 1u],
                (size_t)(sessions->len - index - 1u) * sizeof(sessions->v[0]));
    }
    sessions->len--;
}

static int tp_worker_send_logits(ds4_tp *tp, ds4_session *session,
                                 float *logits, int vocab) {
    if (!logits || vocab <= 0 || (vocab % tp->world) != 0) return 0;
    const uint32_t vchunk = (uint32_t)vocab / (uint32_t)tp->world;
    if (vchunk == 0) return 0;
    return ds4_session_copy_logits(session, logits, vocab) == vocab &&
           ds4_tp_send_logits(tp, logits + tp->rank * vchunk, vchunk);
}

int ds4_tp_worker_run(ds4_engine *engine, const ds4_tp_options *opt) {
    char err[256] = "";
    ds4_tp_identity id = {
        .gguf_bytes = ds4_engine_model_bytes(engine),
        .model_id = (uint32_t)ds4_engine_model_id(engine),
        .n_layer = (uint32_t)ds4_engine_layer_count(engine),
        .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
        .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
        .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
        .ctx_size = 0, /* adopt the leader's */
    };
    ds4_engine_tp_gate_schedule(engine,
                                &id.gate_slot_start,
                                &id.gate_slot_step,
                                &id.gates_per_token);

    ds4_tp *tp = NULL;
    if (!ds4_tp_create(&tp, opt, &id, err, sizeof(err))) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
        return 1;
    }
    if (!ds4_engine_tp_bind(engine, tp, err, sizeof(err))) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
        ds4_tp_free(tp);
        return 1;
    }
    ds4_tp_worker_sessions sessions = {0};
    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = ds4_engine_tp_vocab_split(engine) ?
        malloc((size_t)vocab * sizeof(*logits)) : NULL;
    if (ds4_engine_tp_vocab_split(engine) && !logits) {
        ds4_log(stderr, DS4_LOG_ERROR, "tp worker: logits buffer allocation failed");
        ds4_tp_free(tp);
        return 1;
    }
    ds4_log(stderr, DS4_LOG_OK, "tp worker ready for mirrored sessions");

    int rc = 0;
    ds4_tokens prompt = {0};
    while (1) {
        ds4_tp_command command;
        if (!ds4_tp_recv_command(tp, &command, err, sizeof(err))) {
            ds4_log(stderr, DS4_LOG_ERROR, "tp worker: %s", err);
            rc = 1;
            break;
        }
        if (command.type == DS4_TP_FRAME_STOP) {
            ds4_log(stderr, DS4_LOG_DEFAULT, "tp worker: leader finished");
            ds4_tp_command_free(&command);
            break;
        }

        if (command.type == DS4_TP_FRAME_SESSION_CREATE) {
            ds4_session *session = NULL;
            int status = 1;
            if (command.session_id != 0 && command.value > 0 &&
                tp_worker_session_index(&sessions, command.session_id) < 0 &&
                ds4_session_create(&session, engine, command.value) == 0 &&
                tp_worker_session_add(&sessions, command.session_id, session)) {
                /* Pay the first-submit cost before acknowledging creation so
                 * it cannot land in the leader's first timed prefill. */
                ds4_session_gpu_warmup(session);
                status = 0;
            } else if (session) {
                ds4_session_free(session);
            }
            if (!ds4_tp_send_command_ack(tp, command.session_id, status)) {
                rc = 1;
            }
            ds4_tp_command_free(&command);
            if (rc != 0) break;
            continue;
        }

        if (command.type == DS4_TP_FRAME_SESSION_DESTROY) {
            const int index = tp_worker_session_index(&sessions,
                                                      command.session_id);
            const int status = index >= 0 ? 0 : 1;
            if (index >= 0) tp_worker_session_remove(&sessions, (uint32_t)index);
            if (!ds4_tp_send_command_ack(tp, command.session_id, status)) rc = 1;
            ds4_tp_command_free(&command);
            if (rc != 0) break;
            continue;
        }

        ds4_session *session =
            tp_worker_session_find(&sessions, command.session_id);
        if (command.type != DS4_TP_FRAME_EVAL_BATCH &&
            command.type != DS4_TP_FRAME_MIXED_BATCH && !session) {
            ds4_log(stderr, DS4_LOG_ERROR,
                    "tp worker: unknown session %llu for frame %d",
                    (unsigned long long)command.session_id,
                    (int)command.type);
            ds4_tp_command_free(&command);
            rc = 1;
            break;
        }

        if (command.type == DS4_TP_FRAME_SYNC) {
            prompt.len = 0;
            for (uint32_t i = 0; i < command.n_tokens; i++) {
                ds4_tokens_push(&prompt, command.tokens[i]);
            }
            int sync_rc = ds4_session_sync(session, &prompt, err, sizeof(err));
            if (!ds4_tp_send_command_ack(tp, command.session_id, sync_rc)) {
                rc = 1;
            } else if (sync_rc != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker sync: %s", err);
                rc = 1;
            } else if (ds4_engine_tp_vocab_split(engine) &&
                       !tp_worker_send_logits(tp, session, logits, vocab)) {
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_EVAL) {
            if (ds4_session_eval(session, command.value, err, sizeof(err)) != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker eval: %s", err);
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_VERIFY) {
            int spec_rc = ds4_session_tp_spec_cycle(session, command.tokens,
                                                    (int)command.n_tokens,
                                                    err, sizeof(err));
            if (spec_rc != 0) {
                ds4_log(stderr, DS4_LOG_ERROR, "tp worker verify: %s", err);
                rc = 1;
            }
        } else if (command.type == DS4_TP_FRAME_REWIND) {
            ds4_session_rewind(session, command.value);
        } else if (command.type == DS4_TP_FRAME_INVALIDATE) {
            ds4_session_invalidate(session);
        } else if (command.type == DS4_TP_FRAME_EVAL_BATCH ||
                   command.type == DS4_TP_FRAME_MIXED_BATCH) {
            ds4_decode_item *items =
                calloc(command.n_items, sizeof(*items));
            bool mapped = items != NULL;
            for (uint32_t i = 0; mapped && i < command.n_items; i++) {
                items[i].session = tp_worker_session_find(
                    &sessions, command.items[i].session_id);
                items[i].token = command.items[i].token;
                mapped = items[i].session != NULL;
            }
            ds4_session *prefill = NULL;
            if (mapped && command.type == DS4_TP_FRAME_MIXED_BATCH) {
                prefill = tp_worker_session_find(&sessions,
                                                 command.session_id);
                mapped = prefill != NULL;
                prompt.len = 0;
                for (uint32_t i = 0; mapped && i < command.n_tokens; i++) {
                    ds4_tokens_push(&prompt, command.tokens[i]);
                }
            }
            int batch_rc = 1;
            if (mapped && command.type == DS4_TP_FRAME_EVAL_BATCH) {
                batch_rc = ds4_sessions_eval_batch(
                    items, (int)command.n_items, err, sizeof(err));
            } else if (mapped) {
                batch_rc = ds4_sessions_eval_batch_with_prefill(
                    items, (int)command.n_items, prefill, &prompt,
                    err, sizeof(err));
            }
            if (!ds4_tp_send_command_ack(tp, command.session_id, batch_rc)) {
                rc = 1;
            } else if (batch_rc != 0) {
                ds4_log(stderr, DS4_LOG_ERROR,
                        "tp worker batch: %s", err[0] ? err : "failed");
                rc = 1;
            } else if (ds4_engine_tp_vocab_split(engine)) {
                if (prefill &&
                    !tp_worker_send_logits(tp, prefill, logits, vocab)) {
                    rc = 1;
                }
                for (uint32_t i = 0; rc == 0 && i < command.n_items; i++) {
                    if (!tp_worker_send_logits(tp, items[i].session,
                                               logits, vocab)) rc = 1;
                }
            }
            free(items);
        } else {
            ds4_log(stderr, DS4_LOG_ERROR, "tp worker: unexpected frame %d",
                    (int)command.type);
            rc = 1;
        }
        ds4_tp_command_free(&command);
        if (rc != 0) break;
    }
    ds4_tokens_free(&prompt);
    while (sessions.len != 0) {
        tp_worker_session_remove(&sessions, sessions.len - 1u);
    }
    free(sessions.v);
    free(logits);
    ds4_tp_free(tp);
    return rc;
}
