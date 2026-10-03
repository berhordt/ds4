/* Local (no cluster, no device) correctness test for the sequence-parallel
 * collectives in ds4_tp.c.
 *
 * tests/test_metal_tp_collectives is the physical 4-node RDMA self-test; this
 * is its cheap companion.  It wires `world` bare ds4_tp structs together with
 * socketpairs -- one per ordered pair, which is exactly the data_fd[] topology
 * a mesh uses -- and runs ds4_tp_all_gather / ds4_tp_reduce_scatter on every
 * rank concurrently.  That exercises the whole contribution: the per-peer
 * slice offsets, the all-gather slot assignment, the rank-ordered fold, and
 * the header barrier's desync detection.  It cannot exercise RDMA or measure
 * bandwidth; it is here so the bookkeeping is covered by a fast regression
 * run instead of a four-node cycle.
 *
 * The payloads are kept below the socket buffer so the symmetric
 * write-then-read per link cannot deadlock (the cluster test at 1..64 MB is
 * where the real pipelining is exercised).
 */
#include "../ds4_tp.c"
#include <assert.h>
#include <pthread.h>
#include <signal.h>

#define TEST_MAX_WORLD 4

typedef struct {
    ds4_tp tp;
    uint32_t rank, world;
    uint64_t shard_bytes;
    int ok, again;
} coll_rank;

/* Deterministic pattern that depends on the *global* element index, not just
 * the local offset inside a slice.  That is essential: if every slice of a
 * rank's tensor had the same contents, a reduce-scatter that shipped the
 * wrong slice would still pass.  Values are small enough that every sum below
 * is exact in float32, so the comparisons are equality, not tolerance. */
static float coll_gval(unsigned owner, uint64_t j) {
    return (float)(owner * 1000u + (j % 4093u)) + (float)(j % 97u) * 0.125f;
}

static void coll_fill_shard(float *out, uint64_t words, unsigned owner,
                            uint64_t base) {
    for (uint64_t i = 0; i < words; i++) out[i] = coll_gval(owner, base + i);
}

static void *run_collectives(void *arg) {
    coll_rank *c = arg;
    ds4_tp *tp = &c->tp;
    const uint32_t world = c->world, rank = c->rank;
    const uint64_t bytes = c->shard_bytes, words = bytes / sizeof(float);

    /* all-gather: send this rank's shard and receive the rank-ordered whole
     * buffer, whose slot r must be exactly what rank r sent. */
    float *send = calloc(1, bytes);
    float *recv = calloc(1, bytes * world);
    assert(send && recv);
    coll_fill_shard(send, words, rank, 0);
    if (!ds4_tp_all_gather(tp, 1, send, recv, bytes)) return NULL;
    for (uint32_t r = 0; r < world; r++) {
        for (uint64_t i = 0; i < words; i++) {
            const float want = coll_gval(r, i);
            if (recv[(uint64_t)r * words + i] != want) {
                fprintf(stderr, "all_gather rank=%u src=%u i=%llu want=%g got=%g\n",
                        rank, r, (unsigned long long)i, want,
                        recv[(uint64_t)r * words + i]);
                return NULL;
            }
        }
    }

    /* reduce-scatter: rank p holds a full tensor in_p[j]; rank r's result is
     * sum_p in_p[r*shard + k].  Filling in_p by global index makes each slice
     * distinct, so shipping the wrong slice fails. */
    float *partial = calloc(1, bytes * world);
    float *out = calloc(1, bytes);
    float *scratch = calloc(1, bytes * (world - 1));
    assert(partial && out && scratch);
    coll_fill_shard(partial, words * world, rank, 0);
    if (!ds4_tp_reduce_scatter(tp, 2, partial, out, scratch, bytes)) return NULL;
    for (uint64_t k = 0; k < words; k++) {
        float want = coll_gval(0, (uint64_t)rank * words + k);
        for (uint32_t p = 1; p < world; p++)
            want += coll_gval(p, (uint64_t)rank * words + k);
        if (out[k] != want) {
            fprintf(stderr, "reduce_scatter rank=%u k=%llu want=%g got=%g\n",
                    rank, (unsigned long long)k, want, out[k]);
            return NULL;
        }
    }

    /* A second all-gather on the same buffers must not carry state between
     * rounds. */
    if (!ds4_tp_all_gather(tp, 3, send, recv, bytes)) return NULL;
    for (uint64_t i = 0; i < words; i++) {
        if (recv[(uint64_t)rank * words + i] != coll_gval(rank, i)) return NULL;
    }
    c->ok = 1;
    free(send); free(recv);
    free(partial); free(out); free(scratch);
    return NULL;
}

/* The header barrier must reject a peer that is running a different
 * collective (or a different seq) instead of mixing payloads. */
static void check_desync(void) {
    int fd[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    ds4_tp a = {0}, b = {0};
    a.data_fd[1] = fd[0]; a.rank = 0; a.world = 2;
    b.data_fd[0] = fd[1]; b.rank = 1; b.world = 2;
    float x[8] = {0}, y[8] = {0};
    ds4_tp_gate_header h = { DS4_TP_BATCH_MAGIC,
        (uint16_t)DS4_TP_COLL_REDUCE_SCATTER, DS4_TP_COLL_REDUCE_SCATTER, 99 };
    assert(tp_write_full(fd[1], &h, sizeof(h)));
    assert(!ds4_tp_all_gather(&a, 99, x, y, sizeof(x)));
    close(fd[0]); close(fd[1]);
    puts("collective desync detection: PASS");
}

static void run_world(uint32_t world, uint64_t shard) {
    int fds[TEST_MAX_WORLD][TEST_MAX_WORLD];   /* fds[i][j] on rank i to j */
    for (uint32_t i = 0; i < world; i++)
        for (uint32_t j = 0; j < world; j++) fds[i][j] = -1;
    for (uint32_t i = 0; i < world; i++) {
        for (uint32_t j = i + 1; j < world; j++) {
            int p[2];
            assert(socketpair(AF_UNIX, SOCK_STREAM, 0, p) == 0);
            /* Big enough that a whole round (up to 2 MiB, the RDMA round size)
             * never blocks the symmetric write-then-read exchange. */
            for (unsigned k = 0; k < 2; k++) {
                int sz = 16 * 1024 * 1024;
                setsockopt(p[k], SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
                setsockopt(p[k], SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
            }
            fds[i][j] = p[0];
            fds[j][i] = p[1];
        }
    }
    coll_rank ranks[TEST_MAX_WORLD] = {0};
    pthread_t th[TEST_MAX_WORLD];
    for (uint32_t r = 0; r < world; r++) {
        for (uint32_t m = 0; m < world; m++) {
            if (m == r) continue;
            ranks[r].tp.data_fd[m] = fds[r][m];
        }
        ranks[r].tp.rank = (int)r;
        ranks[r].tp.world = (int)world;
        ranks[r].rank = r;
        ranks[r].world = world;
        ranks[r].shard_bytes = shard;
        assert(pthread_create(&th[r], NULL, run_collectives, &ranks[r]) == 0);
    }
    for (uint32_t r = 0; r < world; r++) {
        assert(pthread_join(th[r], NULL) == 0);
        assert(ranks[r].ok && "collective mismatch");
    }
    for (uint32_t i = 0; i < world; i++)
        for (uint32_t j = i + 1; j < world; j++) {
            close(fds[i][j]); close(fds[j][i]);
        }
    printf("world=%u shard=%llu B all-gather + reduce-scatter: PASS\n",
           world, (unsigned long long)shard);
}

int main(void) {
    const uint32_t worlds[] = {2, 3, 4};
    /* 64 KiB is one round (the decode-sized case); 5 MiB needs three rounds,
     * which is where the per-round offset and barrier bookkeeping lives.  The
     * multi-round case is included because a stale round shape there was a real
     * bug: it only appeared above one 2 MiB round. */
    const uint64_t shards[] = {64 * 1024, 5 * 1024 * 1024};
    for (unsigned wi = 0; wi < sizeof(worlds) / sizeof(*worlds); wi++)
        for (unsigned si = 0; si < sizeof(shards) / sizeof(*shards); si++)
            run_world(worlds[wi], shards[si]);
    check_desync();
    return 0;
}
