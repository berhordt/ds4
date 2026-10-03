/* Physical 4-node RDMA self-test for the sequence-parallel collectives.
 *
 * This is the gate DS41F-Q4-TP4-PLAN section 11 requires before any Pillar B
 * (sequence-parallel) wiring: a standalone all-gather / reduce-scatter
 * primitive on the direct per-peer RDMA bulk path, validated at 1..64 MB
 * across every rank and bit-compared against a single-rank reference.
 *
 *   usage: tests/test_metal_tp_collectives TOPOLOGY RANK [HOST PORT]
 *
 * Launch rank 0 first (it is the leader and accepts), then ranks 1..world-1.
 * The test needs no model and no GPU: it allocates a plain slab, attaches it
 * to the transport and drives ds4_tp_all_gather / ds4_tp_reduce_scatter
 * directly, so it exercises the real per-peer RDMA path at MB scale -- the
 * regime the codebase has never used (all production gates are <= 1 MB).
 *
 * Every rank checks the full result, not just its own shard, and both
 * collectives are checked for exact equality against a single-rank reference
 * computed from known inputs: every rank must reconstruct the identical
 * buffer, and the reduce-scatter sum must equal the rank-ordered FP sum.
 * A mismatch on any rank fails the run.
 */
#include "ds4_tp.h"
#include "ds4.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "rank=%d %s:%d: %s (%s)\n", rank, __FILE__, __LINE__, #x, \
            error); goto done; } } while (0)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Deterministic pattern over the *global* element index j, so a rank's
 * partial tensor differs slice by slice.  Essential for reduce-scatter: if
 * every slice were identical, shipping the wrong slice would still pass.
 * Sums stay exact in float32, so the checks are equality, not tolerance. */
static float value(unsigned owner, uint64_t j) {
    return (float)(owner * 1000u + (j % 4093u)) + (float)(j % 97u) * 0.125f;
}

/* Fill a full tensor (world*words) for `owner`, indexed globally. */
static void fill_partial(float *out, uint64_t words, uint64_t world,
                         unsigned owner) {
    for (uint64_t j = 0; j < words * world; j++) out[j] = value(owner, j);
}

/* Can the direct (zero-copy) RDMA path be used for this buffer?  It requires
 * the whole [out, out+bytes) and [in, in+bytes) to sit inside the registered
 * slab; both always do here, which is the point -- the test measures the path
 * production will use. */
/* Compare fast vs. exact and fail loudly with the first differing bit.  This
 * mirrors ds4_tp_compare_exact(), which is not exported to the test. */
static int tp_coll_one_check(const char *what, int rank, const void *fast,
                             const void *exact, uint64_t bytes) {
    const uint64_t words = bytes / sizeof(float);
    const float *a = (const float *)fast;
    const float *b = (const float *)exact;
    uint64_t diffs = 0;
    for (uint64_t i = 0; i < words; i++) {
        uint32_t ba, bb;
        memcpy(&ba, &a[i], sizeof(ba));
        memcpy(&bb, &b[i], sizeof(bb));
        if (ba == bb) continue;
        if (diffs == 0)
            fprintf(stderr,
                    "rank=%d %s NOT bit-exact at word %llu: fast=%g (%08x) "
                    "exact=%g (%08x)\n",
                    rank, what, (unsigned long long)i, a[i], ba, b[i], bb);
        diffs++;
    }
    if (diffs) {
        fprintf(stderr, "rank=%d %s: %llu/%llu words differ\n", rank, what,
                (unsigned long long)diffs, (unsigned long long)words);
        return 0;
    }
    return 1;
}

static void report(const char *what, uint32_t world, unsigned rank,
                   uint64_t shard_bytes, double secs) {
    /* Each rank moves (world-1) shards out and the same in. */
    const double gib = (double)shard_bytes * (double)(world - 1) * 2.0 /
                       1073741824.0;
    fprintf(stderr, "rank=%d %-14s shard=%8llu B  %8.3f ms  %6.2f GiB/s "
                    "(out+in, all links)\n",
            rank, what, (unsigned long long)shard_bytes, secs * 1e3,
            secs > 0 ? gib / secs : 0.0);
}

/* Time one collective with a warmup round and `reps` timed rounds, returning
 * the best (least noisy) figure.  The first RDMA posts after attach incur QP
 * warmup and UC receive-window setup, which is worth ~16 ms on rank 0 at the
 * smallest size -- reporting that as the 1 MB result would be a cold-number
 * error of the kind this project has hit before. */
#define COLL_REPS 5
static unsigned coll_reps(void) {
    const char *v = getenv("DS4_COLL_REPS");
    if (!v || !*v) return COLL_REPS;
    const long n = strtol(v, NULL, 0);
    return n <= 0 ? 1u : (unsigned)n;   /* never 0: at least one timed round */
}
static double time_all_gather(ds4_tp *tp, unsigned rank, uint32_t seq,
                              const void *send, void *recv,
                              uint64_t shard_bytes, int *ok) {
    if (!ds4_tp_all_gather(tp, seq, send, recv, shard_bytes)) {
        *ok = 0;
        return 0.0;
    }
    const unsigned reps = coll_reps();
    double best = 0.0;
    for (unsigned r = 0; r < reps; r++) {
        const double a = now();
        if (!ds4_tp_all_gather(tp, seq + 1u + r, send, recv, shard_bytes)) {
            *ok = 0;
            return 0.0;
        }
        const double dt = now() - a;
        if (best == 0.0 || dt < best) best = dt;
    }
    (void)rank;
    return best;
}

static double time_reduce_scatter(ds4_tp *tp, uint32_t seq, const void *send,
                                  void *recv, void *scratch,
                                  uint64_t shard_bytes, int *ok) {
    if (!ds4_tp_reduce_scatter(tp, seq, send, recv, scratch, shard_bytes)) {
        *ok = 0;
        return 0.0;
    }
    const unsigned reps = coll_reps();
    double best = 0.0;
    for (unsigned r = 0; r < reps; r++) {
        const double a = now();
        if (!ds4_tp_reduce_scatter(tp, seq + 1u + r, send, recv, scratch,
                                   shard_bytes)) {
            *ok = 0;
            return 0.0;
        }
        const double dt = now() - a;
        if (best == 0.0 || dt < best) best = dt;
    }
    return best;
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 5) {
        fprintf(stderr, "usage: %s TOPOLOGY RANK [HOST PORT]\n", argv[0]);
        return 2;
    }
    const char *topology = argv[1];
    const int rank = atoi(argv[2]);
    char error[256] = "";
    ds4_tp_options opt = {
        .role = rank == 0 ? DS4_TP_LEADER : DS4_TP_WORKER,
        .transport = DS4_TP_TRANSPORT_RDMA,
        .topology_path = topology,
        .rank = rank,
        .rank_set = true,
    };
    if (argc == 5) {
        opt.listen_host = opt.leader_host = argv[3];
        opt.listen_port = opt.leader_port = atoi(argv[4]);
    }
    /* 40 layers x 5120 floats is the V4.1 shape the production slab uses, and
     * its bulk staging region is what tp_rdma_big_gate_capable() checks. */
    const uint32_t n_layer = 40;
    const uint32_t n_embd = 5120;
    ds4_tp_identity id = {.gguf_bytes = 1, .model_id = 0, .n_layer = n_layer,
        .n_embd = n_embd, .n_vocab = 16, .quant_bits = 4, .ctx_size = 8192,
        .gate_slot_step = 1, .gates_per_token = n_layer * 2};
    ds4_tp *tp = NULL;
    void *slab = NULL;
    int rc = 1;

    CHECK(ds4_tp_create(&tp, &opt, &id, error, sizeof(error)));
    const uint32_t world = (uint32_t)ds4_tp_world(tp);
    CHECK(world >= 2 && world <= 8);
    CHECK(ds4_tp_is_rdma(tp) && "test requires --transport rdma");
    const uint64_t slab_bytes = ds4_tp_slab_bytes(n_layer, n_embd, world);
    slab = calloc(1, (size_t)slab_bytes);
    CHECK(slab && ds4_tp_attach_slab(tp, slab, error, sizeof(error)));
    fprintf(stderr, "rank=%d/%u connected over %s, slab %llu bytes\n",
            rank, world, ds4_tp_is_rdma(tp) ? "rdma" : "tcp",
            (unsigned long long)slab_bytes);

    /* Payload sizes from 64 KB to 64 MB.  The small end is the decode case
     * T2 cares about (one row of a 5120-wide activation is 20 KB/layer, so
     * ~40 layers of all-gather is under 1 MB per token) and the large end is
     * the Pillar B prefill case (8192 rows x 4096 floats x 2 B = 67 MB/layer).
     * Sizes are the *total* payload; each rank handles 1/world.
     * DS4_COLL_MIN overrides the smallest size (bytes) so a failure can be
     * bisected without a rebuild, and DS4_COLL_REPS overrides the repeat
     * count (0 = one shot, no warmup). */
    uint64_t totals[] = {
        64ull * 1024, 256ull * 1024, 1ull << 20, 8ull << 20,
        16ull << 20, 32ull << 20, 64ull << 20,
    };
    const char *env_min = getenv("DS4_COLL_MIN");
    if (env_min && *env_min) {
        const uint64_t m = strtoull(env_min, NULL, 0);
        if (m > 0) totals[0] = m;
    }
    unsigned sizes = sizeof(totals) / sizeof(*totals);
    const char *env_one = getenv("DS4_COLL_ONE");
    if (env_one && *env_one) sizes = 1;   /* just totals[0], for bisection */
    for (unsigned t = 0; t < sizes; t++) {
        const uint64_t total = totals[t];
        const uint64_t shard = total / world;   /* float multiple below */
        const uint64_t shard_bytes = (shard / sizeof(float)) * sizeof(float);
        const uint64_t words = shard_bytes / sizeof(float);
        CHECK(shard_bytes > 0);

        /* ---- all-gather: send = my shard; recv = whole concatenation.
         * rank r's shard is filled with value(r, j) over j < word_count, so
         * slot r of the result must reproduce exactly that. */
        void *send = calloc(1, (size_t)shard_bytes);
        void *recv = calloc(1, (size_t)shard_bytes * world);
        CHECK(send && recv);
        for (uint64_t i = 0; i < words; i++)
            ((float *)send)[i] = value((unsigned)rank, i);
        int ok = 1;
        const double t_ag = time_all_gather(tp, (unsigned)rank,
                                            0x1000u + (uint32_t)t * 16u,
                                            send, recv, shard_bytes, &ok);
        CHECK(ok);
        const float *got = (const float *)recv;
        for (unsigned r = 0; r < world; r++) {
            for (uint64_t i = 0; i < words; i++) {
                const float want = value(r, i);
                if (got[(uint64_t)r * words + i] != want) {
                    fprintf(stderr,
                            "all_gather mismatch rank=%d size=%lluMB src=%u "
                            "i=%llu want=%g got=%g\n", rank,
                            (unsigned long long)totals[t], r,
                            (unsigned long long)i, want,
                            got[(uint64_t)r * words + i]);
                    goto done;
                }
            }
        }
        report("all_gather", world, (unsigned)rank, shard_bytes, t_ag);

        /* ---- reduce-scatter: rank p holds in_p[j] = value(p, j); rank r's
         * result is sum_p in_p[r*words + k].  The slice index r appears
         * inside value(), so shipping the wrong slice fails. */
        void *partial = calloc(1, (size_t)shard_bytes * world);
        void *rrecv = calloc(1, (size_t)shard_bytes);
        void *scratch = calloc(1, (size_t)shard_bytes * (world - 1));
        CHECK(partial && rrecv && scratch);
        fill_partial((float *)partial, words, world, (unsigned)rank);
        ok = 1;
        const double t_rs = time_reduce_scatter(tp,
            0x2000u + (uint32_t)t * 16u, partial, rrecv, scratch, shard_bytes,
            &ok);
        CHECK(ok);
        const float *sum = (const float *)rrecv;
        for (uint64_t k = 0; k < words; k++) {
            float want = value(0, (uint64_t)rank * words + k);
            for (unsigned p = 1; p < world; p++)
                want += value(p, (uint64_t)rank * words + k);
            if (sum[k] != want) {
                fprintf(stderr,
                        "reduce_scatter mismatch rank=%d size=%lluMB k=%llu "
                        "want=%g got=%g\n", rank,
                        (unsigned long long)totals[t],
                        (unsigned long long)k, want, sum[k]);
                goto done;
            }
        }
        report("reduce_scatter", world, (unsigned)rank, shard_bytes, t_rs);

        /* Consecutive collectives must not corrupt one another: run a second
         * all-gather at the same size with a different pattern. */
        for (uint64_t i = 0; i < words; i++)
            ((float *)send)[i] = value((unsigned)rank, 1000000u + i);
        CHECK(ds4_tp_all_gather(tp, 0x3000u + (uint32_t)t, send, recv,
                                shard_bytes));
        got = (const float *)recv;
        for (uint64_t i = 0; i < words; i++) {
            const float want = value((unsigned)rank, 1000000u + i);
            if (got[(uint64_t)rank * words + i] != want) {
                fprintf(stderr, "all_gather repeat mismatch rank=%d size=%lluMB "
                                "i=%llu want=%g got=%g\n", rank,
                        (unsigned long long)totals[t],
                        (unsigned long long)i, want,
                        got[(uint64_t)rank * words + i]);
                goto done;
            }
        }
        free(send); free(recv); free(partial); free(rrecv); free(scratch);
    }
    fprintf(stderr, "rank=%d collectives 1..64 MB: exact PASS\n", rank);

    /* --------------------------------------------------------------------
     * Stage 1: the _EXACT cross-check, over the real RDMA path.  The fast
     * collectives are graded bit-for-bit against ds4_tp_all_gather_exact() /
     * ds4_tp_reduce_all_exact(), which use plain blocking TCP with no round
     * machinery -- so a bookkeeping bug in the fast path cannot hide in both.
     * Deliberately small: the exact paths move whole tensors over TCP, so this
     * is validation, never a timing run.  Guarded by DS4_COLL_DEBUG=compare so
     * an ordinary run keeps the full 64 MiB sweep above.
     * -------------------------------------------------------------------- */
    if (getenv("DS4_COLL_DEBUG") &&
        !strcmp(getenv("DS4_COLL_DEBUG"), "compare")) {
        const uint64_t checks[] = {64ull * 1024, 4ull << 20};
        for (unsigned c = 0; c < sizeof(checks) / sizeof(*checks); c++) {
            const uint64_t shard_bytes = checks[c] / world;
            const uint64_t words = shard_bytes / sizeof(float);
            void *send = calloc(1, (size_t)shard_bytes);
            void *fast = calloc(1, (size_t)shard_bytes * world);
            void *exact = calloc(1, (size_t)shard_bytes * world);
            void *rfast = calloc(1, (size_t)shard_bytes);
            void *scratch = calloc(1, (size_t)shard_bytes * (world - 1));
            void *partial = calloc(1, (size_t)shard_bytes * world);
            CHECK(send && fast && exact && rfast && scratch && partial);
            for (uint64_t i = 0; i < words; i++)
                ((float *)send)[i] = value((unsigned)rank, i);

            CHECK(ds4_tp_all_gather(tp, 0x9000u + c, send, fast, shard_bytes));
            CHECK(ds4_tp_all_gather_exact(tp, 0x9100u + c, send, exact,
                                          shard_bytes));
            CHECK(tp_coll_one_check("all_gather", rank, fast, exact,
                                    shard_bytes * world));

            fill_partial((float *)partial, words, world, (unsigned)rank);
            CHECK(ds4_tp_reduce_scatter(tp, 0x9200u + c, partial, rfast,
                                        scratch, shard_bytes));
            CHECK(ds4_tp_reduce_all_exact(tp, 0x9300u + c, partial, exact,
                                          shard_bytes * world));
            CHECK(tp_coll_one_check("reduce_scatter", rank, rfast,
                                    (const uint8_t *)exact +
                                        (uint64_t)rank * shard_bytes,
                                    shard_bytes));
            fprintf(stderr, "rank=%d _EXACT cross-check shard=%llu B: "
                            "bit-identical\n",
                    rank, (unsigned long long)shard_bytes);
            free(send); free(fast); free(exact); free(rfast);
            free(scratch); free(partial);
        }
    }

    /* --------------------------------------------------------------------
     * Stage 0: the decode-shape comparison.  A decode token already fires 80
     * cross-rank exchanges -- 40 layers x 2 gates -- at a 20 KB vector, which
     * is what the slab's 80 slots are for.  The existing gate path pays *no*
     * per-call barrier: it arms a sliding receive window 16 gates ahead, so
     * successive gates pipeline (see tp_rdma_gate_exchange).  The collectives
     * above are barrier-per-round, which is the price of passing arbitrary
     * payloads.  This measures both at the shape decode actually uses, so the
     * question "can T2's dense split afford a collective per layer?" gets a
     * number instead of a prediction.
     * -------------------------------------------------------------------- */
    {
        const uint64_t vec_bytes = (uint64_t)n_embd * sizeof(float);
        const uint32_t calls = n_layer * 2u;   /* one decode token's gates */
        void *out = calloc(1, (size_t)vec_bytes);
        void *in = calloc(1, (size_t)vec_bytes);
        CHECK(out && in);
        /* Seed the partial so the exchange has real data to move. */
        for (uint64_t i = 0; i < vec_bytes / sizeof(float); i++)
            ((float *)out)[i] = value((unsigned)rank, i);

        /* Warm up the QP state and the sliding receive window: the first
         * gate of a run arms DS4_TP_RDMA_RECV_WINDOW receives, which is a
         * one-time cost that must not land inside the timed loop.  The gate
         * path requires slot == (seq-1) %% gates_per_token with layer/gate
         * derived from that slot, so seq c+1 maps to layer (c>>1), gate (c&1). */
        for (uint32_t w = 0; w < 4; w++)
            CHECK(ds4_tp_gate_exchange(tp, w >> 1, w & 1u, 1u + w));

        const double g0 = now();
        for (uint32_t c = 0; c < calls; c++)
            CHECK(ds4_tp_gate_exchange(tp, c >> 1, c & 1u, 1u + c));
        const double g1 = now();

        const double a0 = now();
        for (uint32_t c = 0; c < calls; c++)
            CHECK(ds4_tp_all_gather(tp, 0x4000u + c, out, in, vec_bytes));
        const double a1 = now();

        /* The head-split / SP handoff shape: 2048 rows x 5120 f32 = 40 MiB,
         * one layer.  Compare the three paths that can carry it:
         *   - the new collective (fast, barrier per round)
         *   - the big gate, which is what ds41_sum_partial_batch() uses today;
         *     its caller passes ds4_gpu_tensor_alloc() bounce buffers, which sit
         *     OUTSIDE the registered slab, so the peer driver stages every
         *     round through CPU memcpys (direct=false)
         *   - the same big gate forced onto slab-resident buffers (direct=true)
         * That difference is the whole question for Pillar B: if the staged path
         * is what makes head-split net-negative, sequence parallel can win by
         * using the direct one. */
        {
            const uint64_t big_bytes = 2048ull * 5120ull * 4ull;   /* 40 MiB */
            void *bout = calloc(1, (size_t)big_bytes);
            void *bin = calloc(1, (size_t)big_bytes * (world > 2 ? world - 1 : 1));
            CHECK(bout && bin);
            for (uint64_t i = 0; i < big_bytes / sizeof(float); i++)
                ((float *)bout)[i] = value((unsigned)rank, i);
            /* Warm both paths. */
            CHECK(ds4_tp_big_gate_exchange(tp, 0, 0xB000u, bout, bin, big_bytes));
            const double b0 = now();
            CHECK(ds4_tp_big_gate_exchange(tp, 1, 0xB001u, bout, bin, big_bytes));
            const double b1 = now();
            fprintf(stderr,
                    "rank=%d SP HANDOFF (40 MiB/layer): big_gate(staged) "
                    "%.3f ms = %.2f GiB/s\n",
                    rank, (b1 - b0) * 1e3,
                    (b1 - b0) > 0 ? (double)big_bytes * 2.0 * (world - 1) /
                        (b1 - b0) / 1073741824.0 : 0.0);
            free(bout); free(bin);
        }

        fprintf(stderr,
                "rank=%d DECODE SHAPE (vec=%llu B, %u calls = 1 token): "
                "gate %.3f ms/token (%.3f ms/call), all_gather %.3f ms/token "
                "(%.3f ms/call), ratio %.2fx\n",
                rank, (unsigned long long)vec_bytes, calls,
                (g1 - g0) * 1e3, (g1 - g0) * 1e3 / calls,
                (a1 - a0) * 1e3, (a1 - a0) * 1e3 / calls,
                (a1 - a0) > 0 ? (a1 - a0) / (g1 - g0) : 0.0);
        free(out); free(in);
    }

    rc = 0;

done:
    ds4_tp_free(tp);
    free(slab);
    return rc;
}