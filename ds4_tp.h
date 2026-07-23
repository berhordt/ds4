#ifndef DS4_TP_H
#define DS4_TP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4.h"

/* Tensor-parallel transport and lockstep protocol.
 *
 * Two to six ranks run the same logical model, each with a contiguous slice
 * of the routed experts resident.  Rank 0 (leader) is a normal frontend
 * session that mirrors every ds4_session_sync()/ds4_session_eval() call to
 * all workers over TCP control sockets, so every engine executes the
 * identical graph sequence.
 *
 * Inside each decoded token, partial block outputs are exchanged among all
 * ranks through a registered memory slab: two-sided RDMA SEND/RECV when RDMA
 * over Thunderbolt is available, or a full-duplex TCP exchange as fallback.
 *
 * Topology: 2-4 nodes use an all-to-all mesh (every rank exchanges directly
 * with every other rank).  5-6 nodes use a ring (each rank talks to its two
 * neighbours; partials accumulate as they circulate).  Apple Thunderbolt
 * supports at most ~3 RDMA queue pairs per port, which limits the all-to-all
 * mesh to 4 nodes.
 *
 * Layering: ds4.c calls the session-mirroring and slab entry points;
 * ds4_metal.m only ever sees ds4_tp_gate_exchange() through a callback
 * registered with the GPU gate machinery.  Nothing here touches tensors.
 */

typedef struct ds4_tp ds4_tp;
typedef struct ds4_tp_peer ds4_tp_peer;

enum {
    DS4_TP_GATE_ATTN = 0,
    DS4_TP_GATE_FFN = 1,
    DS4_TP_GATES_PER_LAYER = 2,
    DS4_TP_BATCH_MAX_ROWS = 8,
};

/* Engine identity exchanged in the hello so mismatched peers abort before
 * any inference runs. */
typedef struct {
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
    uint32_t expert_start;
    uint32_t expert_count;
    uint32_t world_size;
    uint32_t rank;
    uint32_t topology;
} ds4_tp_identity;

bool ds4_tp_enabled(const ds4_tp_options *opt);

typedef enum {
    DS4_TP_CLI_ERROR = -1,
    DS4_TP_CLI_NOT_MATCHED = 0,
    DS4_TP_CLI_MATCHED = 1,
} ds4_tp_cli_parse_result;

int ds4_tp_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        ds4_tp_options *opt,
        char *err,
        size_t errlen);
int ds4_tp_adopt_distributed_options(
        ds4_tp_options *tp,
        ds4_distributed_options *dist,
        char *err,
        size_t errlen);
void ds4_tp_usage(FILE *fp);

int ds4_tp_validate_engine_options(
        const ds4_engine_options *opt,
        char *err,
        size_t errlen);

/* Connection bring-up.  The leader listens and accepts all workers; each
 * worker dials the leader with retry.  All ranks then exchange and validate
 * identities in a mesh handshake.  Blocking; call after the engine is loaded
 * (identity needs the shape). */
int ds4_tp_create(
        ds4_tp **out,
        const ds4_tp_options *opt,
        const ds4_tp_identity *id,
        char *err,
        size_t errlen);
void ds4_tp_free(ds4_tp *tp);

int ds4_tp_rank(const ds4_tp *tp);
uint32_t ds4_tp_world_size(const ds4_tp *tp);
uint32_t ds4_tp_get_topology(const ds4_tp *tp);
bool ds4_tp_is_rdma(const ds4_tp *tp);
uint32_t ds4_tp_peer_count(const ds4_tp *tp);
const ds4_tp_peer *ds4_tp_peer_at(const ds4_tp *tp, uint32_t idx);
bool ds4_tp_failed(const ds4_tp *tp);
void ds4_tp_mark_failed(ds4_tp *tp);
uint32_t ds4_tp_expert_start(const ds4_tp *tp);
uint32_t ds4_tp_expert_count(const ds4_tp *tp);

/* Gate slab.  The engine allocates one shared GPU-visible block and hands
 * its base VA here; ds4_tp registers it with NICs (RDMA) and exchanges
 * remote keys with every peer.  Layout, all offsets from base,
 * S = n_layer * 2 slots, P = peer_count:
 *
 *   out vectors        S * vec_bytes           local GPU writes
 *   in vectors         S * vec_bytes * P       per-peer RDMA/TCP partials
 *   in seq flags       S * 8 * P              per-peer seq flags
 *   token slots        16 * P                 leader→each worker
 *   out flags staging  S * 8                  RDMA flag staging
 *   gpu flags          S * 4                  GPU-written gate-ready flags
 *   batch out          L * 8 * vec            local verify-block rows
 *   batch in           L * 8 * vec * P        per-peer verify-block rows
 *
 * vec_bytes = n_embd * 4 (f32 partials, never quantized on the wire).
 * L = n_layer. */
uint64_t ds4_tp_slab_bytes(uint32_t n_layer, uint32_t n_embd, uint32_t peer_count);
uint64_t ds4_tp_slab_out_offset(const ds4_tp *tp, uint32_t layer, uint32_t gate);
uint64_t ds4_tp_slab_in_offset(const ds4_tp *tp, uint32_t peer_idx,
                                uint32_t layer, uint32_t gate);
uint64_t ds4_tp_slab_batch_out_offset(const ds4_tp *tp, uint32_t layer);
uint64_t ds4_tp_slab_batch_in_offset(const ds4_tp *tp, uint32_t peer_idx,
                                      uint32_t layer);
uint64_t ds4_tp_slab_gpu_flags_offset(const ds4_tp *tp);
int ds4_tp_attach_slab(ds4_tp *tp, void *base, char *err, size_t errlen);

/* Exchange one gate across all peers.  Sends the local out[layer][gate] to
 * every peer and waits until all peer partials for `seq` have landed in
 * their respective in[peer][layer][gate] slots.  Called from the GPU gate
 * service thread.  Returns 0 on failure. */
int ds4_tp_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t gate, uint64_t seq);

/* Verify-block batch gate: exchange `rows` row partials for one layer across
 * all peers in one bulk operation. */
int ds4_tp_batch_gate_exchange(ds4_tp *tp, uint32_t layer, uint32_t rows,
                                uint64_t seq);

/* Prefill batch gate: arbitrary-size symmetric payload exchange across all
 * peers. */
int ds4_tp_big_gate_exchange(ds4_tp *tp, uint32_t layer, uint64_t seq,
                              const void *out, void *in, uint64_t bytes);

/* Lockstep mirroring (leader side) and worker loop primitives. */
typedef struct {
    uint64_t session_id;
    int32_t token;
    uint32_t reserved;
} ds4_tp_batch_item;

int ds4_tp_send_session_create(ds4_tp *tp, uint64_t session_id, int ctx_size);
int ds4_tp_send_session_destroy(ds4_tp *tp, uint64_t session_id);
int ds4_tp_send_sync(ds4_tp *tp, uint64_t session_id,
                     const int *tokens, uint32_t n_tokens);
int ds4_tp_send_eval(ds4_tp *tp, uint64_t session_id,
                     uint64_t seq, int token);
int ds4_tp_send_rewind(ds4_tp *tp, uint64_t session_id, int pos);
int ds4_tp_send_invalidate(ds4_tp *tp, uint64_t session_id);
int ds4_tp_send_eval_batch(ds4_tp *tp, const ds4_tp_batch_item *items,
                           uint32_t count);
int ds4_tp_send_mixed_batch(ds4_tp *tp, uint64_t prefill_session_id,
                            const int *prompt, uint32_t prompt_count,
                            const ds4_tp_batch_item *items,
                            uint32_t count);
int ds4_tp_send_command_ack(ds4_tp *tp, uint64_t session_id, int status);
int ds4_tp_wait_command_ack(ds4_tp *tp, uint64_t session_id,
                            const char *operation, char *err, size_t errlen);
int ds4_tp_send_stop(ds4_tp *tp);

typedef enum {
    DS4_TP_FRAME_ERROR = -1,
    DS4_TP_FRAME_SYNC = 1,
    DS4_TP_FRAME_EVAL = 2,
    DS4_TP_FRAME_REWIND = 3,
    DS4_TP_FRAME_INVALIDATE = 4,
    DS4_TP_FRAME_STOP = 5,
    DS4_TP_FRAME_HASH = 6,
    DS4_TP_FRAME_RDMA_INFO = 7,
    DS4_TP_FRAME_SYNC_ACK = 8,
    DS4_TP_FRAME_RDMA_READY = 9,
    DS4_TP_FRAME_LOGITS = 10,
    DS4_TP_FRAME_VERIFY = 11,
    DS4_TP_FRAME_VERIFY_COMMIT = 12,
    DS4_TP_FRAME_SESSION_CREATE = 13,
    DS4_TP_FRAME_SESSION_DESTROY = 14,
    DS4_TP_FRAME_EVAL_BATCH = 15,
    DS4_TP_FRAME_MIXED_BATCH = 16,
    DS4_TP_FRAME_COMMAND_ACK = 17,
} ds4_tp_frame_type;

typedef struct {
    ds4_tp_frame_type type;
    uint64_t session_id;
    uint64_t seq;
    int value;
    int *tokens;
    uint32_t n_tokens;
    ds4_tp_batch_item *items;
    uint32_t n_items;
} ds4_tp_command;

int ds4_tp_recv_command(
        ds4_tp *tp,
        ds4_tp_command *command,
        char *err,
        size_t errlen);
void ds4_tp_command_free(ds4_tp_command *command);

int ds4_tp_hash_check(ds4_tp *tp, uint64_t seq, uint64_t hash,
                      char *err, size_t errlen);

int ds4_tp_send_logits_half(ds4_tp *tp, const float *half, uint32_t count);
int ds4_tp_recv_logits_half(ds4_tp *tp, float *half, uint32_t count);

int ds4_tp_send_verify(ds4_tp *tp, uint64_t session_id,
                       const int *drafts, uint32_t n);
int ds4_tp_send_verify_commit(ds4_tp *tp, int32_t full_accept,
                               int32_t replay_n);
int ds4_tp_recv_verify_commit(ds4_tp *tp, int32_t *full_accept,
                               int32_t *replay_n);

int ds4_tp_worker_run(ds4_engine *engine, const ds4_tp_options *opt);

/* Mesh configuration helpers. */
int ds4_tp_load_config(const char *path, ds4_tp_options *opt,
                       char *err, size_t errlen);
int ds4_tp_discover_peers(ds4_tp *tp, char *err, size_t errlen);

#endif
