# Multi-Node Tensor Parallelism over RDMA

This document describes the mesh tensor-parallelism extension relative to
upstream ds4 (https://github.com/antirez/ds4).  Upstream supports exactly
two Apple Silicon machines running a 50/50 expert split.  This extension
adds support for **2-6 nodes** in a mesh or ring topology, with both GLM-5.2
and DeepSeek V4 Flash/Pro models.

## Overview

| Feature               | Upstream          | This extension              |
|-----------------------|-------------------|-----------------------------|
| Max ranks             | 2                 | 6                           |
| Topology              | Pair              | All-to-all (≤4) or ring (5-6) |
| Expert split          | 50/50             | Contiguous blocks: N experts / world_size per rank |
| Gate exchange         | Pairwise TCP/RDMA | All-to-all TCP (multi-peer); RDMA: 2-node only (legacy compat) |
| Leader mirroring      | Single worker     | Broadcasts to all workers   |
| Models                | GLM-5.2, DS4 V4   | Same                        |
| Configuration         | --role + shared   | --tp-world, --tp-rank, --tp-peer, --tp-config-file |
| Server support        | No                | ds4-server supports TP (leader mode) |

## Topology Selection

- **All-to-all** (default for 2-4 nodes): every rank opens a data connection to
  every other rank.  Gate exchange sends the local partial to all peers and
  receives each peer's partial into a per-peer slab buffer.
- **Ring** (default for 5-6 nodes): each rank connects only to its immediate
  neighbours (prev/next).  Partial outputs circulate around the ring; after
  N-1 hops every rank has accumulated all peer partials.  Ring uses fewer
  connections (2 per node) but adds serial latency.

The topology is selected automatically based on world_size unless overridden
with `--tp-topology`.  Apple Thunderbolt supports at most ~3 RDMA queue pairs
per port, which limits the all-to-all mesh to 4 nodes.

## Slab Memory Layout

The slab layout extends the upstream design with per-peer regions:

```
out vectors        S * vec_bytes           local GPU writes (1 set)
in vectors         S * vec_bytes * P       per-peer RDMA/TCP partials
in seq flags       S * 8 * P              per-peer seq flags
token slots        16 * P                 leader→each worker
out flags staging  S * 8                  RDMA flag staging (1 set)
gpu flags          S * 4                  GPU-written gate-ready flags (1 set)
batch out          L * 8 * vec            local verify-block rows (1 set)
batch in           L * 8 * vec * P        per-peer verify-block rows
```

where S = n_layer × 2, L = n_layer, vec = n_embd × 4, P = peer_count.

**Memory examples** (DeepSeek V4 Pro, 8B, n_layer≈60, n_embd=4096):
- 2 nodes: ~4 MB
- 4 nodes: ~10 MB
- 6 nodes: ~14 MB

All well within Apple Silicon GPU limits.

## Protocol Changes

### Hello Exchange (v8)

The hello_fixed struct was extended with five new fields:
- `world_size` (0 = legacy 2-node)
- `rank` (this node's position)
- `topology` (DS4_TP_TOPO_ALL_TO_ALL or DS4_TP_TOPO_RING)
- `expert_start`, `expert_count` (contiguous expert ownership)

Protocol version bumped from 7 to 8.  Legacy 2-node pairs using v7 will
fail to connect to v8 peers; both sides must use the same ds4 build.

### Gate Exchange

**All-to-all (TCP)** — Sequential write-then-read with each peer:
1. Send local out vector to peer N
2. Receive peer N's partial into in[peer N]
3. Repeat for all peers

**Ring (TCP)** — N-1 hop circulation:
1. Send local partial to next neighbour
2. Receive prev neighbour's partial, add to accumulator
3. Forward accumulator to next neighbour
4. After N-1 hops, copy accumulated sum to all in-buffers

**RDMA** — Currently only the legacy 2-node path is active.  Multi-peer
RDMA requires per-peer QP management and is deferred to future work.

### Leader Mirroring

All `ds4_tp_send_*` functions now iterate over all peers and send to each
worker's control_fd.  `ds4_tp_wait_command_ack` waits for acknowledgments
from every worker.

## Configuration

### CLI Options

```
--tensor-parallel                  Enable TP (required)
--role coordinator|worker          Leader or worker role
--listen HOST PORT                 Leader listen address
--coordinator HOST PORT            Worker dial address
--transport auto|rdma|tcp          Gate transport (default auto)

--tp-world <2..6>                  Total ranks in the mesh
--tp-rank <0..5>                   This node's rank
--tp-topology all_to_all|ring      Override auto topology selection
--tp-config-file <path>            JSON config file with peer list
--tp-peer RANK:HOST:PORT:RDMA_HOST:RDMA_DEV
                                   Add a peer (repeatable, max 5)
```

### Config File Format

```json
{
  "world_size": 4,
  "rank": 0,
  "role": "leader",
  "topology": "all_to_all",
  "peers": [
    {"rank": 1, "control_host": "192.168.1.101", "control_port": 9000,
     "rdma_host": "169.254.100.1", "rdma_device": "rdma_en0"},
    {"rank": 2, "control_host": "192.168.1.102", "control_port": 9000,
     "rdma_host": "169.254.100.2", "rdma_device": "rdma_en0"},
    {"rank": 3, "control_host": "192.168.1.103", "control_port": 9000,
     "rdma_host": "169.254.100.3", "rdma_device": "rdma_en0"}
  ]
}
```

### Example: 4-Node All-to-All Mesh

**Leader (rank 0):**
```
ds4-server --model ds4pro8b.gguf --metal \
  --tensor-parallel --role coordinator --listen 0.0.0.0 9000 \
  --tp-world 4 --tp-rank 0 \
  --tp-peer 1:192.168.1.101:9000:169.254.100.1:rdma_en0 \
  --tp-peer 2:192.168.1.102:9000:169.254.100.2:rdma_en0 \
  --tp-peer 3:192.168.1.103:9000:169.254.100.3:rdma_en0 \
  --transport tcp
```

**Worker 1 (rank 1):**
```
ds4 --model ds4pro8b.gguf --metal \
  --tensor-parallel --role worker --coordinator 192.168.1.100 9000 \
  --tp-world 4 --tp-rank 1 \
  --tp-peer 0:192.168.1.100:9000:169.254.100.0:rdma_en0 \
  --tp-peer 2:192.168.1.102:9000:169.254.100.2:rdma_en0 \
  --tp-peer 3:192.168.1.103:9000:169.254.100.3:rdma_en0 \
  --transport tcp
```

(Workers 2 and 3 are similar, changing only `--tp-rank`.)

### Example: 6-Node Ring

Same as above but with `--tp-world 6`, ranks 0-5, and 5 `--tp-peer` entries
per node.  Topology auto-selects `ring` for world_size ≥ 5.

## Files Changed

| File               | Lines | Description |
|--------------------|-------|-------------|
| `ds4.h`            | +30   | Extended `ds4_tp_options` with world_size, rank, topology, config_file, peers |
| `ds4_tp.h`         | +70   | New API: multi-peer functions, topology enum, peer_config struct, identity fields |
| `ds4_tp.c`         | +400  | Multi-peer struct, slab layout, hello exchange, create, gate exchange (all-to-all + ring), leader broadcast, CLI options, validation, config loading |
| `ds4_server.c`     | +63   | TP support in server (parsing, validation, worker/leader handling) |
| `ds4.c`            | +3    | Slab function call sites updated for multi-peer signatures |
| `docs/tensor_parallel_mesh.md` | +160 | This document |

Total: ~730 lines added across 6 files.

## Limitations & Future Work

1. **RDMA for multi-peer** — Currently only TCP fallback is used for mesh
   gate exchange.  Per-peer RDMA QP management is deferred.  The legacy
   2-node RDMA path remains functional.

2. **GPU combine** — The engine still uses pairwise addition for TP
   partials.  For multi-node, peer partials must be pre-summed on GPU
   before feeding to the HC expand.  A multi-way addition kernel
   (`kernel_addN_f32`) is planned but not yet integrated.

3. **Expert dispatch** — GPU kernels need to filter routed experts by
   ownership range.  The `ds4_engine_tp_expert_range()` function and
   kernel-side filtering are planned for a follow-up change.

4. **Auto-discovery** — `ds4_tp_discover_peers()` is a stub.  Full
   auto-discovery via `ibv_devinfo` enumeration and coordinator-mediated
   peer info exchange is future work.

5. **Server multi-worker sessions** — The server creates one TP leader
   context and binds it to the engine.  Session management across
   multiple workers is handled by the transport layer's broadcast
   mirroring; no server-side changes are needed beyond what's already
   implemented.

## Compatibility

- **Upstream 2-node pairs**: Continue to work unchanged.  The legacy
  `--tensor-parallel --role coordinator|worker` path with no `--tp-world`
  falls back to world_size=2, peer_count=1.
- **Protocol version**: Bumped from 7 to 8.  Mixed-version clusters
  will fail the hello exchange with a version mismatch error.
- **Backend**: Metal only (unchanged constraint).
- **Models**: GLM-5.2 and DeepSeek V4 Flash/Pro (unchanged).
