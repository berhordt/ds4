#!/usr/bin/env python3
"""
gen-mesh.py — generate the ds4 tensor-parallel mesh topology file
(mesh.txt) automatically from the live cluster.

The ds4 N-node mesh runs over non-bridged Thunderbolt links.  Each node
has one link-local (169.254/16) IPv4 address per peer, on its own
interface, and those addresses change on every reboot, so mesh.txt must
be regenerated whenever the cluster comes back up.  This utility does
that automatically:

  * ssh to every node (in --hosts rank order; rank 0 must be the leader)
  * read each node's link-local addresses and interface MACs (ifconfig)
  * map every link to its peer from the node's ARP table, trying in
    order: peer mDNS name, peer MAC, peer IP
  * fall back to ping -S probing (bound to the local link address) for
    any link ARP cannot resolve — a peer's link-local address is only
    reachable on the direct cable, so this is authoritative
  * validate that the mesh is complete and symmetric
  * write mesh.txt (optionally scp it to all nodes with --deploy)

The output format matches ds4_tp_topology_load() in ds4_tp.c: link i of
node R connects to peer (R+i+1) % world, and the port on a node's link
is the port that node listens on for that link.

Examples:
  python3 gen-mesh.py --hosts nemo,gloria,dory,fanny
  python3 gen-mesh.py --hosts nemo,gloria,dory,fanny --port 9911 --deploy
  python3 gen-mesh.py --hosts nemo,gloria,dory,fanny --check mesh.txt
  python3 gen-mesh.py --hosts nemo,gloria,dory,fanny --verify
"""

import argparse
import concurrent.futures
import datetime
import os
import re
import subprocess
import sys

LINKLOCAL_RE = re.compile(r"^169\.254\.\d+\.\d+$")
IFACE_RE = re.compile(r"^(\S+):\s+flags=")
ARP_RE = re.compile(
    r"^(\S+)\s+\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+([0-9a-fA-F:]+)\s+on\s+(\S+)"
)

# arp -a -n: -n disables the reverse mDNS lookup that makes plain
# `arp -a` take ~30s per node; the mDNS peer name is a convenience only
# (MAC/IP matching on complete entries is authoritative).
REMOTE_CMD = (
    "echo 'HOST '$(hostname -s | tr 'A-Z' 'a-z' | sed 's/\\.local$//'); "
    "ifconfig -a; "
    "echo 'ARP-BEGIN'; arp -a -n"
)


# ---------------------------------------------------------------------------
# node data collection
# ---------------------------------------------------------------------------

def ssh(host, command, timeout=10):
    """Run a remote command via ssh.  Raise on failure."""
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=%d" % timeout,
         host, command],
        capture_output=True, text=True, timeout=timeout + 30)
    if proc.returncode != 0:
        raise RuntimeError("ssh %s failed: %s" % (host, proc.stderr.strip() or
                                                  proc.stdout.strip()))
    return proc.stdout


def parse_ifconfig(text):
    """Return {iface: (mac, ip)} for interfaces with a 169.254/16 address."""
    links = {}
    iface = None
    mac = None
    for line in text.splitlines():
        m = IFACE_RE.match(line)
        if m:
            iface = m.group(1)
            mac = None
            continue
        if iface is None:
            continue
        line = line.strip()
        if line.startswith("ether "):
            mac = line.split()[1]
        elif line.startswith("inet ") and not line.startswith("inet6 "):
            toks = line.split()
            if len(toks) >= 2 and LINKLOCAL_RE.match(toks[1]):
                if mac:
                    links[iface] = (mac, toks[1])
    return links


def parse_arp(text):
    """Return [(name, ip, mac, iface)] for 169.254 ARP entries."""
    entries = []
    in_arp = False
    for line in text.splitlines():
        if line.startswith("ARP-BEGIN"):
            in_arp = True
            continue
        if not in_arp:
            continue
        m = ARP_RE.match(line)
        if not m:
            continue
        name, ip, mac, iface = m.groups()
        if not LINKLOCAL_RE.match(ip):
            continue
        entries.append((name, ip, mac.lower(), iface))
    return entries


def collect_node(host, timeout):
    """ssh to host and return its link-local interface + ARP data."""
    out = ssh(host, REMOTE_CMD, timeout)
    hostname = None
    for line in out.splitlines():
        if line.startswith("HOST "):
            hostname = line[5:].strip()
            break
    if hostname is None:
        hostname = host.split("@")[-1].split(".")[0].lower()
    links = parse_ifconfig(out)
    arp = parse_arp(out)
    return {"host": host, "hostname": hostname, "links": links, "arp": arp}


# ---------------------------------------------------------------------------
# ping probing (authoritative fallback + verification)
# ---------------------------------------------------------------------------

def run_ping_matrix(host, src_ips, dst_ips, timeout):
    """ssh to host and ping every dst from every src (bound with -S).

    Returns a set of (src, dst) pairs that replied.  One ssh round trip
    per node; each ping is a single 1s probe (link-local replies are
    sub-millisecond, only wrong-link probes burn the timeout).
    """
    if not src_ips or not dst_ips:
        return set()
    sh = " ".join("'%s'" % s for s in src_ips)
    dh = " ".join("'%s'" % d for d in dst_ips)
    cmd = (
        "for s in %s; do for d in %s; do "
        "ping -S $s -c 1 -t 1 $d >/dev/null 2>&1 && echo \"OK $s $d\"; "
        "done; done; true" % (sh, dh))
    out = ssh(host, cmd, timeout + max(30, 3 * len(src_ips) * len(dst_ips)))
    ok = set()
    for line in out.splitlines():
        toks = line.split()
        if len(toks) == 3 and toks[0] == "OK":
            ok.add((toks[1], toks[2]))
    return ok


def verify_links(hosts, nodes, resolved, timeout):
    """Ping each resolved link from its node, bound to the link source.

    Returns per-node dict {peer: peer_ip_that_replied} (or None).
    """
    world = len(hosts)
    results = {}

    def one(r):
        cmds = []
        for peer, (_iface, ip) in resolved[r].items():
            peer_ips = [oip for _m, oip in nodes[peer]["links"].values()]
            for pip in peer_ips:
                cmds.append(
                    "ping -S %s -c 1 -t 1 %s >/dev/null 2>&1 && "
                    "echo OK %s %s" % (ip, pip, ip, pip))
        out = ssh(hosts[r], "; ".join(cmds) + "; true",
                  timeout + max(30, 5 * len(cmds)))
        ok = set()
        for line in out.splitlines():
            toks = line.split()
            if len(toks) == 3 and toks[0] == "OK":
                ok.add((toks[1], toks[2]))
        return r, ok

    with concurrent.futures.ThreadPoolExecutor(max_workers=world) as ex:
        futs = {ex.submit(one, r): r for r in range(world)}
        for fut in concurrent.futures.as_completed(futs):
            r, ok = fut.result()
            results[r] = ok
    return results


# ---------------------------------------------------------------------------
# peer resolution
# ---------------------------------------------------------------------------

def normalize_name(name):
    return name.lower().rstrip(".local")


def resolve_peer_from_arp(rank, iface, nodes, host_ranks):
    """Return (peer_rank, method) for R's interface, or (None, None)."""
    arp_entries = [e for e in nodes[rank]["arp"] if e[3] == iface]
    for name, pip, pmac, _piface in arp_entries:
        # method 1: the ARP entry carries the peer's mDNS name
        if name != "?":
            key = normalize_name(name)
            if key in host_ranks:
                return host_ranks[key], "arp-name"
        # method 2: match the peer MAC to another node's interface
        for other in range(len(nodes)):
            if other == rank:
                continue
            for omac, _oip in nodes[other]["links"].values():
                if omac == pmac:
                    return other, "arp-mac"
        # method 3: match the peer link-local IP to another node
        for other in range(len(nodes)):
            if other == rank:
                continue
            for _omac, oip in nodes[other]["links"].values():
                if oip == pip:
                    return other, "arp-ip"
    return None, None


def resolve_all(nodes, hosts, timeout, verbose, allow_ping):
    """Return [ {peer_rank: (iface, ip)}, ... ] for every node."""
    world = len(nodes)
    host_ranks = {normalize_name(hosts[r].split("@")[-1]): r
                  for r in range(world)}
    resolved = []

    for r in range(world):
        peer_map = {}
        pending = []
        for iface, (_mac, ip) in nodes[r]["links"].items():
            peer, method = resolve_peer_from_arp(r, iface, nodes, host_ranks)
            if peer is None:
                pending.append((iface, ip))
            else:
                peer_map[peer] = (iface, ip)
                if verbose:
                    print("  node %d %s %-14s -> rank %d (%s)" %
                          (r, iface, ip, peer, method))
        # authoritative ping fallback for unresolved links (only probe
        # the other nodes' link-local addresses, never our own)
        if allow_ping and pending:
            src_ips = [ip for _i, ip in pending]
            dsts = sorted({oip for other in range(world) if other != r
                           for _m, oip in nodes[other]["links"].values()})
            ok = run_ping_matrix(hosts[r], src_ips, dsts, timeout)
            for iface, ip in pending:
                reached = {p for p in range(world) if p != r
                           for _m, oip in nodes[p]["links"].values()
                           if (ip, oip) in ok}
                if len(reached) == 1:
                    peer = reached.pop()
                    peer_map[peer] = (iface, ip)
                    if verbose:
                        print("  node %d %s %-14s -> rank %d (ping)" %
                              (r, iface, ip, peer))
                elif len(reached) > 1:
                    print("  WARNING: node %d %s %s reached multiple peers %s"
                          % (r, iface, ip, sorted(reached)))
                else:
                    print("  WARNING: node %d %s %s: no peer found (ping)"
                          % (r, iface, ip))
        if len(peer_map) != len(nodes[r]["links"]):
            print("  WARNING: node %d resolved %d/%d links" %
                  (r, len(peer_map), len(nodes[r]["links"])))
        resolved.append(peer_map)
    return resolved


# ---------------------------------------------------------------------------
# validation and output
# ---------------------------------------------------------------------------

def validate_mesh(nodes, hosts, resolved):
    world = len(nodes)
    problems = []
    for r in range(world):
        peers = set(resolved[r].keys())
        expected = set(range(world)) - {r}
        if peers != expected:
            problems.append(
                "node %d (%s): links to %s, expected %s" %
                (r, hosts[r], sorted(peers), sorted(expected)))
        for p in sorted(peers):
            if r not in resolved[p]:
                problems.append(
                    "asymmetric: node %d links to %d but not vice versa" %
                    (r, p))
    if problems:
        for p in problems:
            print("ERROR: " + p)
        return False
    return True


def write_mesh(path, hosts, resolved, port):
    world = len(hosts)
    lines = []
    lines.append("# ds4 TP mesh topology - generated %s" %
                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("# hosts (rank order): %s" % ", ".join(hosts))
    lines.append("# link-local IPv4 changes on every reboot; regenerate "
                 "this file after each cluster reboot")
    lines.append("# link i of node R connects to peer (R+i+1)%%world; "
                 "port is the local listen port for that link")
    lines.append("world %d" % world)
    for r in range(world):
        toks = ["node", str(r)]
        for i in range(world - 1):
            peer = (r + i + 1) % world
            iface, ip = resolved[r][peer]
            toks.append(ip)
            toks.append(str(port))
        lines.append(" ".join(toks))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s" % path)


def parse_mesh(path):
    """Parse a mesh.txt the same way ds4_tp_topology_load() does."""
    world = None
    nodes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if toks[0] == "world":
                world = int(toks[1])
            elif toks[0] == "node":
                r = int(toks[1])
                links = []
                for i in range(world - 1):
                    links.append((toks[2 + 2 * i], int(toks[3 + 2 * i])))
                nodes[r] = links
    if world is None or len(nodes) != world:
        raise ValueError("malformed mesh file %s" % path)
    return world, nodes


def check_mesh(path, hosts, resolved):
    world, nodes = parse_mesh(path)
    changed = []
    for r in range(world):
        for i in range(world - 1):
            peer = (r + i + 1) % world
            file_ip = nodes[r][i][0]
            live_ip = resolved[r][peer][1]
            if file_ip != live_ip:
                changed.append((r, i, peer, file_ip, live_ip))
    if not changed:
        print("OK: %s matches the live cluster" % path)
        return 0
    print("STALE: %s differs from the live cluster:" % path)
    for r, i, peer, file_ip, live_ip in changed:
        print("  node %d link %d (peer %d): file %s != live %s" %
              (r, i, peer, file_ip, live_ip))
    return 1


def validate_ring(nodes, hosts, resolved):
    """Ring mode: every node must be directly linked to its ring next and
    prev neighbours (rank order).  Extra links are allowed but unused."""
    world = len(hosts)
    problems = []
    for r in range(world):
        peers = set(resolved[r].keys())
        nxt = (r + 1) % world
        prv = (r + world - 1) % world
        for need in (nxt, prv):
            if need not in peers:
                problems.append(
                    "node %d (%s): no link to ring neighbour %d" %
                    (r, hosts[r], need))
            elif r not in resolved[need]:
                problems.append(
                    "asymmetric: node %d links to %d but not vice versa" %
                    (r, need))
    if problems:
        for p in problems:
            print("ERROR: " + p)
        return False
    return True


def write_ring(path, hosts, resolved, port):
    """Write the ring topology file (peer-labeled links)."""
    world = len(hosts)
    lines = []
    lines.append("# ds4 TP ring topology - generated %s" %
                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("# hosts (rank order): %s" % ", ".join(hosts))
    lines.append("# ring order: %s" % " -> ".join(
        [str(r) for r in range(world)] + ["0"]))
    lines.append("# link-local IPv4 changes on every reboot; regenerate "
                 "this file after each cluster reboot")
    lines.append("# each node lists its ring neighbours as 'peer host port'")
    lines.append("world %d" % world)
    for r in range(world):
        nxt = (r + 1) % world
        prv = (r + world - 1) % world
        nxt_ip = resolved[r][nxt][1]
        prv_ip = resolved[r][prv][1]
        lines.append("node %d %d %s %d %d %s %d" %
                     (r, nxt, nxt_ip, port, prv, prv_ip, port))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s (ring, %d nodes)" % (path, world))


def check_ring(path, hosts, resolved):
    """Check a ring file against the live cluster (peer-labeled)."""
    world = None
    nodes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if toks[0] == "world":
                world = int(toks[1])
            elif toks[0] == "node":
                r = int(toks[1])
                links = {}
                i = 2
                while i + 2 < len(toks):
                    links[int(toks[i])] = toks[i + 1]
                    i += 3
                nodes[r] = links
    if world is None or len(nodes) != world:
        raise ValueError("malformed ring file %s" % path)
    changed = []
    for r in range(world):
        nxt = (r + 1) % world
        prv = (r + world - 1) % world
        for need in (nxt, prv):
            file_ip = nodes[r].get(need)
            live_ip = resolved[r][need][1]
            if file_ip != live_ip:
                changed.append((r, need, file_ip, live_ip))
    if not changed:
        print("OK: %s matches the live cluster" % path)
        return 0
    print("STALE: %s differs from the live cluster:" % path)
    for r, peer, file_ip, live_ip in changed:
        print("  node %d link to %d: file %s != live %s" %
              (r, peer, file_ip, live_ip))
    return 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate the ds4 mesh topology file from the live "
                    "cluster (link-local IPs change on reboot).")
    ap.add_argument("--hosts", default="nemo,gloria,dory,fanny",
                    help="comma-separated node names in rank order "
                         "(rank 0 must be the ds4-server leader); "
                         "use user@host if ssh needs a user")
    ap.add_argument("--port", type=int, default=9911,
                    help="listen port on every link (default 9911)")
    ap.add_argument("--out", default="mesh.txt",
                    help="output path (default mesh.txt)")
    ap.add_argument("--check", metavar="PATH",
                    help="instead of writing, validate PATH against the "
                         "live cluster; exit 0 if current, 1 if stale")
    ap.add_argument("--verify", action="store_true",
                    help="ping-probe every resolved link and report "
                         "connectivity")
    ap.add_argument("--deploy", action="store_true",
                    help="scp the generated mesh.txt to every node")
    ap.add_argument("--deploy-dir", default="~/github/berhordt/ds4",
                    help="remote directory for --deploy (default "
                         "~/github/berhordt/ds4)")
    ap.add_argument("--timeout", type=int, default=10,
                    help="ssh/connect timeout in seconds (default 10)")
    ap.add_argument("--no-ping", action="store_true",
                    help="disable the ping fallback for unresolved links")
    ap.add_argument("--ring", action="store_true",
                    help="generate a ring topology (peer-labeled links to "
                         "each node's next/prev neighbours) instead of the "
                         "fully-connected mesh")
    ap.add_argument("--verbose", action="store_true", help="verbose output")
    args = ap.parse_args()

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if not 2 <= len(hosts) <= 8:
        print("ERROR: world size must be 2..8 (got %d hosts)" % len(hosts))
        return 2

    print("collecting link-local topology from %d nodes..." % len(hosts))
    nodes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as ex:
        futs = {ex.submit(collect_node, h, args.timeout): h for h in hosts}
        for fut in concurrent.futures.as_completed(futs):
            h = futs[fut]
            try:
                nodes.append(fut.result())
                print("  %s: %d link-local interface(s), %d ARP entries" %
                      (h, len(fut.result()["links"]),
                       len(fut.result()["arp"])))
            except Exception as e:
                print("ERROR collecting %s: %s" % (h, e))
                return 2
    nodes.sort(key=lambda n: hosts.index(n["host"]))

    # sanity: each node must report exactly world-1 link-local interfaces
    world = len(hosts)
    for r, n in enumerate(nodes):
        if not args.ring and len(n["links"]) != world - 1:
            print("WARNING: node %d (%s) has %d link-local interfaces, "
                  "expected %d" % (r, n["host"], len(n["links"]), world - 1))

    print("resolving links...")
    resolved = resolve_all(nodes, hosts, args.timeout, args.verbose,
                           not args.no_ping)

    if args.verify:
        print("verifying connectivity with ping probes...")
        ok_by_node = verify_links(hosts, nodes, resolved, args.timeout)
        for r in range(world):
            ok = ok_by_node[r]
            for peer, (iface, ip) in resolved[r].items():
                hits = [pip for _m, pip in nodes[peer]["links"].values()
                        if (ip, pip) in ok]
                if hits:
                    print("  node %d %s %-14s -> rank %d %-14s OK" %
                          (r, iface, ip, peer, hits[0]))
                else:
                    print("  node %d %s %-14s -> rank %d NO REPLY" %
                          (r, iface, ip, peer))

    if args.ring:
        if not validate_ring(nodes, hosts, resolved):
            return 2
        if args.check:
            return check_ring(args.check, hosts, resolved)
        write_ring(args.out, hosts, resolved, args.port)
    else:
        if not validate_mesh(nodes, hosts, resolved):
            return 2
        if args.check:
            return check_mesh(args.check, hosts, resolved)
        write_mesh(args.out, hosts, resolved, args.port)

    if args.deploy:
        for h in hosts:
            dest = "%s:%s/%s" % (h, args.deploy_dir.rstrip("/"),
                                 os.path.basename(args.out))
            proc = subprocess.run(["scp", "-o", "BatchMode=yes",
                                   "-o", "ConnectTimeout=%d" % args.timeout,
                                   args.out, dest],
                                  capture_output=True, text=True)
            if proc.returncode == 0:
                print("  deployed %s to %s" % (args.out, h))
            else:
                print("  ERROR deploying to %s: %s" %
                      (h, proc.stderr.strip()))
                return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
