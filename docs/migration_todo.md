# Migration TODO

The enhanced network stack contains numerous stubs. The following tasks remain
before production hardening can be considered complete:

- **QUIC/TCP Transport** – `NetworkNode._init_transport` needs a real transport
  implementation using `aioquic` with TLS 1.3 and TCP fallback.
- **Kademlia DHT** – `NetworkNode._init_dht` currently raises
  `NotImplementedError`. Integrate a persistent DHT implementation with signed
  records.
- **NAT Traversal** – `p2p.nat` contains placeholders for STUN/TURN setup and UDP
  hole punching. Implement and test across different NAT types.
- **Adaptive Routing** – Routing metrics and ML path prediction are not wired
  into `NetworkNode._routing_update_loop`.
- **DNSSEC Validation** – `dns/overlay.py` signs records but verification is
  missing.
- **Key Rotation** – `security_hardening` generates keys but periodic rotation
  and certificate renewal are not yet automated.

These gaps should be tracked and resolved in future iterations.
