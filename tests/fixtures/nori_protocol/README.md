# Vendored Nori-Protocol contract (test-only)

Byte-copies of `schema/`, `fixtures/`, and `VERSION` from the canonical wire
contract repo:

- Source: `git@github.com:Nori-Robotics/Nori-Protocol.git`
- Copied at commit: `868bceec9335619ceadd1140d74c1de32b9dbb99` (protocol v1)
- Local checkout used: `/Users/ryan/Desktop/interlatent/robots/nori/Nori-Protocol`

Used by `tests/test_nori_protocol_conformance.py` to validate every frame
`interlatent.adapters.nori.protocol` can emit against the strict schemas
(Draft 2020-12, `additionalProperties: false`) and to replay the golden
fixtures through the inbound parser. Runtime code never reads these files —
the adapter hand-rolls the frame shapes; this copy exists so drift from the
canonical contract breaks CI, mirroring how the daemon and NoriLeLab consume
the repo.

## Re-syncing

```sh
PROTO=<path to Nori-Protocol checkout>
SDK=<path to interlatent-sdk>
rm -rf "$SDK/tests/fixtures/nori_protocol/{schema,fixtures}"
cp -R "$PROTO/schema" "$PROTO/fixtures" "$PROTO/VERSION" "$SDK/tests/fixtures/nori_protocol/"
```

Then update the commit hash above (`git -C "$PROTO" rev-parse HEAD`) and run
`pytest tests/test_nori_protocol_conformance.py`. A version bump upstream
(rename/retype/removal) is expected to fail the suite — that is the contract
working as designed; update `adapters/nori/protocol.py` deliberately.
