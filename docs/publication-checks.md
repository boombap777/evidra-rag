# Publication checks — 2026-09-08

Scope: local campus-project source release, not production readiness or hosted CI completion.

## Verified

1363 passed, 37 explicit skips; 73.29% whole-project coverage. The existing core-only 80% experimental gate is a separate scope. The 37 skips include two full benchmark/model-artifact checks; they are not counted as passed claims.

Desktop browser: load three original synthetic examples, query MCP tool boundaries, view three source-labelled snippets and measured retrieval stages. Fourteen UI/component tests passed after the navigation update.

Each verification copy was exported without the original virtual environment or private runtime
files, and installed its own locked dependencies. Publication inventories contain file hashes,
size checks and a bounded common-token scan. The original repositories were not committed or
uploaded; an isolated Git fixture was used only to exercise Teams HEAD-bound EDD.

## Boundaries

Default demo uses local_hash, not Qwen3 semantic embeddings. Full Stack Overflow benchmark and live model integrations require explicit setup/opt-in; no new full retrieval experiment was claimed.

Hosted GitHub Actions have not run because no upload/push was authorized. Desktop browser
rendering and interactions were inspected. Responsive CSS is implemented, but a physical mobile
or effective fixed-width device run was not completed: the browser viewport override continued
to report 1265 px. This is not claimed as mobile-device acceptance.

No secret scanner can establish that all arbitrary text or historical Git objects are safe.
The recommended source release excludes Git history, credentials, caches and runtime databases.
Use the generated source directory for a new repository; do not publish the entire development
workspace or the parent directory containing model assets and verification environments.
