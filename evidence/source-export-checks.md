# Source export verification

Checked October 18, 2026, on Node 22.23.2 and macOS using a temporary local copy of the original installed dependencies.

- `npm test`: 148 passed, 2 skipped, 2 todo across 5 test files. The two skips are generated-dataset checks because the optional training-case directory is omitted. The todo cases are retained from the original suite.
- `npm run build:public`: passed, including `tsc --noEmit` and the static Vite production build.
- A scan for private keys, known credential-token formats, quoted credential assignments and literal bearer tokens found no high-confidence secrets in the exported files.
- Application source, scripts, tests, runtime data and prepared family terrain grids are copied from the original project without code changes.

The temporary dependency copy and generated build directory were removed after verification. No server was started. Browser tests were not rerun for this export; their earlier recorded results remain described in `EVIDENCE.md`.
