<!-- markdownlint-disable-file -->

# Task Research Notes: codex-web-setup-crash-104

## Research Executed

### File Analysis

- `/workspaces/lexile-corpus-tuner/.github/codex/codex-web-setup.sh`
  - `install_system_packages()` runs `apt-get update -qq` and `apt-get install ...` without retries/timeouts.
  - `ensure_pwsh()` also runs multiple `apt-get update -qq` / `apt-get install ...` operations without retries/timeouts.
  - Script uses the non-root sudo re-exec pattern:
    - `sudo bash -c "$(declare -f fn); fn"`
    - Implication for the planned fix: any shared apt helper functions introduced must either be included
      in the `declare -f` payload or otherwise made available inside the sudo shell.
  - Script already contains a retry wrapper for Poetry (`install_with_retries`) intended to tolerate transient PyPI/DNS issues.

- `/workspaces/lexile-corpus-tuner/docs/features/active/2026-01-21-codex-web-setup-crash-104/issue.md`
  - Captures repro + exact apt error snippet (HTTP 502 from Ubuntu mirror) and reports exit code 100.

- `/workspaces/lexile-corpus-tuner/docs/features/active/2026-01-21-codex-web-setup-crash-104/spec.md`
  - Specifies the target script and proposes adding apt retries/timeout handling.

- `/workspaces/lexile-corpus-tuner/.devcontainer/local/Dockerfile`
  - Establishes the parity target: a long `apt-get install` list including `shellcheck`, `shfmt`, `nodejs`, `npm`, etc.
  - Uses `apt-get update`/`apt-get install` in a Docker build context (network/mirror flakiness can still apply, but the environment differs from Codex web jobs).

- `/workspaces/lexile-corpus-tuner/.github/codex/codex-web-maintenance.sh`
  - Only performs lightweight verification (prints versions and lists directory).

### Code Search Results

- `apt-get update|apt-get install` in `.github/codex/codex-web-setup.sh`
- `sudo bash -c "$(declare -f ...)"` in `.github/codex/codex-web-setup.sh`
  - The sudo re-exec approach is used for apt operations; planned helper functions must be included in the
    sudo payload to avoid missing-function errors in non-root runs.

- `codex-web-setup.sh` repo-wide
  - Matches found in:
    - `docs/features/active/2026-01-21-codex-web-setup-crash-104/{spec,issue}.md`
    - `artifacts/chats/251219-04 Comparing devcontainer setup with codex-web-setup.md`

### External Research

- #fetch:https://manpages.debian.org/bookworm/apt/apt-get.8.en.html
  - `apt-get` returns decimal **100** on error.
  - `-m, --ignore-missing, --fix-missing` exists; description: ignore missing packages; if packages cannot be retrieved or fail integrity check, hold back those packages and handle the result; if a package is selected and cannot be downloaded, it will be silently held back.

- #fetch:https://manpages.debian.org/bookworm/apt/apt.conf.5.en.html
  - `Acquire::Retries`: “If this is non-zero APT will retry failed files the given number of times.”

- #fetch:https://manpages.debian.org/bookworm/apt/apt-transport-http.1.en.html
  - Proxy is supported via the `http_proxy` environment variable (system-wide).
  - APT-specific proxies can be configured via `Acquire::http::Proxy`, including host-specific `Acquire::http::Proxy::host`.
  - Proxy configuration supports special value `DIRECT` (no proxy) and honors `no_proxy`.
  - `Acquire::http::Timeout`: sets timeout timer used by the HTTP method; applies to connection and data timeout.
  - `Acquire::http::Pipeline-Depth` can be set to `0` to disable pipelining for misbehaving proxies.

- #fetch:https://manpages.debian.org/bookworm/apt/apt-transport-https.1.en.html
  - HTTPS transport inherits the HTTP transport options via `Acquire::https` and defaults to values specified for `Acquire::http`.
  - Examples include host-specific proxy configuration and `DIRECT` (no proxy), as well as `Timeout` under the `Acquire::https` scope.

### Project Conventions

- Standards referenced: repo toolchain + parity intent documented in `README.md` and `docs/developer-tooling.md`.
- Instructions followed: Task Researcher (research-only) mode; evidence-backed findings; write only to `artifacts/research/`.

## Key Discoveries

### Project Structure

- The bug is explicitly about `.github/codex/codex-web-setup.sh` failing during apt installs in Codex web jobs.
- The active issue/spec already points toward “apt retries/timeout handling” as the targeted fix, and the
  current script still uses direct `apt-get update` / `apt-get install` calls (no apt retry logic yet).
- The setup script already includes a retry mechanism for Poetry operations (`install_with_retries`), suggesting retries are an accepted resilience pattern for this environment.

### Root Cause Signals (from the captured error)

- The error message includes `502 Bad Gateway [IP: 172.31.1.19 8080]` when fetching from `http://archive.ubuntu.com/...`.
  - This strongly indicates the Codex execution environment is traversing an HTTP proxy/gateway at `172.31.1.19:8080`.
  - A 502 is an upstream/gateway failure class error (proxy couldn't successfully fetch/serve the requested upstream response).
  - Evidence limitation: the repo does not contain the full Codex job log or network configuration; only the apt snippet is present in `issue.md`/`spec.md`.

- Additional evidence from the expanded job log (provided by user on 2026-01-21):
  - The install proceeds successfully through hundreds of `Get:` lines and reports:
    - `Fetched 30.7 MB in 56s (552 kB/s)`
  - It then fails on a *single* package fetch:
    - `Failed to fetch ... node-cjs-module-lexer_1.2.3+dfsg-1_all.deb  502  Bad Gateway [IP: 172.31.1.19 8080]`
  - Implication:
    - This looks like *intermittent gateway/proxy failure* rather than a blanket outbound block: the proxy successfully served a large portion of the install, then returned 502 for one artifact.
    - The failing artifact is in the Ubuntu `universe` pool for `node-cjs-module-lexer`, consistent with the environment pulling Ubuntu Noble Node tooling via apt.

### Implementation Patterns

- The setup script is written as a sequence of functions with imperative execution, using `set -euo pipefail`.
- It conditionally executes apt operations only when root/sudo is present; if not root it re-execs a function body via `sudo bash -c "$(declare -f fn); fn"`.
  - Implication: if apt retry helpers are introduced, they must either:
    - be defined within the function passed to sudo, or
    - be included in the `declare -f` payload (i.e., pass helper definitions too).
  - This is a key correctness constraint for the planned apt resilience fix: helpers must be visible inside
    the sudo shell to avoid command-resolution failures in non-root runs.

### Complete Examples

```bash
# From apt-get(8): fix-missing / ignore-missing option
# (Verified via #fetch:https://manpages.debian.org/bookworm/apt/apt-get.8.en.html)
apt-get install -y --no-install-recommends --fix-missing <packages>

# From apt.conf(5): download retries
# (Verified via #fetch:https://manpages.debian.org/bookworm/apt/apt.conf.5.en.html)
apt-get -o Acquire::Retries=3 update

# From apt-transport-http(1): HTTP method timeout
# (Verified via #fetch:https://manpages.debian.org/bookworm/apt/apt-transport-http.1.en.html)
apt-get -o Acquire::http::Timeout=30 update

# From apt-transport-http(1): APT can be configured to use/bypass proxies
# (Verified via #fetch:https://manpages.debian.org/bookworm/apt/apt-transport-http.1.en.html)
# Examples of the relevant knobs:
# - environment variable: http_proxy / no_proxy
# - apt config: Acquire::http::Proxy and host-specific Acquire::http::Proxy::host
# - special value: DIRECT (no proxy)

# From apt-transport-http(1): proxies can misbehave; pipelining can be disabled
# (Verified via #fetch:https://manpages.debian.org/bookworm/apt/apt-transport-http.1.en.html)
apt-get -o Acquire::http::Pipeline-Depth=0 update

# From apt-transport-https(1): HTTPS inherits HTTP options under Acquire::https
# (Verified via #fetch:https://manpages.debian.org/bookworm/apt/apt-transport-https.1.en.html)
apt-get -o Acquire::https::Timeout=30 update
```

### API and Schema Documentation

- No new repo-level API contracts involved.
- Relevant external “API” is apt’s config interface via command-line `-o key=value`:
  - `Acquire::Retries=<int>`
  - `Acquire::http::Timeout=<seconds>`
  - `APT::Get::Fix-Missing=<bool>` (mirrors `--fix-missing`)

### Configuration Examples

```bash
# Proposed environment-driven defaults (script-level, not system-wide):
APT_RETRY_ATTEMPTS=5
APT_RETRY_DELAY_SECONDS=5
APT_HTTP_TIMEOUT_SECONDS=30

# Example use via apt-get -o:
apt-get -o Acquire::Retries="$APT_RETRY_ATTEMPTS" \
        -o Acquire::http::Timeout="$APT_HTTP_TIMEOUT_SECONDS" \
        update -qq

# Example of a wrapper-level approach (proposed):
apt_with_retries apt-get \
  -o "Acquire::Retries=${APT_RETRY_ATTEMPTS}" \
  -o "Acquire::http::Timeout=${APT_HTTP_TIMEOUT_SECONDS}" \
  -o "Acquire::https::Timeout=${APT_HTTP_TIMEOUT_SECONDS}" \
  install -y --no-install-recommends --fix-missing <packages>
```

### Technical Requirements

- Primary failure mode (evidence): apt fails to fetch a `.deb` from `archive.ubuntu.com` with an HTTP 502 attributed to `172.31.1.19:8080` (as captured in `issue.md` and `spec.md`).
- Objective behavior: the setup script completes provisioning reliably in the Codex environment.
- Constraint: changes should preserve the “devcontainer parity” intent (system packages should remain aligned with `.devcontainer/local/Dockerfile` unless explicitly justified).
- Must avoid “false success” states (e.g., `--fix-missing` silently holding back critical packages, leaving tooling missing).

**Mandatory unachievable objective callout**:
- None identified in this research pass.

## Recommended Approach

The evidence (HTTP 502 attributed to an intermediate `172.31.1.19:8080` gateway) indicates the *root cause is environmental network/proxy instability or policy*, not a deterministic logic bug in the repo. The repo can still mitigate this class of failure.

User decision (confirmed 2026-01-21): focus on **Option A + Option C**.

### Testability / unit-test design (Bats)

The script currently executes imperatively at top-level. For deterministic unit testing in Bats (without
network/root), the key requirement is to make it **safe to source**.

Recommended minimal seams:

- Add a `main()` function and guard it:
  - `if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi`
  - This prevents `apt-get`, `git clone`, etc. from running when tests `source` the script.

- Add env overrides for OS detection used by `ensure_pwsh()`:
  - If `CODEX_OS_ID` / `CODEX_OS_VERSION` are set, prefer them over `/etc/os-release`.
  - This avoids host-dependent branching in unit tests.

- (Planned) Introduce `apt_with_retries`, `apt_update`, and `apt_install` as top-level functions.
  - Rationale: these are good, unit-testable targets that can be exercised with in-process command stubs.

Bats test placement (compatible with repo tooling):

- Place tests under `tests/shell/` (or `tests/bash/`).
  - Verified: `scripts/dev_tools/shell_qc.py` discovers and runs bats tests in these directories.

Bats stubbing approach (no temp files required):

- Define bash functions in the Bats process to replace external commands:
  - `apt-get() { ... }`, `curl() { ... }`, `wget() { ... }`, `dpkg() { ... }`, `pwsh() { ... }`, `sleep() { :; }`
  - Then `source` the script and call the target function with `run`.

Suggested unit tests (high signal):

- (After apt resilience helpers are implemented) `apt_with_retries`:
  - retries a failing command and succeeds once it returns 0.
  - fails after N attempts with the expected error message.

- (After apt resilience helpers are implemented) `apt_update` / `apt_install`:
  - call `apt-get` with the expected `-o Acquire::Retries`, timeout options, and (for install) `--fix-missing`.
  - note: stub `apt-get` to record argv and return controlled exit codes.

- `check_pypi_connectivity`:
  - honors `ALLOW_OFFLINE_INSTALL=1` and does not call `curl`.
  - fails with the documented message when `curl` fails.

### Root Cause Analysis (evidence-based hypotheses)

H1) **Transient gateway/proxy outage or overload**

- Why it fits: a 502 “Bad Gateway” originates from a gateway/proxy that failed to get a valid upstream response.
- Expected behavior if true: retries (bounded) usually succeed on subsequent attempts; failures are intermittent.
- Expected behavior if true: retries (bounded) usually succeed on subsequent attempts; failures are intermittent and may occur late in a large install (as seen in the provided log).
- Best mitigations: retries + timeouts; optional backoff.

H2) **Proxy-specific incompatibility (HTTP pipelining / caching / policy)**

- Why it fits: APT can pipeline HTTP requests; some proxies misbehave. The HTTP transport explicitly documents `Acquire::http::Pipeline-Depth` as a knob to work around non-conforming proxies.
- Expected behavior if true: failures correlate with certain packages/paths; retries might still fail if proxy behavior is deterministic.
- Best mitigations: set `Acquire::http::Pipeline-Depth=0`; potentially adjust proxy/no_proxy behavior.

H3) **Runner base-image mismatch (Ubuntu “noble” apt sources vs devcontainer’s Debian Bookworm baseline)**

- Why it fits: the observed URL is `archive.ubuntu.com/ubuntu/...`, but devcontainer parity target is Debian Bookworm. Different repos/mirrors and package graphs can change what gets fetched (e.g., `nodejs` pulling `node-cjs-module-lexer`).
- Expected behavior if true: failures repeat at the same dependency edges when Ubuntu mirrors are flaky; Debian-based images might avoid the exact failing artifact path.
- Best mitigations: prefer a Debian-based runner (if Codex supports it) or make the script distro-aware and use alternative installation sources for node.

### Mental model diagram (why Option C can help)

The key observation from the log is that apt is not talking directly to `archive.ubuntu.com`; it is going through a gateway/proxy (`172.31.1.19:8080`).

When apt uses HTTP pipelining, it can send multiple "give me this .deb" requests quickly, before the prior ones fully finish. Some proxies handle that poorly.

```text
                  (Codex runner network path)

  apt-get
    |
    |  HTTP requests for .deb files
    v
  +-------------------+
  | Proxy / Gateway   |   502 happens here when it can't fetch/serve upstream
  | 172.31.1.19:8080  |<-----------------------------------------------+
  +-------------------+                                                |
    |
    |  forwards requests to upstream
    v
  +-------------------+
  | Ubuntu archive     |
  | archive.ubuntu.com |
  +-------------------+


Pipelining ON (faster, but stresses flaky proxies):
  apt:  request #1, #2, #3, #4 quickly  ---> proxy tries to juggle them

Pipelining OFF (more reliable through proxies):
  apt:  request #1 (wait) -> response #1
        request #2 (wait) -> response #2
        ...

Option C = tell apt: "do the one-at-a-time style" (Pipeline-Depth=0).
```

### Selected mitigation: Option A + Option C

Option A — **Add bounded retries + timeouts for all apt operations (resilience)**

- Mechanism:
  - Script-level retry wrapper around apt update/install (bounded, with backoff).
  - APT-native retries via `-o Acquire::Retries=<n>`.
  - Add `-o Acquire::http::Timeout=<seconds>`.
- What it addresses: H1 (transient failures), partial coverage for H2.
- Risks/downsides: does not fix deterministic proxy policy issues; can increase job duration.


Option B — **Proxy-aware configuration: bypass or scope the proxy per-host**

- Mechanism (documented in apt-transport-http):
  - Detect proxy via `http_proxy`/`no_proxy` env or existing apt config.
  - Optionally set host-specific `Acquire::http::Proxy::archive.ubuntu.com "DIRECT";` or `no_proxy=archive.ubuntu.com` to bypass a problematic proxy.
- What it addresses: H2 if the proxy is the problem and direct egress is allowed.
- Risks/downsides: if the environment requires a proxy for outbound traffic, `DIRECT` bypass will fail; must be conditional.

Option C — **Disable HTTP pipelining for apt transports**

- Mechanism (documented in apt-transport-http):
  - Set `Acquire::http::Pipeline-Depth=0` (and/or `Acquire::https::Pipeline-Depth=0`).
- What it addresses: H2 (proxy incompatibility) without needing to bypass the proxy.
- Risks/downsides: may reduce performance; may not affect the specific 502 case if it’s pure upstream outage.

Option D — **Prefer HTTPS sources / align sources with the expected base**

- Mechanism:
  - Where the environment uses HTTP apt endpoints, prefer HTTPS apt endpoints when available (APT supports HTTPS transport and inherits proxy/timeouts under `Acquire::https`).
- What it addresses: reduces some network middlebox issues and improves integrity in transit; may change how proxies handle traffic.
- Risks/downsides: still goes through proxies in many environments; doesn’t inherently fix a proxy returning 502.

Option E — **Avoid distro-provided nodejs/npm; install via a separate channel**

- Mechanism:
  - Instead of `apt-get install nodejs npm`, install node via a single-purpose installer (still network-dependent) or skip Graphite CLI unless needed.
- What it addresses: H3 by removing the `node-cjs-module-lexer` dependency edge.
- Risks/downsides: diverges from devcontainer parity; introduces additional external endpoints.

### Rejected alternatives (brief, non-exhaustive)

- Option B (bypass proxy via `DIRECT` / `no_proxy`): rejected for now because we do not have evidence that direct egress is permitted in Codex jobs; bypassing a required proxy could make the failure deterministic.
- Option D (prefer HTTPS sources): not selected initially because it may not materially change proxy behavior (many proxies still intercept/forward HTTPS), and it can expand the change surface beyond the minimal mitigation.
- Option E (avoid Ubuntu node packages): not selected initially because it diverges from devcontainer parity and introduces additional external endpoints.
- Adding an apt proxy/cache service: rejected because Codex jobs likely cannot rely on a stable shared cache service.
- Hard pinning to a specific external mirror: rejected because it can reduce portability and may violate network policy in managed runners.

## Implementation Guidance

- **Objectives**:
  - Improve reliability of `.github/codex/codex-web-setup.sh` in Codex web jobs by addressing environmental network/proxy failure modes (502 from an intermediate gateway).
  - Preserve devcontainer parity intent unless an option is explicitly chosen that trades parity for reliability.

- **Key Tasks**:
  - Add minimal instrumentation to surface proxy context in job logs (e.g., print whether `http_proxy`/`https_proxy`/`no_proxy` are set, without leaking credentials).
  - Implement Option A: env-configurable apt retries/delay/timeouts used consistently for all apt operations.
  - Implement Option C: configure `Acquire::http::Pipeline-Depth=0` (and consider `Acquire::https::Pipeline-Depth=0`) for the apt calls.
  - Decide whether to include `--fix-missing` and, if so, add post-install verification so the script fails if required tools are missing.
  - **Fix the sudo re-exec correctness issue introduced by the new `apt_*` helpers**:
    - Ensure the sudo shell has definitions for `apt_with_retries`, `apt_update`, and `apt_install`, and
      has access to the `APT_*` variables.
    - Alternatives include:
      - passing `declare -f apt_with_retries apt_update apt_install install_system_packages` to sudo, or
      - re-execing the whole script under sudo once (simpler control-flow), or
      - moving the apt helper logic inside `install_system_packages` (least reusable).
  - Add Bats unit tests for the new helper functions once the script is sourceable.
  - Defer Option B (DIRECT/no_proxy) unless later evidence shows direct egress is permitted.

- **Dependencies**:
  - None new at the repo dependency level.
  - Relies on apt behavior documented in Debian apt manpages (see External Research).

- **Success Criteria**:
  - The setup script logs enough context to distinguish “proxy outage” vs “proxy misbehavior/policy” vs “mirror outage” on failures (at least: whether proxy env vars are set; which apt URL failed).
  - Transient gateway/mirror failures no longer fail the job immediately.
  - Persistent failures still fail with a clear error after bounded retries.
  - A successful run proves required tools are actually present.
  - Unit tests (Bats) validate retry behavior and correct apt option construction without requiring network/root.
