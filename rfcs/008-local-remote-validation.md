# RFC 008: Local and Remote Validation Architecture

**Status**: Draft
**Created**: 2026-07-11
**Authors**: @openenv
**RFC ID**: 008

## Summary

OpenEnv validation consumes a spec-neutral `ValidationSubject` produced by a
trusted, versioned spec adapter. The subject identifies the source spec and
execution model and carries normalized requirements for identity, resources,
timeouts, networking, health checks, verifier settings, and artifacts. A
versioned OpenEnv policy combines that subject with runner capabilities and
runtime discovery to produce a `ValidationPlan`.

`openenv validate` initially selects the served OpenEnv adapter. That adapter
can use Harbor v0.5/schema 1.1 `task.toml` as an auxiliary requirements source,
but Harbor is not a core planner or executor dependency. Future workflows can
register one-shot or externally hosted task specs without routing them through
OpenEnv's served-runtime checks.

The same check implementations and report schema run through two execution
paths:

- `openenv validate` runs checks on the developer host or, when requested, in
  a dedicated Hugging Face Sandbox and returns an unofficial author report.
- Hub-triggered remote certification runs isolated, privileged, multi-host,
  artifact, and model-based checks.

This design does not add a `metadata.openenv` extension to Harbor. OpenEnv
policy belongs in the versioned validator, not in submitted task metadata, and
the normalized subject is an internal contract rather than a new package spec.

## Motivation

### Problem statement

The existing validator answers two narrower questions: whether a source tree
looks deployable and whether a running HTTP API exposes a small set of
endpoints. It cannot say which checks were unavailable, bind results to a
source revision or image, enforce resource and network declarations, or
distinguish local validation from official certification.

Running every check in GitHub Actions would also put uploaded code inside a
trusted build system. Conversely, using only a hosted Space cannot prove
cross-host reproducibility, deny-by-default egress, or reference-model
behavior. Validation therefore needs one policy and report format across
multiple trust domains.

### Goals

1. Give every criterion a stable identity, normative requirement, capability
   set, severity, timeout, and structured evidence.
2. Produce comparable local and remote reports.
3. Make unavailable checks explicit rather than silently treating them as
   passes.
4. Keep coordinator credentials out of submitted and verifier workloads.
5. Bind official results to an exact repository SHA, policy version, and image
   digest.
6. Preserve a quick local workflow without overstating what it proves.
7. Keep planners, executors, and reports independent of any one task-package
   format or execution model.
8. Give environment authors a strict pre-publish loop with structured,
   machine-readable guidance for fixing invalid environments.

### Non-goals

- `openenv validate` does not issue an official certification.
- Schema 1.1 does not gain OpenEnv-specific metadata or endpoint allowlists.
- This RFC does not implement PostTrain/BenchFlow parsing, `validate-task`,
  one-shot oracle replay, or public plugin discovery.
- This RFC does not certify an external spec until a policy defines its
  lifecycle, reward boundary, and isolation requirements.
- This RFC does not make build determinism blocking before an isolated build
  backend is deployed and calibrated.
- GitHub Actions does not execute uploaded environments.

## Design

### Architecture overview

```text
Space revision
  -> Hub repo-content webhook
  -> trusted validation-coordinator HF Job
  -> ValidationPlan
      |-- dedicated HF Sandbox: runtime/resource/replay checks
      |-- dedicated ACA Sandbox: egress and containment checks
      |-- artifact inspector Job: image/SBOM/signature checks
      `-- dedicated CPU/GPU Jobs: learnability and cross-host checks
  -> signed report registry
  -> collection eligibility/status
```

Every Space content revision triggers a trusted, pinned coordinator Job. The
webhook event identifies the repository and head SHA. The coordinator waits for
the Space build, resolves its immutable image/revision, creates a plan, and
dispatches independent lanes. Hugging Face documents both
[webhook-triggered Jobs](https://huggingface.co/docs/hub/en/jobs-webhooks) and
[Jobs hardware, timeout, log, label, volume, and cancellation controls](https://huggingface.co/docs/huggingface_hub/en/guides/jobs).

### Core abstractions

#### `ValidationSpecAdapter`

A spec adapter performs signature-only detection and safe structural loading.
It returns a `ValidationSubject` containing:

- stable spec and adapter IDs and versions;
- an execution model such as `served`, `one_shot`, or `external`;
- the relative signature path and whether selection was automatic or explicit;
- normalized, report-safe `ValidationRequirements`; and
- declarations needed by the applicable policy, such as an isolated verifier.

Adapters never import or execute submitted code. Automatic detection rejects
zero or multiple matches instead of selecting by registry order. A malformed
signature still selects its adapter and produces an invalid result; it must not
be mistaken for an absent spec. Official coordinators use only pinned adapters
installed in the trusted validator image. Adapter code discovered inside a
submission cannot participate in certification.

The constructor-injected registry in this RFC is an internal extension seam.
A future `openenv validate-task --spec auto|...` workflow may expose a public
registry, but that command and its plugin-discovery mechanism are outside this
RFC.

#### `ValidationRequirements`

Adapters translate spec-specific documents into normalized resource, timeout,
network, image, verifier, health, artifact, and step requirements. Unknown
values remain unknown rather than acquiring format-specific defaults. Network
requirements distinguish unspecified, deny-all, allow-all, and endpoint
allowlist modes.

The initial served OpenEnv adapter treats Harbor `task.toml` as an optional
auxiliary requirements source. This composition is explicit: at an OpenEnv
environment under a Harbor task root, OpenEnv remains the primary spec and
Harbor supplies requirements. A future task-package workflow may instead treat
Harbor as the primary spec at the task root.

#### `ValidationCheck`

A check has:

- a stable criterion ID;
- the RFC requirement it enforces;
- required capabilities, such as `source`, `runtime`, `container_image`,
  `network_enforcement`, `gpu`, or `cross_host`;
- a blocking or advisory policy default;
- a timeout; and
- an implementation returning structured evidence.

Checks do not decide where they run. A local executor and each remote lane run
the same contract. A runner without the required capabilities emits `skip`; it
must not synthesize a pass.

#### `ValidationPlan`

The planner combines:

- the `ValidationSubject` returned by a trusted spec adapter;
- normalized `ValidationRequirements`;
- the selected version of OpenEnv validation policy;
- runner capabilities; and
- runtime-discovered schemas, tools, tasks, rubrics, seeds, and trajectory
  declarations.

Harbor v0.5/schema 1.1 is the initial auxiliary requirements baseline. Its
networking field can express public internet or deny-all, but not endpoint
allowlists. The Harbor adapter maps those declarations into the normalized
network modes. Under this baseline, checks for an internet-dependent
environment's precise egress policy remain advisory.

During migration, local validation records missing normalized requirements as
a blocking `skip`: it does not make otherwise valid legacy environments fail
locally, but it prevents certification eligibility. A present malformed or
unsupported auxiliary document is a blocking `fail`. Spec detection and
requirements completeness are separate criteria, so a valid `openenv.yaml`
does not imply that resource or network requirements were declared.

#### Shared report

Local and remote executors emit report schema version 1.0. Each criterion has
`pass`, `fail`, `skip`, or `error`, plus severity, evidence, duration, timeout,
and required capabilities. The report contains:

- policy and report-schema versions;
- spec ID/version, adapter ID/version, execution model, and detection
  provenance;
- target and runner provenance;
- repository SHA and image digest when verified;
- start, finish, and duration;
- criterion results and status counts; and
- separate `passed`, `certification_eligible`, and `certified` fields.

Static, runtime, and full profiles distinguish unavailable checks from failed
checks: only blocking `fail` and `error` results make those executions fail,
while a blocking `skip` prevents certification eligibility. The strict
`publish` profile treats a blocking `skip` as incomplete and therefore
non-passing. A report becomes certified only after an official dedicated
runner completes the blocking policy and the signed report is verified. The
executor itself can only emit a signature-ready eligible report with
`certified: false`; the report registry derives certified status after
verifying the immutable signed envelope.

Failed, errored, and incomplete criteria may also carry typed diagnostics and
remediation. Diagnostics use stable reason codes and repository-relative
locations (including a document pointer or source line when known).
Remediation is trusted policy or adapter output, represented as structured
edits, documentation links, or command argument vectors rather than shell
strings. It is display-only, never executes automatically, and cannot change a
criterion's status or severity. Submitted verifier output and runtime evidence
cannot author remediation instructions.

#### Environment-specific criteria

An optional `tests/test.sh` declares additional task-specific criteria. It is
additive: it cannot replace a built-in criterion, reuse a built-in ID, or
suppress a built-in failure. Submitted verifier scripts run only in a separate
disposable verifier sandbox with no coordinator credentials. Static local
validation may inspect the declaration but does not execute it on the host.

### Local execution

`openenv validate` validates served OpenEnv environments and supports four
explicit profiles:

- `--profile static`: manifest, layout, dependencies, lockfile, source schema,
  and Dockerfile/image-declaration checks.
- `--profile runtime`: static checks plus launching a local server or connecting
  to `--url`, followed by HTTP/WebSocket, tool, task, reward, seed, and
  trajectory checks supported by the running API. The tool checks also enforce
  the agent/control boundary: MCP cannot expose `reset`, `step`, or `state`.
- `--profile full`: every policy check; unavailable certification-only
  capabilities are reported as skipped.
- `--profile publish`: the runtime check set used as an author release gate.
  Every blocking criterion must pass; a blocking skip, failure, or error exits
  non-zero.

`--json` writes the shared schema to stdout, and `--output PATH` writes the same
JSON document to disk. All local invocations use the shared planner and report;
unqualified invocations differ only in presentation. Human output includes
actionable diagnostics and remediation, and `--verbose` additionally renders
safe criterion evidence.

`--remote` archives the selected source revision, uploads it to a newly created
dedicated Hugging Face Sandbox, runs the pinned validator there, and returns
the same report schema. This author-triggered path never uses `SandboxPool`,
never forwards the developer's Hugging Face token into the workload, and never
claims certification. The archive excludes credentials, local virtual
environments, VCS metadata, caches, and prior generated reports. The exact
validator source is uploaded alongside the environment so the report is bound
to the CLI policy version that initiated the run.

`openenv push` requires a passing remote `publish` report before uploading an
environment to the Hub. It writes a path-normalized copy to
`.openenv/validation-report.json` inside the staging tree, so the author report
is versioned with the Space revision it gated. This report is explicitly
unofficial and is useful for author and agent feedback; it does not satisfy
collection admission or replace the independently generated certification
report. Custom-registry pushes use the same strict profile locally until that
workflow has an equivalent isolated runner.

Local execution never claims certification for cross-host reproducibility,
enforced egress isolation, official reference-model results, signatures, or
remote image identity.

Execution dispatch is model-aware. Only `served` subjects use the current
Uvicorn launch path. A subject whose execution model has no local runner emits
a structured unsupported skip rather than being launched as an OpenEnv server.
The separate task-package command proposed in
[issue #898](https://github.com/huggingface/OpenEnv/issues/898) can later supply
one-shot runners without changing the shared planner, executor, or report.

Automatic local launch runs the checkout as the current user and is only for
trusted development source. Untrusted submissions must be connected through an
already isolated runtime or sent to the dedicated remote runner.

### Remote runner architecture

#### Coordinator

The coordinator is a trusted, pinned image. It selects a pinned spec adapter,
parses source documents, fetches source, waits for the Space build, creates
plans, dispatches work, and assembles signed reports. It never imports or
executes submitted Python.

Executions are keyed by `(repo_id, head_sha, spec_id, adapter_version,
policy_version)` for idempotency. Duplicate webhook deliveries reuse the same
execution, and a newer revision cancels obsolete work.

The author-triggered `--remote` runner is not this coordinator. It performs one
ephemeral publish validation and returns its report directly to the caller; it
has no service token, registry-writing authority, certification key, or
collection-admission role.

#### Hugging Face runtime lane

Uploaded environments run in dedicated Hugging Face Sandboxes. Dedicated
sandboxes map one Job to one VM and support GPUs; `SandboxPool` places multiple
same-user workloads on a shared host. Hugging Face explicitly recommends the
dedicated form for mutually untrusted code. See the
[HF Sandboxes isolation comparison](https://huggingface.co/docs/huggingface_hub/guides/sandbox).

`HFSandboxProvider` therefore has explicit `pooled` and `dedicated` modes.
Pooled remains useful for trusted rollout fan-out, but official certification
rejects it and never falls back to it after a dedicated launch failure.

#### ACA security lane

Ephemeral Azure Container Apps Sandboxes run containment and network checks
with deny-by-default egress. The runner consumes only normalized network
requirements. With the initial Harbor auxiliary adapter, schema 1.1
`allow_internet = false` maps to a blocking deny-all check; public-internet
tasks receive advisory network results because that schema cannot express an
endpoint allowlist.

#### Artifact lane

A trusted HF Job uses daemonless tools to inspect image manifests, layers,
labels, SBOMs, signatures, and registry identity. It records the immutable
digest in the report. It does not start submitted entry points.

Reproducible rebuilding is deferred to isolated rootless BuildKit workers.
Build-determinism results remain advisory until that backend is deployed and
its thresholds are calibrated.

#### Cross-host and model lanes

Independent dedicated Jobs/Sandboxes replay trajectories and run pinned
reference models. Hardware is selected from normalized resource requirements.
Cross-host, learnability, and reference-model checks remain advisory until
infrastructure and thresholds are calibrated through a policy update.

### Security and result storage

1. Only the coordinator receives the service token. Tokens are never forwarded
   into subject or verifier sandboxes.
2. Private source is fetched by the coordinator and copied to workers as a
   revision-pinned archive.
3. Spec adapters are pinned coordinator code and are never loaded from the
   submitted archive.
4. Task verifiers run separately, without credentials.
5. Environment-variable values are never serialized into plans, logs, reports,
   or artifacts; only variable names may appear as evidence.
6. Immutable official reports live in an OpenEnv validation Dataset or object
   store. Large logs and trajectories are referenced artifacts.
7. Official reports are not committed into submitted Spaces, avoiding webhook
   loops. The unofficial pre-publish report is included in the original
   submitted revision and never added through a follow-up commit.
8. The official collection reads the report registry and admits only the exact
   revision passing the current blocking policy.
9. GitHub Actions tests and publishes trusted validator/coordinator images but
   never runs uploaded environments.

## Key design decisions

### Specs adapt into normalized requirements

**Chosen approach:** Put format and execution-model knowledge behind trusted
spec adapters, then run OpenEnv policy over normalized requirements.

**Rationale:** PostTrain one-shot tasks, OpenEnv served environments, and Harbor
packages have different lifecycles. A normalized internal contract preserves
shared planning and reporting without pretending their checks are identical.
Harbor already has versioned homes for the initial resource and trust-relevant
declarations, so the first adapter reuses them rather than creating an OpenEnv
metadata island.

**Trade-off:** Each supported spec needs a pinned adapter and applicable policy
check catalog. Schema 1.1 still cannot describe endpoint allowlists, so some
security results stay advisory under the initial adapter.

### Capabilities produce skips

**Chosen approach:** Plan the complete selected policy, then emit explicit skips
where the runner lacks a capability.

**Rationale:** Users can distinguish "not checked" from "passed", and local and
remote result sets stay comparable.

**Trade-off:** Local full reports are longer and are not certification reports.

### Dedicated isolation for certification

**Chosen approach:** Reject pooled HF execution for uploaded code.

**Rationale:** Pool uid/Landlock isolation is designed for same-user trust;
dedicated Sandboxes provide the VM boundary required here.

**Trade-off:** Certification has higher cold-start time and cost than rollout
fan-out.

### Registry instead of repository commits

**Chosen approach:** Store signed immutable reports out of band.

**Rationale:** This avoids webhook loops and prevents a submitter from replacing
the evidence associated with a revision.

**Trade-off:** Collection admission depends on a separate highly available
registry.

## Examples

```bash
# Fast source checks with a human report
openenv validate envs/echo_env --profile static

# Connect to a running environment and emit the shared schema
openenv validate --url http://localhost:8000 --profile runtime --json

# Run every local check and record remote-only skips
openenv validate envs/echo_env --profile full --output validation.json

# Run the strict author gate inside a dedicated HF Sandbox
openenv validate envs/echo_env --profile publish --remote --output validation.json
```

Example result excerpt:

```json
{
  "report_schema_version": "1.0",
  "policy_version": "rfc008-v1",
  "spec": {
    "id": "openenv",
    "version": "1",
    "adapter": {"id": "openenv", "version": "1"},
    "execution_model": "served"
  },
  "profile": "full",
  "passed": true,
  "certification_eligible": false,
  "certified": false,
  "repo_sha": "<sha>",
  "image_digest": null,
  "criteria": [
    {
      "id": "remote.cross_host_reproducibility",
      "status": "skip",
      "severity": "advisory",
      "required_capabilities": ["cross_host"]
    }
  ]
}
```

## Delivery phases

1. Implement the trusted spec-adapter registry, normalized requirements,
   planner, report schema, Harbor auxiliary loader, local profiles, structured
   remediation, strict publish gate, and dedicated author-triggered HF Sandbox
   runner.
2. Deploy the webhook-triggered coordinator Job and dedicated HF runtime lane.
3. Add ACA security and artifact-inspection lanes.
4. Add cross-host, GPU/reference-model, and learnability lanes.
5. Add isolated reproducible-build workers and promote calibrated advisory
   checks through an RFC policy update.

## Test plan

- Prove normalized criterion parity between local and remote executors using
  `echo_env`.
- Register a test-only second spec without modifying planner or executor code;
  test automatic, explicit, absent, duplicate, and ambiguous selection.
- Verify one-shot and external execution models never receive the served
  OpenEnv launch path or WebSocket/MCP check catalog.
- Generate plans across Harbor resource, timeout, network, verifier, health,
  and artifact configurations.
- Test malformed, unsupported, legacy-aliased, and absent manifests without
  leaking environment values.
- Test webhook replay, duplicate delivery, stale-revision cancellation, build
  failure, worker timeout, and partial-lane failure.
- Verify coordinator credentials never appear in worker environments, logs,
  reports, or artifacts.
- Test dedicated-versus-pooled HF enforcement.
- Exercise ACA deny-all and future allowlist behavior using the existing
  integration-test pattern.
- Test signed report verification and collection gating against exact repo SHA,
  image digest, spec/adapter version, and policy version.

## Assumptions and open work

- The initial primary spec is served OpenEnv; Harbor v0.5/schema 1.1 is its
  auxiliary requirements baseline.
- Public third-party adapter discovery and task-package CLI dispatch remain
  follow-up work for
  [issue #898](https://github.com/huggingface/OpenEnv/issues/898).
- HF Jobs and dedicated Sandboxes are the primary remote platform; ACA is a
  specialized security runner.
- Every Space content revision triggers certification.
- Advanced checks remain advisory until their infrastructure and thresholds
  are calibrated.
- The remote-runner abstraction belongs in this RFC because it creates a major
  trust boundary even though its deployment is phased.
