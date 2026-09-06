# openenvd: operator-only process supervision

`openenvd` starts, stops, and restarts auxiliary processes inside an existing
environment container. An authenticated operator submits a command; one lifecycle
loop owns that command's launch, restart delay, and process-group cleanup.

This is an experimental building block for [issue #1053](https://github.com/huggingface/OpenEnv/issues/1053).
It does **not** implement the proposed grader/observer policies, agent tools,
Gym-like reset API, filesystem snapshots, or a complete container init system.
Existing environments do not enable it automatically.

## Trust boundary

The control token grants **arbitrary command execution**, including the ability
to run as the daemon user. Give it only to trusted operators. A trajectory writer
using this token is also an operator, not a restricted observer.

| Concern | Enforcement |
| --- | --- |
| Control API | Required bearer token on every route except the minimal `/health` probe; loopback binding by default |
| Daemon secrets | Children receive only a standard `PATH` and their explicitly configured `env` |
| Task identity | Dedicated nonzero UID/GID by default; no fallback when unavailable or exhausted |
| Privilege escalation | Linux children use `no_new_privs`; supplementary groups are cleared when changing identity |
| Network isolation | Opt-in `network_isolated=True` requires a private Linux network namespace and nonzero UID/GID; setup failure prevents execution |
| Process cleanup | Stop, exit, restart, and graceful shutdown clean up the process group, escalating from SIGTERM to SIGKILL |

UID separation does not hide world-readable files. Protect daemon assets and
output directories with filesystem permissions. Children create private files
with a `077` umask and receive closed stdin, but explicitly configured resources
remain the operator's responsibility.

The default UID range is 65536–69631. The deployment must reserve this range for
one daemon, independently of other accounts or processes. IDs are not recycled
during the daemon's lifetime: files and escaped descendants can outlive a task.
Explicit UIDs are also reserved to prevent collisions. Recreate the container and
clean task-owned persistent storage before reusing identities.

Process groups are not a sandbox. A hostile child can leave its group with
`setsid()`, exhaust resources, or access shared files permitted by its UID.
The outer container must provide containment, resource limits, and orphan
reaping (for example, an init process). Daemon SIGKILL/crash recovery and complete
hostile-process-tree cleanup are outside this implementation's guarantees.

## Run

Set a high-entropy `OPENENVD_TOKEN` through the deployment's secret configuration,
then start:

```bash
python -m openenv.core.openenvd
```

The default address is `127.0.0.1:8100`. If a deployment exposes another address
with `--host`, it must supply TLS and network access controls. Tokens are never
accepted as CLI arguments.

The default task identity allocation requires Linux root in the initial user
namespace. Network isolation additionally requires permission to create a network
namespace. Containers with remapped UIDs can supply explicit mapped nonzero
`uid` and `gid` values instead of automatic allocation. Missing capabilities
cause registration or launch to fail.

For example, from a trusted operator process with the token:

```python
import os

import httpx

with httpx.Client(
    base_url="http://127.0.0.1:8100",
    headers={"Authorization": f"Bearer {os.environ['OPENENVD_TOKEN']}"},
) as client:
    response = client.post(
        "/tasks",
        json={
            "name": "example",
            "argv": ["/bin/sleep", "30"],
            "network_isolated": True,
        },
    )
    response.raise_for_status()
    client.post("/tasks/example/stop").raise_for_status()
```

For **trusted local development commands only**, set `auto_uid=False` and omit
`uid`/`gid` to run under the daemon's identity. This explicitly gives up identity
isolation. It is not suitable for an untrusted agent or workload.

Commands use an argument list, without an implicit shell. The working directory
defaults to `/`; a configured `cwd` must be absolute. Pass interpreter settings,
application configuration, and credentials explicitly through `env`. The task
specification is copied at registration; later caller mutations cannot change it.

## Lifecycle and events

`POST /tasks` registers and starts a task; `autostart=False` only registers it.
`GET /tasks` and `GET /tasks/{name}` return status without command arguments or
environment variables. `POST /tasks/{name}/start` is idempotent while running or
waiting to restart. An explicit start after termination renews the restart budget.
`POST /tasks/{name}/stop` interrupts restart delays immediately.
`DELETE /tasks/{name}` stops and removes a registration.

`never` leaves a command exited; `on_failure` retries nonzero exits up to
`max_retries`; `always` restarts after any exit. Backoff doubles to a maximum of
30 seconds. Launch/setup failures produce `FAILED` status; failed launches stay
registered so operators can inspect or remove them. HTTP errors and lifecycle
events omit command arguments, environment values, and exception messages.

`GET /events?after=N` returns recent lifecycle events from a bounded memory buffer.
Sequence numbers reset with the daemon. The bundled `trajectory_writer_spec`
appends these events to JSONL; it records process lifecycle, not agent actions.
Slow readers can miss events, writer restarts can replay them, and standalone
writers must restart when the daemon restarts. This is a best-effort diagnostic
log, not a durable audit trail.

## Review and verification

Read `models.py` for the task contract, `supervisor.py` for lifecycle ownership,
`isolation.py` for the OS boundary, and `daemon.py` for authentication and routing.
The Linux setup helper is deliberately a standalone standard-library program:
it starts with an empty environment and disabled Python startup hooks, receives
configuration through a private descriptor, applies isolation, then executes the
command. This avoids Python callbacks between fork and exec in a threaded server.
See the [Python subprocess guidance](https://docs.python.org/3/library/subprocess.html#subprocess.Popen)
and [Linux no_new_privs documentation](https://www.man7.org/linux/man-pages/man2/PR_SET_NO_NEW_PRIVS.2const.html).

```bash
PYTHONPATH=src:envs uv run pytest tests/core/test_openenvd*.py -q
```

Portable tests cover authentication, secret handling, setup failures, lifecycle
races, and process-group cleanup. Linux tests additionally exercise real UID and
network separation, privilege restrictions, and helper descriptor handling.
