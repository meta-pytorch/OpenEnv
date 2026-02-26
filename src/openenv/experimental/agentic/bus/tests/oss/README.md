# OSS Integration Tests

These tests spin up a DynamoDBLocal container using testcontainers.

## Setup

Install Podman on your devserver:

```bash
sudo dnf install -y podman
```

Start the Podman daemon with the socket location expected by the test:

```bash
podman system service --time=0 unix:///tmp/podman.sock &
```

## Running the tests with podman

```bash
DOCKER_HOST=unix:///tmp/podman.sock TESTCONTAINERS_RYUK_DISABLED=true cargo test -p macro_tests_oss
```

## Running the tests with podman in host networking mode

In some environments, the default bridge networking mode may not work due to missing netavark/iptables support. Set the `HOST_NETWORK` environment variable to switch to host networking:

```bash
DOCKER_HOST=unix:///tmp/podman.sock TESTCONTAINERS_RYUK_DISABLED=true HOST_NETWORK=true cargo test -p macro_tests_oss -- --test-threads=1
```
