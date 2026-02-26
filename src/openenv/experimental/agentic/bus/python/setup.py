#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Custom setup to generate protobuf files at install time."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithProto(build_py):
    """Custom build_py that generates protobuf files before building."""

    def run(self):
        self._generate_proto()
        super().run()

    def _generate_proto(self):
        from grpc_tools import protoc

        # Paths relative to setup.py location
        base_dir = Path(__file__).parent
        proto_file = base_dir.parent / "proto" / "agent_bus.proto"
        out_dir = base_dir / "src" / "agentbus" / "proto"

        # Ensure output directory exists
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate protobuf files
        # Use the proto directory as the include path so files are generated
        # directly in out_dir without an extra proto/ subdirectory
        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{proto_file.parent}",
                f"--python_out={out_dir}",
                f"--pyi_out={out_dir}",
                f"--grpc_python_out={out_dir}",
                proto_file.name,
            ]
        )

        if result != 0:
            raise RuntimeError(f"protoc failed with exit code {result}")

        # Fix imports in generated gRPC file to use absolute imports
        grpc_file = out_dir / "agent_bus_pb2_grpc.py"
        if grpc_file.exists():
            content = grpc_file.read_text()
            # Handle both old-style "from proto import" and new-style "import agent_bus_pb2"
            content = content.replace("from proto import", "from agentbus.proto import")
            content = content.replace(
                "import agent_bus_pb2", "from agentbus.proto import agent_bus_pb2"
            )
            grpc_file.write_text(content)

        # Create __init__.py if it doesn't exist
        init_file = out_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Auto-generated protobuf bindings\n")


setup(cmdclass={"build_py": BuildPyWithProto})
