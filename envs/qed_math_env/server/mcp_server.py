# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP tool definitions for the QED Math Environment.

Tools are registered on the environment's FastMCP instance inside
QEDMathEnvironment.__init__:
- get_problem(): Return current problem, reference solution, and guidelines.
- submit_proof(proof): Grade proof via MathProofRubric and return score/reward.
- get_grading_guidelines(): Return the rubric for the current problem.

Reference:
    training/pipelinerl/domains/math/verifier_api.py - Proof verification logic
    envs/echo_env/server/echo_environment.py - MCP tool registration pattern
"""
