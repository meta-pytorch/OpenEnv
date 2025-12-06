# Copyright (c) Yogesh Singla and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Julia Environment HTTP Client.

This module provides the client for connecting to a Julia Environment server
over HTTP.
"""

from typing import Dict

from core.client_types import StepResult
from core.http_env_client import HTTPEnvClient

from .models import JuliaAction, JuliaObservation, JuliaState


class JuliaEnv(HTTPEnvClient[JuliaAction, JuliaObservation]):
    """
    HTTP client for the Julia Environment.
    
    This client connects to a JuliaEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.
    
    Example:
        >>> # Connect to a running server
        >>> client = JuliaEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.stdout)
        >>>
        >>> # Execute Julia code
        >>> action = JuliaAction(code='''
        ... function multiply(a, b)
        ...     return a * b
        ... end
        ... 
        ... using Test
        ... @test multiply(3, 4) == 12
        ... ''')
        >>> result = client.step(action)
        >>> print(result.observation.tests_passed)  # 1
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = JuliaEnv.from_docker_image("julia-env:latest")
        >>> result = client.reset()
        >>> result = client.step(JuliaAction(code="println(2 + 2)"))
        >>> print(result.observation.stdout)  # "4\n"
        >>> client.close()
    """

    def _step_payload(self, action: JuliaAction) -> Dict:
        """
        Convert JuliaAction to JSON payload for step request.
        
        Args:
            action: JuliaAction instance
            
        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "core_code": action.core_code,
            "test_code": action.test_code
        }

    def _parse_result(self, payload: Dict) -> StepResult[JuliaObservation]:
        """
        Parse server response into StepResult[JuliaObservation].
        
        Args:
            payload: JSON response from server
            
        Returns:
            StepResult with JuliaObservation
        """
        obs_data = payload.get("observation", {})
        observation = JuliaObservation(
            stdout=obs_data.get("stdout", ""),
            stderr=obs_data.get("stderr", ""),
            exit_code=obs_data.get("exit_code", 0),
            tests_passed=obs_data.get("tests_passed", 0),
            tests_failed=obs_data.get("tests_failed", 0),
            code_compiles=obs_data.get("code_compiles", True),
            metadata=obs_data.get("metadata", {}),
        )
        
        return StepResult[JuliaObservation](
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> JuliaState:
        """
        Parse server response into JuliaState object.
        
        Args:
            payload: JSON response from /state endpoint
            
        Returns:
            JuliaState object with episode metadata
        """
        return JuliaState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            last_exit_code=payload.get("last_exit_code", 0),
            last_code_compiles=payload.get("last_code_compiles", True),
            total_tests_passed=payload.get("total_tests_passed", 0),
            total_tests_failed=payload.get("total_tests_failed", 0),
        )

