"""
Julia Code Action Environment.

This environment mirrors the PythonCodeActEnv but runs Julia code instead.
It executes Julia code using JuliaExecutor, captures output,
tracks the last exit code, and returns a JuliaObservation.
"""

import re
import uuid

from core.env_server import Environment
from core.tools import JuliaExecutor
from ..models import JuliaAction, JuliaObservation, JuliaState
from .julia_transforms import create_safe_julia_transform


class JuliaCodeActEnv(Environment):
    """
    Julia Code Action Environment for executing code and tracking state.

    This environment executes Julia code submitted as CodeAction during step,
    maintains the last exit code in its state, and returns results wrapped
    in CodeObservation.

    Example:
        >>> env = JuliaCodeActEnv()
        >>> obs = env.reset()
        >>> action = CodeAction(code='println("Hello, Julia!")')
        >>> obs = env.step(action)
        >>> print(obs.stdout)  # "Hello, Julia!\n"
        >>> print(obs.exit_code)  # 0
        >>> print(env.state.last_exit_code)  # 0
    """

    def __init__(self):
        """Initialize the Julia Code Act Environment."""
        self._executor = JuliaExecutor()
        self._state = JuliaState()
        self.transform = create_safe_julia_transform()

    def reset(self) -> JuliaObservation:
        """
        Reset environment for a fresh Julia execution session.
        Returns an empty JuliaObservation with exit_code=0.
        """
        self._state = JuliaState(episode_id=str(uuid.uuid4()), step_count=0)
        self._state.last_exit_code = 0
        self._state.last_code_compiles = True
        self._executor = JuliaExecutor()

        observation = JuliaObservation(
            stdout="",
            stderr="",
            exit_code=0,
            reward=0.0,
            metadata={"core_code": "", "test_code": ""},
            tests_passed=0,
            tests_failed=0,
            code_compiles=True,
        )

        observation = self._apply_transform(observation)
        return observation

    def step(self, action: JuliaAction) -> JuliaObservation:
        """
        Execute Julia code and return the result as JuliaObservation.

        Optimized single-pass execution:
        - Runs core_code + test_code together
        - Infers compilation status from combined execution
        - 2x faster than double execution
        """
        if not isinstance(action, JuliaAction):
            raise ValueError(f"Expected JuliaAction, got {type(action)}")

        # Single execution: Run core_code + test_code together
        combined_code = action.core_code + "\n\n" + action.test_code
        full_result = self._executor.run(combined_code)

        # Parse test results from execution output
        tests_passed, tests_failed = self._parse_test_results(
            full_result.stdout, full_result.stderr
        )

        # Infer compilation status from execution
        # If tests ran, code compiled successfully
        # If exit_code != 0 and no tests ran, code didn't compile
        code_compiles = (
            full_result.exit_code == 0  # Clean execution
            or tests_passed > 0  # Some tests passed (code must have compiled)
            or tests_failed > 0  # Some tests failed (code compiled but tests failed)
        )

        # If no tests detected and non-zero exit, check for compilation errors
        if not code_compiles and tests_passed == 0 and tests_failed == 0:
            # Check stderr for compilation errors
            stderr_lower = full_result.stderr.lower()
            if any(
                err in stderr_lower
                for err in ["error", "syntax", "undefined", "loadError"]
            ):
                code_compiles = False
            else:
                # If no clear compilation error, assume it compiled
                code_compiles = True

        # Calculate reward based on compilation and test results
        reward = self._calculate_reward(code_compiles, tests_passed, tests_failed)

        # Update environment state
        self._state.step_count += 1
        self._state.last_exit_code = full_result.exit_code
        self._state.last_code_compiles = code_compiles
        self._state.total_tests_passed = tests_passed
        self._state.total_tests_failed = tests_failed

        # Build observation
        observation = JuliaObservation(
            stdout=full_result.stdout,
            stderr=full_result.stderr,
            exit_code=full_result.exit_code,
            reward=reward,
            metadata={"core_code": action.core_code, "test_code": action.test_code},
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            code_compiles=code_compiles,
        )

        # Apply safety and quality transforms
        observation = self._apply_transform(observation)

        return observation

    def _parse_test_results(self, stdout: str, stderr: str) -> tuple[int, int]:
        """
        Parse Julia test output to count passed/failed tests.

        Julia's Test module outputs results like:
        "Test Summary:      | Pass  Fail  Total  Time"
        "Add function Tests |    1     1      2  1.5s"

        Also checks error messages:
        "Some tests did not pass: 1 passed, 1 failed, 0 errored, 0 broken."

        Args:
            stdout: Standard output from Julia execution
            stderr: Standard error from Julia execution

        Returns:
            Tuple of (tests_passed, tests_failed)
        """
        # Combine stdout and stderr for analysis
        passed = 0
        failed = 0
        output = stdout + "\n" + stderr

        # Method 1: Look for "Some tests did not pass" error message
        # Pattern: "Some tests did not pass: X passed, Y failed, Z errored, W broken."
        error_pattern = r"Some tests did not pass:\s*(\d+)\s+passed,\s*(\d+)\s+failed,\s*(\d+)\s+errored"
        match = re.search(error_pattern, output)

        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            errored = int(match.group(3))
            return passed, failed + errored  # Treat errors as failures

        # Method 2: Look for Test Summary table
        # Multiple possible formats:
        # All pass:     "Test Summary: | Pass  Total  Time"
        #               "My Tests     |    3      3  0.5s"
        # Some fail:    "Test Summary: | Pass  Fail  Total  Time"
        #               "My Tests     |    2     1      3  0.5s"
        # All error:    "Test Summary: | Error  Total  Time"
        #               "My Tests     |     3      3  0.9s"
        # Mixed:        "Test Summary: | Pass  Fail  Error  Total  Time"
        #               "My Tests     |    1     1      1      3  0.5s"
        summary_lines = output.split("\n")
        for i, line in enumerate(summary_lines):
            if "Test Summary:" in line and i + 1 < len(summary_lines):
                header_line = line
                next_line = summary_lines[i + 1]

                # Determine which columns are present
                has_pass = "Pass" in header_line
                has_fail = "Fail" in header_line
                has_error = "Error" in header_line

                # Extract all numbers from the line
                all_numbers = re.findall(r"\d+", next_line)
                if not all_numbers:
                    continue

                # Last number is always Total, second to last is Time (skip it)
                # Extract based on which columns exist
                if has_pass and has_fail and has_error:
                    # Pass  Fail  Error  Total  Time
                    if len(all_numbers) >= 5:
                        passed = int(all_numbers[0])
                        failed = int(all_numbers[1]) + int(
                            all_numbers[2]
                        )  # Fail + Error
                        return passed, failed
                elif has_pass and has_fail:
                    # Pass  Fail  Total  Time
                    if len(all_numbers) >= 4:
                        passed = int(all_numbers[0])
                        failed = int(all_numbers[1])
                        return passed, failed
                elif has_pass and has_error:
                    # Pass  Error  Total  Time
                    if len(all_numbers) >= 4:
                        passed = int(all_numbers[0])
                        failed = int(all_numbers[1])  # Treat errors as failures
                        return passed, failed
                elif has_fail and has_error:
                    # Fail  Error  Total  Time (no passes)
                    if len(all_numbers) >= 4:
                        passed = 0
                        failed = int(all_numbers[0]) + int(all_numbers[1])
                        return passed, failed
                elif has_pass:
                    # Pass  Total  Time (no failures/errors)
                    if len(all_numbers) >= 3:
                        passed = int(all_numbers[0])
                        failed = 0
                        return passed, failed
                elif has_error:
                    # Error  Total  Time (all errors, no passes)
                    if len(all_numbers) >= 3:
                        passed = 0
                        failed = int(all_numbers[0])  # Treat all errors as failures
                        return passed, failed
                elif has_fail:
                    # Fail  Total  Time (all failures, no passes)
                    if len(all_numbers) >= 3:
                        passed = 0
                        failed = int(all_numbers[0])
                        return passed, failed

        return passed, failed

    def _calculate_reward(
        self, code_compiles: bool, tests_passed: int, tests_failed: int
    ) -> int:
        """
        Optimized integer reward for Julia GRPO.
        Strong signal shaping: rewards correctness, penalizes instability,
        and gives higher incentive for near-perfect results.
        """

        # Code doesn't compile — immediate strong penalty
        if not code_compiles:
            return -3

        reward = 1

        reward += 3 * tests_passed - 1 * tests_failed

        if tests_failed == 0 and tests_passed > 0:
            reward += 2

        return reward

    def _apply_transform(self, observation: JuliaObservation) -> JuliaObservation:
        """Apply safety and quality transforms to observation."""
        if self.transform:
            observation = self.transform(observation)
        return observation

    @property
    def state(self) -> JuliaState:
        """Return current environment state."""
        return self._state
