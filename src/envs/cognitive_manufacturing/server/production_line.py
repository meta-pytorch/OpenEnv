"""
Production Line Simulator - Phase 1

Simulates a 4-machine production line with material flow, buffers, and dependencies.

Architecture:
    [M1: Prep] → Buffer → [M2: Assembly] → Buffer → [M3: Finishing] → Buffer → [M4: QC]
"""

import uuid
from dataclasses import dataclass, field
from typing import Literal

from .simulator import SimulatedMachine
from ..models import ProductUnit, Buffer, LineMetrics


class ProductionLineSimulator:
    """
    Simulates a 4-machine production line with material flow.

    Machines:
        M1 (Prep): Prepares raw materials
        M2 (Assembly): Assembles components
        M3 (Finishing): Applies finishing/coating
        M4 (QC): Quality control inspection

    Flow:
        Raw → M1 → Buffer → M2 → Buffer → M3 → Buffer → M4 → Finished
    """

    def __init__(self):
        """Initialize production line with 4 machines and 3 buffers."""
        # Create machines
        self.machines = {
            "M1": SimulatedMachine(machine_id="M1"),
            "M2": SimulatedMachine(machine_id="M2"),
            "M3": SimulatedMachine(machine_id="M3"),
            "M4": SimulatedMachine(machine_id="M4"),
        }

        # Create buffers between machines
        self.buffers = {
            "M1_M2": Buffer(buffer_id="M1_M2", capacity=10),
            "M2_M3": Buffer(buffer_id="M2_M3", capacity=10),
            "M3_M4": Buffer(buffer_id="M3_M4", capacity=10),
        }

        # Line metrics
        self.metrics = LineMetrics()

        # Output storage (finished products)
        self.finished_products: list[ProductUnit] = []

        # Raw material supply (unlimited for now)
        self.raw_material_available = True

    def step(self, dt: float) -> dict:
        """
        Step the entire production line forward in time.

        Process:
        1. Step each machine's physics (temperature, wear, etc.)
        2. Process production at each stage
        3. Move materials between stages
        4. Update buffers
        5. Calculate metrics

        Args:
            dt: Time step in hours

        Returns:
            Dictionary with step results
        """
        results = {
            "machines": {},
            "buffers": {},
            "produced": 0,
            "defects": 0,
            "bottleneck": None,
        }

        # Step 1: Update all machine physics
        for machine_id, machine in self.machines.items():
            machine_result = machine.step(dt)
            results["machines"][machine_id] = machine_result

        # Step 2: Process production at each stage (in reverse order to allow flow)

        # M4 (QC): Inspect and output finished products
        if self.machines["M4"].status == "running":
            m4_result = self._process_qc(dt)
            results["produced"] += m4_result["produced"]
            results["defects"] += m4_result["defects"]

        # M3 (Finishing): Process from M2_M3 buffer, output to M3_M4 buffer
        if self.machines["M3"].status == "running":
            self._process_finishing(dt)

        # M2 (Assembly): Process from M1_M2 buffer, output to M2_M3 buffer
        if self.machines["M2"].status == "running":
            self._process_assembly(dt)

        # M1 (Prep): Take raw materials, output to M1_M2 buffer
        if self.machines["M1"].status == "running":
            self._process_prep(dt)

        # Step 3: Update buffer levels in results
        for buffer_id, buffer in self.buffers.items():
            results["buffers"][buffer_id] = {
                "level": buffer.current_level,
                "capacity": buffer.capacity,
                "utilization": buffer.current_level / buffer.capacity,
            }

        # Step 4: Identify bottleneck
        results["bottleneck"] = self._identify_bottleneck()

        # Step 5: Update metrics
        self._update_metrics(dt)

        return results

    def _process_prep(self, dt: float):
        """M1: Prepare raw materials."""
        machine = self.machines["M1"]
        buffer = self.buffers["M1_M2"]

        if not self.raw_material_available:
            return

        # Check if buffer has space
        if buffer.current_level >= buffer.capacity:
            return  # Buffer full, M1 blocked

        # Calculate production
        production_rate = machine.production_output  # units/hour
        units_to_produce = int(production_rate * dt)

        for _ in range(units_to_produce):
            if buffer.current_level >= buffer.capacity:
                break

            # Create new product unit
            unit = ProductUnit(
                unit_id=str(uuid.uuid4())[:8],
                created_at=self.metrics.total_produced,  # Use as timestamp
                stage="prep",
                quality=1.0 - (machine.defect_rate * 0.1),  # Slight quality variation
            )

            # Add to buffer
            if buffer.add_unit(unit):
                machine.units_produced += 1

    def _process_assembly(self, dt: float):
        """M2: Assemble components from buffer."""
        machine = self.machines["M2"]
        input_buffer = self.buffers["M1_M2"]
        output_buffer = self.buffers["M2_M3"]

        # Check input and output availability
        if input_buffer.current_level == 0:
            return  # No input, M2 starved

        if output_buffer.current_level >= output_buffer.capacity:
            return  # Output full, M2 blocked

        # Calculate how many we can process
        production_rate = machine.production_output
        units_to_process = int(production_rate * dt)
        units_available = min(units_to_process, input_buffer.current_level)

        for _ in range(units_available):
            if output_buffer.current_level >= output_buffer.capacity:
                break

            # Take unit from input
            units = input_buffer.remove_units(1)
            if not units:
                break

            unit = units[0]
            unit.stage = "assembly"

            # Quality may degrade based on machine condition
            if machine.health_score < 80:
                unit.quality *= 0.95

            # Add to output
            if output_buffer.add_unit(unit):
                machine.units_produced += 1

    def _process_finishing(self, dt: float):
        """M3: Apply finishing/coating."""
        machine = self.machines["M3"]
        input_buffer = self.buffers["M2_M3"]
        output_buffer = self.buffers["M3_M4"]

        if input_buffer.current_level == 0:
            return

        if output_buffer.current_level >= output_buffer.capacity:
            return

        production_rate = machine.production_output
        units_to_process = int(production_rate * dt)
        units_available = min(units_to_process, input_buffer.current_level)

        for _ in range(units_available):
            if output_buffer.current_level >= output_buffer.capacity:
                break

            units = input_buffer.remove_units(1)
            if not units:
                break

            unit = units[0]
            unit.stage = "finishing"

            # Quality may degrade based on machine condition
            if machine.temperature > 85:
                unit.quality *= 0.97

            if output_buffer.add_unit(unit):
                machine.units_produced += 1

    def _process_qc(self, dt: float) -> dict:
        """M4: Quality control inspection."""
        machine = self.machines["M4"]
        input_buffer = self.buffers["M3_M4"]

        produced = 0
        defects = 0

        if input_buffer.current_level == 0:
            return {"produced": 0, "defects": 0}

        # QC inspects units
        production_rate = machine.production_output
        units_to_inspect = int(production_rate * dt)
        units_available = min(units_to_inspect, input_buffer.current_level)

        for _ in range(units_available):
            units = input_buffer.remove_units(1)
            if not units:
                break

            unit = units[0]
            unit.stage = "qc"

            # QC inspection: quality threshold
            qc_threshold = 0.85
            if unit.quality >= qc_threshold:
                unit.passed_qc = True
                self.finished_products.append(unit)
                produced += 1
            else:
                unit.passed_qc = False
                unit.defect_type = "quality_below_threshold"
                defects += 1
                # Defective units are scrapped (not added to finished)

            machine.units_produced += 1

        return {"produced": produced, "defects": defects}

    def _identify_bottleneck(self) -> str:
        """Identify which machine is the bottleneck."""
        # Bottleneck is the machine with lowest production capacity
        # that's currently running or blocked

        min_rate = float('inf')
        bottleneck = None

        for machine_id, machine in self.machines.items():
            if machine.status in ["running", "idle"]:
                rate = machine.production_output
                if rate < min_rate:
                    min_rate = rate
                    bottleneck = machine_id

        return bottleneck

    def _update_metrics(self, dt: float):
        """Update line-wide performance metrics."""
        # Update total production
        total_produced = sum(m.units_produced for m in self.machines.values())
        self.metrics.total_produced = total_produced

        # QC pass rate
        total_finished = len(self.finished_products)
        if total_finished > 0:
            passed = sum(1 for u in self.finished_products if u.passed_qc)
            self.metrics.qc_pass_rate = passed / total_finished

        # Throughput rate (units/hour) - based on M4 output
        if dt > 0:
            m4_output = self.machines["M4"].units_produced
            self.metrics.throughput_rate = m4_output / dt if dt > 0 else 0.0

        # Line efficiency (how balanced the line is)
        speeds = [m.speed for m in self.machines.values() if m.status == "running"]
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
            # Efficiency is 1.0 when all speeds are equal, lower when imbalanced
            self.metrics.line_efficiency = 1.0 / (1.0 + variance / 100.0)

        # Energy consumption
        total_energy = sum(
            (m.speed / 100.0) * 0.1 * dt  # Simple energy model
            for m in self.machines.values()
        )
        self.metrics.total_energy += total_energy

        # Energy per unit
        if total_produced > 0:
            self.metrics.energy_per_unit = self.metrics.total_energy / total_produced

        # Bottleneck
        self.metrics.bottleneck_machine = self._identify_bottleneck()

    def get_line_status(self) -> dict:
        """Get complete status of production line."""
        return {
            "machines": {
                machine_id: {
                    "status": m.status,
                    "speed": m.speed,
                    "temperature": m.temperature,
                    "health_score": m.health_score,
                    "production_output": m.production_output,
                    "units_produced": m.units_produced,
                }
                for machine_id, m in self.machines.items()
            },
            "buffers": {
                buffer_id: {
                    "level": b.current_level,
                    "capacity": b.capacity,
                    "utilization": b.current_level / b.capacity if b.capacity > 0 else 0.0,
                }
                for buffer_id, b in self.buffers.items()
            },
            "metrics": {
                "total_produced": self.metrics.total_produced,
                "total_defects": self.metrics.total_defects,
                "throughput_rate": self.metrics.throughput_rate,
                "qc_pass_rate": self.metrics.qc_pass_rate,
                "line_efficiency": self.metrics.line_efficiency,
                "bottleneck": self.metrics.bottleneck_machine,
                "total_energy": self.metrics.total_energy,
                "energy_per_unit": self.metrics.energy_per_unit,
                "finished_products": len(self.finished_products),
            },
        }

    def optimize_line_speed(self, target: str = "balanced") -> dict:
        """
        Automatically optimize all machine speeds.

        Args:
            target: "throughput", "quality", "energy", or "balanced"

        Returns:
            Dictionary with old/new speeds and expected performance
        """
        old_speeds = {m_id: m.speed for m_id, m in self.machines.items()}

        if target == "throughput":
            # Set all to match the bottleneck's maximum sustainable speed
            bottleneck = self._identify_bottleneck()
            bottleneck_machine = self.machines[bottleneck]
            # Set based on bottleneck's health
            target_speed = min(80.0, bottleneck_machine.health_score * 0.8)

            for machine in self.machines.values():
                machine.set_speed(target_speed)

        elif target == "quality":
            # Lower speeds for better quality
            target_speed = 50.0
            for machine in self.machines.values():
                machine.set_speed(target_speed)

        elif target == "energy":
            # Moderate speeds for energy efficiency
            target_speed = 40.0
            for machine in self.machines.values():
                machine.set_speed(target_speed)

        else:  # balanced
            # Balance all machines to same speed
            avg_health = sum(m.health_score for m in self.machines.values()) / len(self.machines)
            target_speed = min(60.0, avg_health * 0.6)

            for machine in self.machines.values():
                machine.set_speed(target_speed)

        new_speeds = {m_id: m.speed for m_id, m in self.machines.items()}

        # Calculate expected performance
        expected_throughput = min(m.production_output for m in self.machines.values())
        expected_energy = sum((m.speed / 100.0) * 0.1 for m in self.machines.values())

        return {
            "old_speeds": old_speeds,
            "new_speeds": new_speeds,
            "bottleneck": self._identify_bottleneck(),
            "expected_throughput": expected_throughput,
            "expected_energy": expected_energy,
        }
