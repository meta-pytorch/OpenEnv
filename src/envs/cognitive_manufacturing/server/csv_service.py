"""CSV export and import service for data interchange."""

import csv
import os
from pathlib import Path
from typing import Any
from datetime import datetime

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class CSVService:
    """Handles CSV export and import operations."""

    def __init__(self, export_dir: str = "data/exports"):
        """Initialize CSV service.

        Args:
            export_dir: Directory for exported CSV files
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_to_csv(self, data: list[dict], filename: str) -> str:
        """Export data to CSV file.

        Args:
            data: List of dictionaries to export
            filename: Output filename (with or without .csv extension)

        Returns:
            Full path to exported file
        """
        if not filename.endswith(".csv"):
            filename += ".csv"

        filepath = self.export_dir / filename

        if PANDAS_AVAILABLE:
            # Use pandas for better handling of complex data
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        else:
            # Fallback to csv module
            if not data:
                # Empty data, create empty file
                with open(filepath, "w", newline="") as f:
                    f.write("")
                return str(filepath)

            # Get all unique keys from all dictionaries
            fieldnames = set()
            for row in data:
                fieldnames.update(row.keys())
            fieldnames = sorted(fieldnames)

            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)

        return str(filepath)

    def import_from_csv(self, filename: str) -> list[dict]:
        """Import data from CSV file.

        Args:
            filename: Input filename (with or without .csv extension)

        Returns:
            List of dictionaries

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not filename.endswith(".csv"):
            filename += ".csv"

        # Check multiple locations
        possible_paths = [
            Path(filename),  # Absolute or relative path
            self.export_dir / filename,  # In export directory
            Path("data") / filename,  # In data directory
        ]

        filepath = None
        for path in possible_paths:
            if path.exists():
                filepath = path
                break

        if not filepath:
            raise FileNotFoundError(f"CSV file not found: {filename}")

        if PANDAS_AVAILABLE:
            # Use pandas for better handling
            df = pd.read_csv(filepath)
            return df.to_dict(orient="records")
        else:
            # Fallback to csv module
            data = []
            with open(filepath, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(dict(row))
            return data

    def validate_csv(self, filename: str, required_columns: list[str]) -> tuple[bool, str | None]:
        """Validate CSV file has required columns.

        Args:
            filename: CSV filename
            required_columns: List of required column names

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            data = self.import_from_csv(filename)
            if not data:
                return False, "CSV file is empty"

            # Check required columns
            first_row = data[0]
            missing_columns = [col for col in required_columns if col not in first_row]

            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"

            return True, None

        except FileNotFoundError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Error validating CSV: {str(e)}"

    def get_export_path(self, filename: str) -> str:
        """Get full path for an export file.

        Args:
            filename: Filename

        Returns:
            Full path string
        """
        if not filename.endswith(".csv"):
            filename += ".csv"
        return str(self.export_dir / filename)

    def list_exports(self) -> list[str]:
        """List all exported CSV files.

        Returns:
            List of filenames in export directory
        """
        return [f.name for f in self.export_dir.glob("*.csv")]

    def delete_export(self, filename: str) -> bool:
        """Delete an exported CSV file.

        Args:
            filename: Filename to delete

        Returns:
            True if deleted, False if file didn't exist
        """
        if not filename.endswith(".csv"):
            filename += ".csv"

        filepath = self.export_dir / filename
        if filepath.exists():
            filepath.unlink()
            return True
        return False
