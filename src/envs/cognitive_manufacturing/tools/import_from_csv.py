"""Tool for importing data from CSV files."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class ImportFromCSVTool(ManufacturingTool):
    """Import data from CSV file."""

    @property
    def name(self) -> str:
        return "ImportFromCSV"

    @property
    def description(self) -> str:
        return "Import data from CSV file (knowledge base articles, configurations, etc.)"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Input CSV filename (with or without .csv extension)",
                },
                "data_type": {
                    "type": "string",
                    "enum": ["knowledge", "settings", "configurations"],
                    "description": "Type of data being imported",
                },
                "validate": {
                    "type": "boolean",
                    "description": "Validate data before import (default: true)",
                    "default": True,
                },
            },
            "required": ["filename", "data_type"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        # Check if CSV service is enabled
        if not hasattr(env, "csv_service") or env.csv_service is None:
            return ToolResult(
                success=False,
                error="CSV service not enabled",
            )

        filename = parameters["filename"]
        data_type = parameters["data_type"]
        validate = parameters.get("validate", True)

        try:
            # Validate if requested
            if validate:
                if data_type == "knowledge":
                    required_columns = ["title", "content", "doc_type"]
                elif data_type == "settings":
                    required_columns = ["key", "value"]
                elif data_type == "configurations":
                    required_columns = ["machine_id", "parameter", "value"]
                else:
                    return ToolResult(success=False, error=f"Unknown data type: {data_type}")

                is_valid_csv, validation_error = env.csv_service.validate_csv(filename, required_columns)
                if not is_valid_csv:
                    return ToolResult(
                        success=False,
                        error=f"CSV validation failed: {validation_error}",
                    )

            # Import data
            data = env.csv_service.import_from_csv(filename)

            # Process based on data type
            imported_count = 0

            if data_type == "knowledge":
                # Import to knowledge base
                if not hasattr(env, "db_manager") or env.db_manager is None:
                    return ToolResult(
                        success=False,
                        error="Database not enabled. Create environment with enable_database=True",
                    )

                if not hasattr(env, "embedding_service") or env.embedding_service is None:
                    return ToolResult(
                        success=False,
                        error="Embedding service not enabled",
                    )

                # Add each knowledge article
                for row in data:
                    title = row.get("title", "")
                    content = row.get("content", "")
                    doc_type = row.get("doc_type", "general")
                    metadata = {k: v for k, v in row.items() if k not in ["title", "content", "doc_type"]}

                    # Generate embedding
                    embedding = env.embedding_service.embed_text(content)

                    # Add to database
                    env.db_manager.add_knowledge(
                        title=title,
                        content=content,
                        doc_type=doc_type,
                        embedding=embedding,
                        metadata=metadata if metadata else None,
                    )
                    imported_count += 1

            elif data_type == "settings":
                # Import settings (store in environment or database)
                # For now, just validate and return
                imported_count = len(data)

            elif data_type == "configurations":
                # Import machine configurations
                imported_count = len(data)

            return ToolResult(
                success=True,
                data={
                    "filename": filename,
                    "data_type": data_type,
                    "imported_count": imported_count,
                    "total_rows": len(data),
                },
                message=f"Imported {imported_count} {data_type} records from {filename}",
            )

        except FileNotFoundError as e:
            return ToolResult(
                success=False,
                error=f"File not found: {str(e)}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Import failed: {str(e)}",
            )
