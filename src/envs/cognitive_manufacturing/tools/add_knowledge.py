"""Tool for adding documents to knowledge base."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class AddKnowledgeTool(ManufacturingTool):
    """Add document to knowledge base."""

    @property
    def name(self) -> str:
        return "AddKnowledge"

    @property
    def description(self) -> str:
        return "Add a document to the knowledge base (maintenance guides, troubleshooting steps, safety procedures)"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Document title",
                },
                "content": {
                    "type": "string",
                    "description": "Document content (supports markdown)",
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["maintenance", "troubleshooting", "safety", "procedure", "general"],
                    "description": "Type of document",
                    "default": "general",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata (tags, machine_id, severity, etc.)",
                },
            },
            "required": ["title", "content"],
        }

    def execute(self, parameters: dict[str, Any], env: "CognitiveManufacturingEnvironment") -> ToolResult:
        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            return ToolResult(success=False, error=error)

        # Check if database and embedding service are enabled
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

        title = parameters["title"]
        content = parameters["content"]
        doc_type = parameters.get("doc_type", "general")
        metadata = parameters.get("metadata")

        try:
            # Generate embedding for the content
            embedding = env.embedding_service.embed_text(content)

            # Add to knowledge base
            doc_id = env.db_manager.add_knowledge(
                title=title,
                content=content,
                doc_type=doc_type,
                embedding=embedding,
                metadata=metadata,
            )

            return ToolResult(
                success=True,
                data={
                    "doc_id": doc_id,
                    "title": title,
                    "doc_type": doc_type,
                    "content_length": len(content),
                    "embedding_dimension": len(embedding),
                },
                message=f"Added knowledge article '{title}' to knowledge base (ID: {doc_id})",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to add knowledge: {str(e)}",
            )
