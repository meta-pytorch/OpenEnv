"""Tool for searching knowledge base using semantic search."""

from typing import Any, TYPE_CHECKING
from .base import ManufacturingTool, ToolResult

if TYPE_CHECKING:
    from ..server.environment import CognitiveManufacturingEnvironment


class SearchKnowledgeTool(ManufacturingTool):
    """Search knowledge base using semantic similarity."""

    @property
    def name(self) -> str:
        return "SearchKnowledge"

    @property
    def description(self) -> str:
        return "Search knowledge base using natural language query (semantic search for troubleshooting, maintenance, safety info)"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query (e.g., 'how to fix overheating', 'maintenance schedule')",
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["maintenance", "troubleshooting", "safety", "procedure", "general"],
                    "description": "Optional filter by document type",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["query"],
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

        query = parameters["query"]
        doc_type = parameters.get("doc_type")
        top_k = parameters.get("top_k", 5)

        try:
            # Generate query embedding
            query_embedding = env.embedding_service.embed_text(query)

            # Search knowledge base
            results = env.db_manager.search_knowledge(
                query_embedding=query_embedding,
                doc_type=doc_type,
                top_k=top_k,
            )

            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "doc_id": result["doc_id"],
                    "title": result["title"],
                    "content": result["content"],
                    "doc_type": result["doc_type"],
                    "similarity": round(result["similarity"], 4),
                    "metadata": result.get("metadata"),
                })

            if not results:
                message = f"No results found for query: '{query}'"
            else:
                best_match = results[0]
                message = (
                    f"Found {len(results)} results. "
                    f"Best match: '{best_match['title']}' ({best_match['similarity']:.1%} similar)"
                )

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": formatted_results,
                    "result_count": len(results),
                    "doc_type_filter": doc_type,
                },
                message=message,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Knowledge search failed: {str(e)}",
            )
