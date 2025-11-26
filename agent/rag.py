"""RAG (Retrieval-Augmented Generation) for documentation search.

This module provides question-answering over the AgentBricks documentation.
"""

from pathlib import Path
from typing import Any, Dict, List

# Simple keyword-based RAG implementation
# In production, this would use embeddings and vector search


def search_documentation(query: str) -> List[Dict[str, Any]]:
    """Search documentation for relevant content.

    Args:
        query: The search query

    Returns:
        List of relevant documentation snippets
    """
    docs_path = Path("docs")
    if not docs_path.exists():
        return []

    results = []
    query_lower = query.lower()
    query_terms = query_lower.split()

    # Search through markdown files
    for md_file in docs_path.rglob("*.md"):
        try:
            content = md_file.read_text()
            content_lower = content.lower()

            # Simple relevance scoring
            score = 0
            for term in query_terms:
                score += content_lower.count(term)

            if score > 0:
                # Extract relevant snippet
                lines = content.split("\n")
                relevant_lines = []
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    if any(term in line_lower for term in query_terms):
                        # Get context around the line
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        snippet = "\n".join(lines[start:end])
                        relevant_lines.append(snippet)

                if relevant_lines:
                    results.append(
                        {
                            "file": str(md_file.relative_to(Path.cwd())),
                            "score": score,
                            "snippets": relevant_lines[:3],  # Top 3 snippets
                        }
                    )
        except Exception:
            continue

    # Sort by score
    def get_score(item: Dict[str, Any]) -> int:
        """Extract score as integer for sorting."""
        score = item.get("score", 0)
        if isinstance(score, int):
            return score
        try:
            return int(score)
        except (ValueError, TypeError):
            return 0

    results.sort(key=get_score, reverse=True)
    return results[:5]  # Top 5 results


def answer_question(question: str) -> str:
    """Answer a question using RAG over documentation.

    Args:
        question: The user's question

    Returns:
        str: Answer to the question
    """
    # Search documentation
    results = search_documentation(question)

    if not results:
        return """I couldn't find specific documentation for your question.

Here are some general tips:
- Check the README.md files in each brick directory
- Review the docs/ directory for detailed guides
- Use `agentbricks hint <task_number>` for task-specific help
- Check the code examples in the src/ directories

If you're stuck, try rephrasing your question or asking about a specific concept."""

    # Build answer from search results
    answer_parts = []

    # Add relevant snippets
    for result in results:
        answer_parts.append(f"**From {result['file']}:**\n")
        for snippet in result["snippets"]:
            answer_parts.append(f"```\n{snippet}\n```\n")

    # Add general guidance
    answer_parts.append("\n---\n")
    answer_parts.append("**Additional Resources:**\n")
    answer_parts.append("- Check the project README.md for overview")
    answer_parts.append("- Review brick-specific README files")
    answer_parts.append("- Use `agentbricks hint <task>` for task-specific hints")

    return "\n".join(answer_parts)
