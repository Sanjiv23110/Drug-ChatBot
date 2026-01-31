"""
Retrieval Router - Main entry point for the retrieval system.

Provides clean interface for:
- Query processing
- Retrieval path selection
- Result formatting for LLM consumption

DESIGN: NO HARDCODED SECTIONS - works with dynamic section names
"""
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from app.retrieval.retrieve import RetrievalEngine, RetrievalResult, RetrievalPath
from app.retrieval.intent_classifier import QueryIntent

logger = logging.getLogger(__name__)


@dataclass
class FormattedContext:
    """Context formatted for LLM consumption."""
    # Main content
    context_text: str  # Concatenated section texts
    context_markdown: str  # Markdown version
    
    # Metadata
    drug_name: Optional[str] = None
    sections_found: List[str] = None
    image_paths: List[str] = None
    attribute_name: Optional[str] = None  # NEW: For attribute-scoped queries
    
    # Retrieval info
    path_used: str = ""
    total_chunks: int = 0
    
    # For citation
    sources: List[Dict[str, Any]] = None


class RetrievalRouter:
    """
    Main router for the retrieval system.
    
    Provides:
    - Simple query interface
    - Context formatting for LLM
    - Metadata for citations
    
    DYNAMIC: Works with any section names, no hardcoded enums
    """
    
    def __init__(self, enable_vector_fallback: bool = True):
        """
        Initialize the router.
        
        Args:
            enable_vector_fallback: Allow vector search as last resort
        """
        self.engine = RetrievalEngine(enable_vector_fallback=enable_vector_fallback)
        logger.info("RetrievalRouter initialized")
    
    async def route(self, query: str) -> FormattedContext:
        """
        Route a query and return formatted context.
        
        Args:
            query: User's natural language query
            
        Returns:
            FormattedContext ready for LLM consumption
        """
        # Retrieve using engine
        result = await self.engine.retrieve(query)
        
        # Format for LLM
        return self._format_context(result)
    
    async def route_with_result(self, query: str) -> tuple[FormattedContext, RetrievalResult]:
        """
        Route a query and return both formatted context and raw result.
        
        Useful for debugging and auditing.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (FormattedContext, RetrievalResult)
        """
        result = await self.engine.retrieve(query)
        formatted = self._format_context(result)
        return formatted, result
    
    def _format_context(self, result: RetrievalResult) -> FormattedContext:
        """
        Format retrieval result for LLM consumption.
        
        Args:
            result: Raw RetrievalResult
            
        Returns:
            FormattedContext with concatenated text
        """
        if not result.sections:
            return FormattedContext(
                context_text="No relevant information found in the drug monographs.",
                context_markdown="*No relevant information found.*",
                path_used=result.path_used.value,
                drug_name=result.drug_name,
                sections_found=[],
                image_paths=result.image_paths or [],
                total_chunks=0,
                sources=[]
            )
        
        # Build context text (plain)
        context_parts = []
        context_md_parts = []
        sources = []
        sections_found = set()
        
        for section in result.sections:
            # DYNAMIC: use section_name (string) instead of section_category (enum)
            section_name = section.get('section_name', 'unknown')
            sections_found.add(section_name)
            
            header = section.get('original_header', section_name)
            content = section.get('content_text', '')
            
            # Plain text format
            context_parts.append(f"[{section_name.upper()}]\n{content}")
            
            # Markdown format
            context_md_parts.append(f"## {header}\n\n{content}")
            
            # Source for citation
            sources.append({
                "drug_name": section.get('drug_name'),
                "section": section_name,
                "header": header,
                "page_start": section.get('page_start'),
                "char_count": section.get('char_count'),
                "content_text": content  # Add content for answer generation
            })
        
        # Join sections
        context_text = "\n\n---\n\n".join(context_parts)
        context_markdown = "\n\n---\n\n".join(context_md_parts)
        
        return FormattedContext(
            context_text=context_text,
            context_markdown=context_markdown,
            drug_name=result.drug_name,
            sections_found=list(sections_found),
            image_paths=result.image_paths or [],
            path_used=result.path_used.value,
            total_chunks=len(result.sections),
            sources=sources,
            # NEW: Propagate attribute for Answer Generator
            attribute_name=getattr(result, "attribute_name", None)
        )
    
    async def get_section(
        self,
        drug_name: str,
        section_name: str  # DYNAMIC - any string
    ) -> FormattedContext:
        """
        Direct section lookup (bypasses intent classification).
        
        Args:
            drug_name: Drug name (lowercase)
            section_name: Section name (DYNAMIC - any string)
            
        Returns:
            FormattedContext with section content
        """
        # Create a synthetic query
        query = f"{drug_name} {section_name.replace('_', ' ')}"
        return await self.route(query)
    
    async def get_structure_image(self, drug_name: str) -> FormattedContext:
        """
        Get chemical structure image for a drug.
        
        Args:
            drug_name: Drug name
            
        Returns:
            FormattedContext with image paths
        """
        query = f"show me the chemical structure of {drug_name}"
        return await self.route(query)
    
    async def list_drugs(self) -> List[str]:
        """Get list of all available drugs."""
        return await self.engine.get_available_drugs()
    
    async def list_sections(self, drug_name: str) -> List[str]:
        """Get available sections for a drug (DYNAMIC)."""
        return await self.engine.get_drug_sections(drug_name)
    
    async def list_all_section_types(self) -> List[Dict[str, Any]]:
        """
        Get all known section types with usage stats.
        
        Useful for building UI dropdowns or understanding the data.
        """
        return await self.engine.get_all_section_types()
    
    async def search_sections(
        self,
        drug_name: str,
        section_query: str
    ) -> List[Dict[str, Any]]:
        """
        Search for sections using fuzzy matching.
        
        Args:
            drug_name: Drug to search within
            section_query: Section name to search for
            
        Returns:
            List of matching sections with similarity scores
        """
        return await self.engine.search_sections(drug_name, section_query)


# Convenience functions
async def get_context(query: str) -> FormattedContext:
    """
    Get formatted context for a query.
    
    Args:
        query: User query
        
    Returns:
        FormattedContext ready for LLM
    """
    router = RetrievalRouter()
    return await router.route(query)


async def get_drug_section(drug_name: str, section_name: str) -> FormattedContext:
    """
    Direct lookup of a specific section (DYNAMIC).
    
    Args:
        drug_name: Drug name
        section_name: Section name (any string, will be fuzzy matched)
        
    Returns:
        FormattedContext with section content
    """
    router = RetrievalRouter()
    return await router.get_section(drug_name, section_name)
