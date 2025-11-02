"""Base class for extracting documents from SEC submission .txt files.

Inspired by secedgar.parser.MetaParser, this provides reusable logic for extracting
embedded documents from SEC EDGAR submission files.

SEC submission format:
    <SEC-DOCUMENT>
        <SEC-HEADER>...metadata...</SEC-HEADER>
        <DOCUMENT>
            <TYPE>10-K</TYPE>
            <SEQUENCE>1</SEQUENCE>
            <FILENAME>form10k.htm</FILENAME>
            <TEXT>...content...</TEXT>
        </DOCUMENT>
        <DOCUMENT>
            <TYPE>EX-99.1</TYPE>
            <SEQUENCE>2</SEQUENCE>
            <FILENAME>exhibit99_1.htm</FILENAME>
            <TEXT>...content...</TEXT>
        </DOCUMENT>
    </SEC-DOCUMENT>
"""

import re
from typing import Optional, List, Dict, Any


class DocumentMetadata:
    """Metadata for an embedded document within a submission."""
    
    def __init__(
        self,
        doc_type: Optional[str] = None,
        sequence: Optional[str] = None,
        filename: Optional[str] = None,
        description: Optional[str] = None
    ):
        self.type = doc_type
        self.sequence = sequence
        self.filename = filename
        self.description = description
    
    def __repr__(self):
        return f"DocumentMetadata(type={self.type}, sequence={self.sequence}, filename={self.filename})"


class ExtractedDocument:
    """Container for an extracted document with its metadata and content."""
    
    def __init__(self, metadata: DocumentMetadata, content: str):
        self.metadata = metadata
        self.content = content
    
    def __repr__(self):
        content_preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"ExtractedDocument(type={self.metadata.type}, content_length={len(self.content)})"


class BaseDocumentExtractor:
    """Base class for extracting documents from SEC submission files.
    
    Provides methods to:
    - Extract individual documents by type (e.g., "EX-99.1")
    - Extract all documents from a submission
    - Parse document metadata (type, sequence, filename)
    
    Subclasses can override or extend these methods for specific filing types.
    """
    
    def __init__(self):
        """Initialize regex patterns for document extraction."""
        # Match <DOCUMENT>...</DOCUMENT> blocks
        self.re_doc = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", flags=re.DOTALL | re.IGNORECASE)
        
        # Match <TEXT>...</TEXT> content
        self.re_text = re.compile(r"<TEXT>(.*?)</TEXT>", flags=re.DOTALL | re.IGNORECASE)
        
        # Match metadata fields
        self.re_type = re.compile(r"<TYPE>(.*?)(?:\n|</TYPE>)", flags=re.IGNORECASE)
        self.re_sequence = re.compile(r"<SEQUENCE>(.*?)(?:\n|</SEQUENCE>)", flags=re.IGNORECASE)
        self.re_filename = re.compile(r"<FILENAME>(.*?)(?:\n|</FILENAME>)", flags=re.IGNORECASE)
        self.re_description = re.compile(r"<DESCRIPTION>(.*?)(?:\n|</DESCRIPTION>)", flags=re.IGNORECASE)
    
    def extract_document_metadata(self, doc_text: str) -> DocumentMetadata:
        """Extract metadata from a document block.
        
        Args:
            doc_text: Text of a <DOCUMENT>...</DOCUMENT> block
            
        Returns:
            DocumentMetadata object with parsed fields
        """
        metadata = DocumentMetadata()
        
        # Extract TYPE
        type_match = self.re_type.search(doc_text)
        if type_match:
            metadata.type = type_match.group(1).strip()
        
        # Extract SEQUENCE
        seq_match = self.re_sequence.search(doc_text)
        if seq_match:
            metadata.sequence = seq_match.group(1).strip()
        
        # Extract FILENAME
        fn_match = self.re_filename.search(doc_text)
        if fn_match:
            metadata.filename = fn_match.group(1).strip()
        
        # Extract DESCRIPTION (optional)
        desc_match = self.re_description.search(doc_text)
        if desc_match:
            metadata.description = desc_match.group(1).strip()
        
        return metadata
    
    def extract_document_content(self, doc_text: str) -> Optional[str]:
        """Extract the <TEXT>...</TEXT> content from a document block.
        
        Args:
            doc_text: Text of a <DOCUMENT>...</DOCUMENT> block
            
        Returns:
            Content between <TEXT> tags, or None if not found
        """
        text_match = self.re_text.search(doc_text)
        if text_match:
            return text_match.group(1).strip()
        return None
    
    def extract_all_documents(self, txt_content: str) -> List[ExtractedDocument]:
        """Extract all documents from a submission file.
        
        Args:
            txt_content: Full text content of SEC submission .txt file
            
        Returns:
            List of ExtractedDocument objects
        """
        documents = []
        
        # Find all <DOCUMENT> blocks
        for doc_match in self.re_doc.finditer(txt_content):
            doc_text = doc_match.group(1)
            
            # Extract metadata
            metadata = self.extract_document_metadata(doc_text)
            
            # Extract content
            content = self.extract_document_content(doc_text)
            
            if content:
                documents.append(ExtractedDocument(metadata, content))
        
        return documents
    
    def extract_document_by_type(
        self, 
        txt_content: str, 
        doc_type: str,
        exact_match: bool = False
    ) -> Optional[ExtractedDocument]:
        """Extract a specific document by its TYPE field.
        
        Args:
            txt_content: Full text content of SEC submission .txt file
            doc_type: Document type to search for (e.g., "EX-99.1", "10-K")
            exact_match: If True, type must match exactly. If False, allows prefix matching
                        (e.g., "EX-99" matches "EX-99.1", "EX-99.2")
            
        Returns:
            First matching ExtractedDocument, or None if not found
        """
        documents = self.extract_all_documents(txt_content)
        
        doc_type_upper = doc_type.upper()
        
        for doc in documents:
            if not doc.metadata.type:
                continue
            
            doc_type_check = doc.metadata.type.upper()
            
            if exact_match:
                if doc_type_check == doc_type_upper:
                    return doc
            else:
                # Prefix match (e.g., "EX-99" matches "EX-99.1")
                if doc_type_check.startswith(doc_type_upper):
                    return doc
        
        return None
    
    def extract_documents_by_type_pattern(
        self,
        txt_content: str,
        type_pattern: str
    ) -> List[ExtractedDocument]:
        """Extract all documents matching a type pattern.
        
        Args:
            txt_content: Full text content of SEC submission .txt file
            type_pattern: Regex pattern to match document types (e.g., "EX-99.*")
            
        Returns:
            List of matching ExtractedDocument objects
        """
        documents = self.extract_all_documents(txt_content)
        pattern = re.compile(type_pattern, re.IGNORECASE)
        
        matching_docs = []
        for doc in documents:
            if doc.metadata.type and pattern.match(doc.metadata.type):
                matching_docs.append(doc)
        
        return matching_docs
    
    def get_document_types(self, txt_content: str) -> List[str]:
        """Get a list of all document types in a submission.
        
        Useful for debugging and understanding submission structure.
        
        Args:
            txt_content: Full text content of SEC submission .txt file
            
        Returns:
            List of document type strings (e.g., ["10-K", "EX-99.1", "EX-99.2"])
        """
        documents = self.extract_all_documents(txt_content)
        return [doc.metadata.type for doc in documents if doc.metadata.type]
