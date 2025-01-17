from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

@dataclass
class Document:
    content: str
    metadata: Dict
    embedding: np.ndarray = None

class QueryAnalyzer:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    
    def decompose_query(self, query: str) -> List[str]:
        """Generate multiple perspective queries from original query."""
        base_aspects = [
            f"Define {query}",
            f"Key components of {query}",
            f"Research findings about {query}",
            f"Practical applications of {query}"
        ]
        return base_aspects
    
    def reformulate_query(self, query: str, feedback: Dict) -> str:
        """Reformulate query based on retrieval feedback."""
        if feedback.get('missing_context'):
            return f"{query} AND ({' OR '.join(feedback['missing_context'])})"
        return query
    
class Retriever:
    def __init__(self, documents: List[Document], model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve relevant documents using semantic search."""
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        similarities = []
        for doc in self.documents:
            if doc.embedding is None:
                doc.embedding = self.model.encode(doc.content)
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc.embedding.reshape(1, -1)
            )[0][0]
            similarities.append((similarity, doc))
        
        return [doc for _, doc in sorted(similarities, reverse=True)[:top_k]]
    
class RelevanceAnalyzer:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2))
        
    def analyze(self, query: str, documents: List[Document]) -> Tuple[List[Document], Dict]:
        """Analyze relevance of retrieved documents."""
        relevant_docs = []
        missing_aspects = set()
        
        for doc in documents:
            if self._calculate_relevance_score(query, doc) > self.threshold:
                relevant_docs.append(doc)
            else:
                missing_aspects.add(self._extract_missing_aspect(doc))
                
        feedback = {'missing_context': list(missing_aspects)} if missing_aspects else {}
        return relevant_docs, feedback
    
    def _calculate_relevance_score(self, query: str, document: Document) -> float:
        """
        Calculate relevance score using multiple metrics:
        1. Semantic similarity using sentence transformers
        2. TF-IDF similarity
        3. Named entity overlap
        4. Key concept coverage
        5. Information density
        """
        # 1. Semantic Similarity (40% weight)
        query_embedding = self.sentence_transformer.encode(query)
        doc_embedding = self.sentence_transformer.encode(document.content)
        semantic_similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            doc_embedding.reshape(1, -1)
        )[0][0]
        
        # 2. TF-IDF Similarity (20% weight)
        tfidf_matrix = self.tfidf.fit_transform([query, document.content])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # 3. Named Entity Overlap (15% weight)
        query_doc = self.nlp(query)
        content_doc = self.nlp(document.content)
        
        query_entities = set([ent.text.lower() for ent in query_doc.ents])
        content_entities = set([ent.text.lower() for ent in content_doc.ents])
        
        entity_overlap = 0.0
        if query_entities:
            entity_overlap = len(query_entities.intersection(content_entities)) / len(query_entities)
        
        # 4. Key Concept Coverage (15% weight)
        query_concepts = set([token.text.lower() for token in query_doc if token.pos_ in ['NOUN', 'VERB']])
        content_concepts = set([token.text.lower() for token in content_doc if token.pos_ in ['NOUN', 'VERB']])
        
        concept_coverage = 0.0
        if query_concepts:
            concept_coverage = len(query_concepts.intersection(content_concepts)) / len(query_concepts)
        
        # 5. Information Density (10% weight)
        content_words = len([token for token in content_doc if not token.is_stop and not token.is_punct])
        total_words = len(document.content.split())
        information_density = content_words / total_words if total_words > 0 else 0
        
        # Calculate weighted score
        final_score = (
            0.4 * semantic_similarity +
            0.2 * tfidf_similarity +
            0.15 * entity_overlap +
            0.15 * concept_coverage +
            0.1 * information_density
        )
        
        return final_score
    
    def _extract_missing_aspect(self, document: Document) -> str:
        """
        Extract missing information aspects using:
        1. Topic modeling
        2. Key phrase extraction
        3. Semantic gap analysis
        """
        doc = self.nlp(document.content)
        
        # Extract key phrases using noun chunks and verb phrases
        noun_chunks = set([chunk.text.lower() for chunk in doc.noun_chunks])
        verb_phrases = set([token.text.lower() for token in doc if token.pos_ == 'VERB'])
        
        # Build semantic graph
        G = nx.Graph()
        
        # Add nodes for key phrases
        for phrase in noun_chunks:
            G.add_node(phrase, type='noun_chunk')
        for phrase in verb_phrases:
            G.add_node(phrase, type='verb')
            
        # Add edges based on semantic similarity
        phrases = list(noun_chunks.union(verb_phrases))
        if len(phrases) > 1:
            embeddings = self.sentence_transformer.encode(list(phrases))
            
            # Create edges between semantically similar phrases
            for i in range(len(phrases)):
                for j in range(i + 1, len(phrases)):
                    similarity = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0][0]
                    if similarity > 0.5:  # Threshold for semantic similarity
                        G.add_edge(phrases[i], phrases[j], weight=similarity)
        
        # Find main topics using PageRank
        if len(G.nodes()) > 0:
            pagerank = nx.pagerank(G)
            main_topics = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            
            # Return the highest-ranked topic as the missing aspect
            if main_topics:
                return main_topics[0][0]
        
        return "general_context"  # Default return if no specific aspect is identified
    
class Generator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def generate(self, query: str, documents: List[Document]) -> str:
        """
        Generate response using:
        1. Information fusion
        2. Coherence optimization
        3. Query-focused summarization
        """
        # Extract and organize information
        all_sentences = []
        sentence_embeddings = []
        sentence_scores = {}
        
        query_embedding = self.sentence_transformer.encode(query)
        
        for doc in documents:
            doc_sentences = [sent.text.strip() for sent in self.nlp(doc.content).sents]
            
            # Get embeddings for sentences
            sent_embeddings = self.sentence_transformer.encode(doc_sentences)
            
            # Score sentences based on query relevance
            for sent, embedding in zip(doc_sentences, sent_embeddings):
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                sentence_scores[sent] = similarity
                all_sentences.append(sent)
                sentence_embeddings.append(embedding)
        
        # Select top sentences while maintaining coherence
        selected_sentences = []
        covered_embeddings = []
        
        while len(selected_sentences) < min(10, len(all_sentences)):
            best_score = -1
            best_sentence = None
            best_embedding = None
            
            for sent, embedding in zip(all_sentences, sentence_embeddings):
                if sent in selected_sentences:
                    continue
                
                # Calculate coherence score with previously selected sentences
                coherence_score = 0
                if covered_embeddings:
                    coherence_score = np.mean([
                        cosine_similarity(
                            embedding.reshape(1, -1),
                            prev_embedding.reshape(1, -1)
                        )[0][0]
                        for prev_embedding in covered_embeddings
                    ])
                
                # Combined score considering both relevance and coherence
                combined_score = 0.7 * sentence_scores[sent] + 0.3 * coherence_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_sentence = sent
                    best_embedding = embedding
            
            if best_sentence:
                selected_sentences.append(best_sentence)
                covered_embeddings.append(best_embedding)
            else:
                break
        
        # Organize and format the response
        response = " ".join(selected_sentences)
        return response
    
class ResponseValidator:
    def __init__(self, completeness_threshold: float = 0.8):
        self.completeness_threshold = completeness_threshold
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
    def validate(self, response: str, query: str) -> Tuple[bool, Dict]:
        """Validate response completeness and accuracy."""
        completeness_score = self._calculate_completeness(response, query)
        
        if completeness_score < self.completeness_threshold:
            missing_aspects = self._identify_missing_aspects(response, query)
            return False, {'missing_context': missing_aspects}
        return True, {}
    
    def _calculate_completeness(self, response: str, query: str) -> float:
        """
        Calculate response completeness using:
        1. Topic coverage
        2. Question-answer alignment
        3. Information sufficiency
        """
        query_doc = self.nlp(query)
        response_doc = self.nlp(response)
        
        # 1. Topic Coverage (40% weight)
        query_topics = set([token.text.lower() for token in query_doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']])
        response_topics = set([token.text.lower() for token in response_doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']])
        
        topic_coverage = len(query_topics.intersection(response_topics)) / len(query_topics) if query_topics else 1.0
        
        # 2. Question-Answer Alignment (40% weight)
        query_embedding = self.sentence_transformer.encode(query)
        response_embedding = self.sentence_transformer.encode(response)
        
        alignment_score = cosine_similarity(
            query_embedding.reshape(1, -1),
            response_embedding.reshape(1, -1)
        )[0][0]
        
        # 3. Information Sufficiency (20% weight)
        # Check if response has enough detail based on sentence structure
        response_sentences = list(response_doc.sents)
        avg_sentence_length = np.mean([len(sent) for sent in response_sentences]) if response_sentences else 0
        sentence_complexity = min(avg_sentence_length / 20, 1.0)  # Normalize to [0,1]
        
        final_score = (
            0.4 * topic_coverage +
            0.4 * alignment_score +
            0.2 * sentence_complexity
        )
        
        return final_score
    
    def _identify_missing_aspects(self, response: str, query: str) -> List[str]:
        """
        Identify missing aspects using:
        1. Expected vs. actual content analysis
        2. Semantic gap detection
        """
        query_doc = self.nlp(query)
        response_doc = self.nlp(response)
        
        # Extract expected aspects from query
        query_aspects = {
            'entities': set([ent.text.lower() for ent in query_doc.ents]),
            'key_phrases': set([chunk.text.lower() for chunk in query_doc.noun_chunks]),
            'actions': set([token.text.lower() for token in query_doc if token.pos_ == 'VERB'])
        }
        
        # Extract covered aspects from response
        response_aspects = {
            'entities': set([ent.text.lower() for ent in response_doc.ents]),
            'key_phrases': set([chunk.text.lower() for chunk in response_doc.noun_chunks]),
            'actions': set([token.text.lower() for token in response_doc if token.pos_ == 'VERB'])
        }
        
        # Identify missing aspects
        missing_aspects = []
        
        # Check for missing entities
        missing_entities = query_aspects['entities'] - response_aspects['entities']
        if missing_entities:
            missing_aspects.extend(list(missing_entities))
        
        # Check for missing key phrases
        missing_phrases = query_aspects['key_phrases'] - response_aspects['key_phrases']
        if missing_phrases:
            missing_aspects.extend(list(missing_phrases))
        
        # Check for missing actions
        missing_actions = query_aspects['actions'] - response_aspects['actions']
        if missing_actions:
            missing_aspects.extend(list(missing_actions))
        
        return list(set(missing_aspects))  # Remove duplicates
    
class ActiveRAG:
    def __init__(self, documents: List[Document]):
        self.query_analyzer = QueryAnalyzer()
        self.retriever = Retriever(documents)
        self.relevance_analyzer = RelevanceAnalyzer()
        self.generator = Generator()
        self.validator = ResponseValidator()
        
    def process_query(self, query: str, max_iterations: int = 3) -> str:
        """Process query using active RAG pipeline."""
        current_query = query
        iteration = 0
        
        while iteration < max_iterations:
            # Generate sub-queries
            sub_queries = self.query_analyzer.decompose_query(current_query)
            
            all_relevant_docs = []
            for sub_query in sub_queries:
                # Retrieve documents
                retrieved_docs = self.retriever.retrieve(sub_query)
                
                # Analyze relevance
                relevant_docs, feedback = self.relevance_analyzer.analyze(sub_query, retrieved_docs)
                
                if feedback:
                    # Reformulate query based on feedback
                    current_query = self.query_analyzer.reformulate_query(current_query, feedback)
                    continue
                
                all_relevant_docs.extend(relevant_docs)
            
            # Generate response
            response = self.generator.generate(query, all_relevant_docs)
            
            # Validate response
            is_valid, feedback = self.validator.validate(response, query)
            
            if is_valid:
                return response
                
            # Update query based on validation feedback
            current_query = self.query_analyzer.reformulate_query(current_query, feedback)
            iteration += 1
        
        return response
    
documents = [
    Document(content="Mediterranean diet includes olive oil, vegetables, fruits, and whole grains.", 
            metadata={"source": "nutrition_guide"}),
    Document(content="Studies show Mediterranean diet reduces heart disease risk.", 
            metadata={"source": "medical_journal"})
]

active_rag = ActiveRAG(documents)
response = active_rag.process_query("What are the health benefits of Mediterranean diet?")