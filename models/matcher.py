import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from config import MATCHING_CONFIG

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FaceMatcher:
    def __init__(self, config=None):
        self.config = config or MATCHING_CONFIG
        self.method = self.config.get('method', 'cosine')
        self.threshold = self.config.get('similarity_threshold', 0.6)
        self.top_k = self.config.get('top_k', 5)
        self.use_faiss = self.config.get('use_faiss', False) and HAS_FAISS
        
        self.embeddings_db = None
        self.identities_db = None
        self.faiss_index = None
    
    def build_index(self, embeddings: np.ndarray, identities: List[str]):
        self.embeddings_db = embeddings
        self.identities_db = identities
        
        if self.use_faiss and embeddings.shape[0] > 0:
            self._build_faiss_index(embeddings)
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        try:
            embeddings = embeddings.astype(np.float32)
            dimension = embeddings.shape[1]
            
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(embeddings)
            print(f"✓ FAISS index built with {embeddings.shape[0]} vectors")
        except Exception as e:
            print(f"⚠ Failed to build FAISS index: {e}")
            self.use_faiss = False
    
    def search(self, query_embedding: np.ndarray, threshold: float = None) -> List[Dict]:
        threshold = threshold or self.threshold
        
        if self.embeddings_db is None or len(self.embeddings_db) == 0:
            return []
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if self.use_faiss:
            matches = self._search_faiss(query_embedding, threshold)
        else:
            matches = self._search_similarity(query_embedding, threshold)
        
        return matches
    
    def _search_similarity(self, query_embedding: np.ndarray, threshold: float) -> List[Dict]:
        matches = []
        
        if self.method == 'cosine':
            # Compute cosine similarity
            similarities = 1 - cosine_distances(query_embedding, self.embeddings_db)[0]
            distances = cosine_distances(query_embedding, self.embeddings_db)[0]
        elif self.method == 'euclidean':
            # Compute Euclidean distance
            distances = euclidean_distances(query_embedding, self.embeddings_db)[0]
            similarities = 1 / (1 + distances)  # Convert distance to similarity
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Get top-k results
        top_indices = np.argsort(distances)[:self.top_k]
        
        for rank, idx in enumerate(top_indices):
            if similarities[idx] >= threshold:
                matches.append({
                    'identity': self.identities_db[idx],
                    'similarity': float(similarities[idx]),
                    'distance': float(distances[idx]),
                    'rank': rank + 1
                })
        
        return matches
    
    def _search_faiss(self, query_embedding: np.ndarray, threshold: float) -> List[Dict]:
        matches = []
        
        try:
            query_embedding = query_embedding.astype(np.float32)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_embedding, self.top_k)
            
            # Convert L2 distance to similarity
            for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                # L2 to cosine similarity approximation
                similarity = 1 / (1 + distance)
                
                if similarity >= threshold and idx < len(self.identities_db):
                    matches.append({
                        'identity': self.identities_db[idx],
                        'similarity': float(similarity),
                        'distance': float(distance),
                        'rank': rank + 1
                    })
        except Exception as e:
            print(f"FAISS search error: {e}")
            # Fallback to similarity search
            matches = self._search_similarity(query_embedding, threshold)
        
        return matches
    
    def batch_search(self, query_embeddings: np.ndarray, threshold: float = None) -> List[List[Dict]]:
        results = []
        for embedding in query_embeddings:
            matches = self.search(embedding, threshold)
            results.append(matches)
        
        return results
    
    def verify(self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = None) -> Dict:
        threshold = threshold or self.threshold
        
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        if self.method == 'cosine':
            similarity = 1 - cosine_distances(embedding1, embedding2)[0, 0]
            distance = cosine_distances(embedding1, embedding2)[0, 0]
        else:
            distance = euclidean_distances(embedding1, embedding2)[0, 0]
            similarity = 1 / (1 + distance)
        
        return {
            'match': similarity >= threshold,
            'similarity': float(similarity),
            'distance': float(distance)
        }


class SimilarityScorer:
    @staticmethod
    def compute_roc_scores(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        from sklearn.metrics import roc_curve, auc, roc_auc_score
        
        scores = []
        for i in range(len(embeddings) - 1):
            for j in range(i + 1, len(embeddings)):
                similarity = 1 - cosine_distances(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0, 0]
                scores.append(similarity)
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    
    @staticmethod
    def find_optimal_threshold(embeddings: np.ndarray, labels: np.ndarray) -> float:
        from sklearn.metrics import roc_curve
        
        scores = []
        for i in range(len(embeddings) - 1):
            for j in range(i + 1, len(embeddings)):
                similarity = 1 - cosine_distances(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0, 0]
                scores.append(similarity)
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find threshold with best F1 score
        f1_scores = 2 * (tpr * (1 - fpr)) / (tpr + 1 - fpr + 1e-10)
        best_idx = np.argmax(f1_scores)
        
        return float(thresholds[best_idx])