import torch
from typing import Dict, List, Tuple, Optional
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SummaryEvaluator:
    """
    A summary evaluation class that assesses entailment between a source text and its summary.
    """
    
    def __init__(self, nli_model_name='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli', 
                entailment_threshold=0.84,
                device=None):
        """
        Initialize the SummaryEvaluator with the NLI model.
        
        Args:
            nli_model_name: Name of the pre-trained NLI model for entailment checking
            entailment_threshold: Threshold for determining entailment
            device: Device to run inference on ('cuda' or 'cpu')
        """
        
        # Set device for inference
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        
        # Load NLI model
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli_model = self.nli_model.to(self.device)
        
        self.entailment_threshold = entailment_threshold

    def extract_sentences(self, summary_text: str) -> List[str]:
        """
        Extract sentences from summary text.
        
        Args:
            summary_text: The summary text as a string
            
        Returns:
            List of sentences from the summary
        """
        return sent_tokenize(summary_text)

    def _get_entailment_score(self, premise: str, hypothesis: str) -> float:
        """
        Get NLI-based entailment score between premise and hypothesis.
        
        Args:
            premise: Source text
            hypothesis: Target text (typically a sentence from summary)
            
        Returns:
            Entailment score (0-1)
        """
        inputs = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.nli_model(**inputs)
        logits = outputs.logits[0, [0, 2]]  # Entailment and contradiction scores
        return torch.softmax(logits, -1).tolist()[0]  # Return entailment probability

    def calculate_entailment(self, sentence: str, source_text: str) -> Tuple[float, bool]:
        """
        Calculate entailment score for a sentence.
        
        Args:
            sentence: Summary sentence to evaluate
            source_text: Source text to check against
            
        Returns:
            Tuple of (entailment score, is_entailed flag)
        """
        # Fast path: if sentence is contained in source, it's fully entailed
        if sentence in source_text:
            return 1.0, True
        
        # Otherwise, use NLI model to calculate entailment
        score = self._get_entailment_score(source_text, sentence)
        return score, score >= self.entailment_threshold

    def evaluate_summary(self, source_text: str, summary_text: str) -> Dict:
        """
        Evaluate summary for entailment.
        
        Args:
            source_text: Original source document text
            summary_text: Generated summary text
            
        Returns:
            Dictionary with entailment scores and flagged sentences
        """
        # Extract sentences from summary
        sentences = self.extract_sentences(summary_text)
        
        # Calculate entailment for each sentence
        flagged_sentences = []
        total_score = 0
        sentence_scores = {}
        
        for sentence in sentences:
            score, is_entailed = self.calculate_entailment(sentence, source_text)
            sentence_scores[sentence] = score
            
            if not is_entailed:
                flagged_sentences.append(sentence)
                
            total_score += score
        
        # Calculate overall entailment score
        sentence_count = len(sentences)
        overall_score = total_score / sentence_count if sentence_count > 0 else 0
        
        # Compile results
        return {
            "Entailment Score": overall_score,
            "Sentence Scores": sentence_scores,
            "Flagged Sentences": flagged_sentences
        }

