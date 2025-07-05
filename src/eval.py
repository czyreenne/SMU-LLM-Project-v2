import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk import sent_tokenize

nltk.download('punkt', quiet=True)

class SummaryChecker:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def tokenize_sentences(self, text):
        return sent_tokenize(text)

    def check_sentences(self, source, hypothesis):
        inputs = self.tokenizer(source, hypothesis, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        return probabilities

    def evaluate_summary(self, source_document, summary):
        source_sentences = self.tokenize_sentences(source_document)
        summary_sentences = self.tokenize_sentences(summary)

        total_contradiction = 0
        total_entailment = 0
        coverage_hits = [False] * len(source_sentences)
        count = 0

        for source_idx, source_sentence in enumerate(source_sentences):
            source_covered = False
            for summary_sentence in summary_sentences:
                result = self.check_sentences(source_sentence, summary_sentence)
                print(f"Entailment: {result[0][0].item()*100:.2f}%, Neutral: {result[0][1].item()*100:.2f}%, Contradiction: {result[0][2].item()*100:.2f}%")
                contradiction_probability = result[0][2].item()
                entailment_probability = result[0][0].item()
                total_contradiction += contradiction_probability
                total_entailment += entailment_probability
                count += 1

                if entailment_probability > 0.5:
                    source_covered = True

            coverage_hits[source_idx] = source_covered

        average_entailment = total_entailment / count if count else 0
        average_contradiction = total_contradiction / count if count else 0
        coverage_percentage = sum(coverage_hits) / len(source_sentences) * 100

        return {
            "Average Entailment Score": average_entailment * 100,
            "Average Contradiction Score": average_contradiction * 100,
            "Coverage Percentage": coverage_percentage
        }