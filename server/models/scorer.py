from rouge_score import rouge_scorer

class TextScorer:
    @staticmethod
    def compute_rouge(generated_text, reference_text):
        """
        Compute the ROUGE-L score between generated and reference text.

        Args:
            generated_text (str): Model-generated text.
            reference_text (str): Ground truth/reference text.

        Returns:
            float: ROUGE-L F1 score.
        """
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, generated_text)
        return scores['rougeL'].fmeasure
