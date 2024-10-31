import editdistance
from sklearn.metrics import precision_score, recall_score, f1_score

class Metric:
    def evaluate(self, expected, predicted):
        raise NotImplementedError("Subclasses should implement this method.")

class CharacterErrorRateMetric(Metric):
    def evaluate(self, expected, predicted):
        total_distance = sum(editdistance.eval(e, p) for e, p in zip(expected, predicted))
        total_chars = sum(len(e) for e in expected)
        return total_distance / total_chars if total_chars > 0 else 0

class WordErrorRateMetric(Metric):
    def evaluate(self, expected, predicted):
        total_distance = sum(editdistance.eval(e.split(), p.split()) for e, p in zip(expected, predicted))
        total_words = sum(len(e.split()) for e in expected)
        return total_distance / total_words if total_words > 0 else 0

class PrecisionMetric(Metric):
    def evaluate(self, expected, predicted):
        y_true = [char for line in expected for char in line]
        y_pred = [char for line in predicted for char in line]
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]
        return precision_score(y_true, y_pred, average='micro', zero_division=0)

class RecallMetric(Metric):
    def evaluate(self, expected, predicted):
        y_true = [char for line in expected for char in line]
        y_pred = [char for line in predicted for char in line]
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]
        return recall_score(y_true, y_pred, average='micro', zero_division=0)

class F1ScoreMetric(Metric):
    def evaluate(self, expected, predicted):
        y_true = [char for line in expected for char in line]
        y_pred = [char for line in predicted for char in line]
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]
        return f1_score(y_true, y_pred, average='micro', zero_division=0)
