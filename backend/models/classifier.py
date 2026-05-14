from transformers import pipeline


class MentalHealthClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )

    def predict(self, text):
        results = self.classifier(text)

        # results is a list containing another list
        predictions = results[0]

        sorted_results = sorted(
            predictions,
            key=lambda x: x["score"], # type: ignore
            reverse=True
        )

        top_prediction = sorted_results[0]

        return {
            "text": text,
            "emotion": top_prediction["label"], # type: ignore
            "confidence": round(top_prediction["score"], 4), # type: ignore
            "all_predictions": sorted_results
        }


if __name__ == "__main__":
    classifier = MentalHealthClassifier()

    sample_text = "I feel anxious and stressed about my future."

    prediction = classifier.predict(sample_text)

    print(prediction)