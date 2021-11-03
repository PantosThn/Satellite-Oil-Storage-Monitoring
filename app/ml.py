from pathlib import Path
from app.model import create_trained_model, make_prediction
from dataclasses import dataclass

@dataclass
class AIModel:
    model_path: Path

    def __post_init__(self):
        self.model = create_trained_model(self.model_path)


    def get_model(self):
        if not self.model:
            raise Exception("Model not implemeted")
        return self.model

    def predict_image(self, image):
        model = self.get_model()
        preds = make_prediction(model, image)

        return preds
