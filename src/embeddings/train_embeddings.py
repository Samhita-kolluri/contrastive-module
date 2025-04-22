from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import json
import yaml
import logging
import os

def train_stance_model(data_path="data/contradiction_data.json", 
                       output_path="models/stance_model", 
                       config_path=os.path.join("configs", "models.yaml")):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load model from config
    model = SentenceTransformer(cfg.get("topic_embedding_model", "stance_embedding_model"))

    # Load dataset
    with open(data_path, "r") as f:
        pairs = json.load(f)

    # Format into InputExample (premise, hypothesis)
    train_examples = [
        InputExample(texts=[p["premise"], p["hypothesis"]], label=0.0)
        for p in pairs
    ]

    # Prepare dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=cfg["batch_size"])

    # Contrastive loss
    train_loss = losses.ContrastiveLoss(model=model)

    # Train model
    # model.fit(
    # train_objectives=[(train_dataloader, train_loss)],
    # epochs=cfg["epochs"],
    # warmup_steps=100,
    # optimizer_params={'lr': float(cfg["learning_rate"])},
    # show_progress_bar=True)

    # Save model
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)

    logging.basicConfig(level="INFO")
    logging.getLogger(__name__).info("Stance model saved at %s", output_path)


if __name__ == "__main__":
    train_stance_model()
