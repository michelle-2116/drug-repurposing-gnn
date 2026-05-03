#Training is done on Google Colab with a GPU runtime (T4 or A100)
#Runtime → Change runtime type → T4 GPU
#Run: !pip install pykeen
#Then run this script as a cell
import os
from pykeen.pipeline import pipeline
from pykeen.datasets import Hetionet

MODEL_DIR = "/content/models"
CHECKPOINT_DIR = "/content/checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

results = pipeline(
    dataset=Hetionet,
    model="RotatE",
    model_kwargs=dict(embedding_dim=256),
    optimizer="Adam",
    optimizer_kwargs=dict(lr=0.001),
    training_loop="sLCWA",
    training_kwargs=dict(
        num_epochs=300,
        batch_size=512,
        checkpoint_name="rotate_checkpoint.pt",
        checkpoint_directory=CHECKPOINT_DIR,
        checkpoint_frequency=50,
    ),
    evaluator_kwargs=dict(filtered=True),
    random_seed=42,
    device="cuda",
)

results.save_to_directory(MODEL_DIR)
print(f"Hits@10: {results.get_metric('hits@10'):.4f}")
print(f"Hits@1:  {results.get_metric('hits@1'):.4f}")
print(f"MRR:     {results.get_metric('mrr'):.4f}")
