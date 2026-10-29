import torch
ckpt = torch.load("artifacts/checkpoints/best_mlflow_checkpoint.ckpt", map_location="cpu", weights_only=False)
for state in ckpt.get("callbacks", {}).values():
    if isinstance(state, dict):
        for k in ("best_model_path", "dirpath", "last_model_path"):
            if k in state:
                state[k] = ""
        state.get("best_k_models", {}).clear()   # keys are paths
torch.save(ckpt, "artifacts/checkpoints/best_mlflow_checkpoint.ckpt")

