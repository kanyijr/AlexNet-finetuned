import torch
from pathlib import Path



def save_model(model:torch.nn.Module,
               target_path:str,
               model_name:str):
    
    target_dir_path = Path(target_path)

    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model name should end with .pth or .pt"

    model_save_path = target_dir_path/model_name
    
    print(f"saving the model to {model_save_path}...")
    torch.save(obj=model.state_dict(), f=model_save_path)

    