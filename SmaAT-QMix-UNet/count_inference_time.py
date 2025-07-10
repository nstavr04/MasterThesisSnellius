import time
import torch
from pathlib import Path

from root import ROOT_DIR
from utils import dataset_precip, model_classes

def measure_full_test_time(ckpt_path: Path,
                           batch_size: int = 8,
                           device: str = "cuda") -> float:
    # load and prepare model
    model_cls, model_name = model_classes.get_model_class(ckpt_path.name)
    model = model_cls.load_from_checkpoint(ckpt_path)
    model = model.to(device).eval()

    image = torch.rand(batch_size, 12, 288, 288, device=device)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print(torch.cuda.is_available())
    with torch.no_grad():
        final_time = 0
        for i in range(10):
            start.record()
            model(image)
            end.record()
            torch.cuda.synchronize()
            final_time += start.elapsed_time(end)

    return model_name, final_time / 10

def main():
    # adjust these paths if you need to
    ckpt_dir  = ROOT_DIR / "checkpoints" / "comparison"

    for ckpt in sorted(ckpt_dir.glob("*.ckpt")):
        elapsed = measure_full_test_time(ckpt)

        name, ms = measure_full_test_time(ckpt, batch_size=8)
        print(f"{name:60s}  →  {ms:.2f} ms")

if __name__ == "__main__":
    main()
