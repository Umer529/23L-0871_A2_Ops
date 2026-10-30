"""Task 4.1 evidence - run the inference module on image files from the console.

  python app/predict_cli.py data/raw/AiArtData/x.jpg data/raw/RealArt/y.jpg
"""
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]

from config import Config  # noqa: E402
from inference import LocalPredictor  # noqa: E402

cfg = Config.from_env()
t0 = time.perf_counter()
predictor = LocalPredictor(cfg.model_path, cfg.threshold)
print(f"model loaded once in {time.perf_counter() - t0:.2f}s: {predictor.model_name} "
      f"v{predictor.model_version}, img_size={predictor.img_size}, threshold={predictor.threshold}")
for path in sys.argv[1:]:
    r = predictor.predict(Path(path).read_bytes())
    print(f"{Path(path).parent.name}/{Path(path).name[:40]:40s} -> {r['label']:13s} "
          f"confidence={r['confidence']:.3f}  p_ai={r['p_ai']:.3f}  {r['latency_ms']:.0f} ms")
