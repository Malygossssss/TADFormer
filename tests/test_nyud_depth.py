import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

from data.mtl_ds import NYUD_MT
from evaluation.eval_depth import eval_depth_predictions


def _prepare_minimal_nyud_root(root: Path):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "edge").mkdir(parents=True, exist_ok=True)
    (root / "segmentation").mkdir(parents=True, exist_ok=True)
    (root / "normals").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "gt_sets").mkdir(parents=True, exist_ok=True)

    (root / "gt_sets" / "val.txt").write_text("00001\n", encoding="utf-8")

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    segmentation = np.ones((2, 2), dtype=np.uint8)
    normals = np.dstack([
        np.ones((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    ])
    depth_mm = np.full((2, 2), 1000.0, dtype=np.float32)

    Image.fromarray(image).save(root / "images" / "00001.jpg")
    np.save(root / "edge" / "00001.npy", np.zeros((2, 2), dtype=np.float32))
    Image.fromarray(segmentation).save(root / "segmentation" / "00001.png")
    np.save(root / "normals" / "00001.npy", normals)
    np.save(root / "depth" / "00001.npy", depth_mm)


class NyudDepthTest(unittest.TestCase):
    def test_loader_converts_depth_from_mm_to_meters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "NYUD_MT"
            _prepare_minimal_nyud_root(root)

            dataset = NYUD_MT(root=str(root), split="val", do_depth=True)
            sample = dataset[0]

            np.testing.assert_allclose(sample["depth"], np.ones((2, 2), dtype=np.float32))

    def test_eval_depth_predictions_uses_mtl_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            root = tmp_path / "NYUD_MT"
            save_dir = tmp_path / "predictions"
            depth_pred_dir = save_dir / "depth"

            _prepare_minimal_nyud_root(root)
            depth_pred_dir.mkdir(parents=True, exist_ok=True)
            sio.savemat(depth_pred_dir / "00001.mat", {"depth": np.ones((2, 2), dtype=np.float32)})

            results = eval_depth_predictions("NYUD", str(save_dir), gt_root=str(root))

            self.assertAlmostEqual(results["rmse"], 0.0, places=6)
            self.assertAlmostEqual(results["log_rmse"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
