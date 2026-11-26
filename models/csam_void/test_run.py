import unittest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from run import VoidNet, MixStyle, loss_fn, threshold_for, metrics, copy_void, H, W, save_panel, pixel_metrics

class TestCSAM(unittest.TestCase):
    def test_native_shape_and_grad(self):
        model=VoidNet(mix=True)
        x=torch.rand(2,1,H,W)
        y=torch.zeros_like(x)
        y[0,0,30,40]=1
        z=model(x)
        self.assertEqual(z.shape,x.shape)
        loss=loss_fn(z,y,hard=True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
    def test_mix_eval_identity(self):
        x=torch.rand(2,16,H,W)
        self.assertTrue(torch.equal(MixStyle().eval()(x),x))
    def test_calibration_and_regions(self):
        y=np.zeros((H,W),np.float32)
        y[10,10]=1
        rows=[dict(label=1,y=y,equipment="a"),dict(label=0,y=y*0,equipment="b")]
        maps=np.zeros((2,H,W),np.float32)
        maps[0,10,10]=.6
        maps[0,20,20]=.9
        maps[1,20,20]=.7
        th=threshold_for(rows,maps,.95,.9)
        self.assertAlmostEqual(th,.6)
        m=metrics(rows,maps,th)
        self.assertEqual(m["regions"]["1px"]["recall"],1)
        self.assertEqual(m["fpr"],1)
    def test_three_panel_native_mask_and_fixed_heat_scale(self):
        for h, w in ((61, 81), (81, 61)):
            x = np.full((h, w), .3, np.float32)
            score = np.zeros_like(x)
            score[10, 20] = .6
            score[11, 21] = .3
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "panel.png"
                save_panel(x, score, .5, p)
                with Image.open(p) as image:
                    a = np.asarray(image)
                self.assertEqual(a.shape, (h+96, w*3+48, 3))
                left, top = 12+2*(w+12), 32
                original = a[top:top+h, 12:12+w, 0]
                np.testing.assert_array_equal(original, (x*255).astype(np.uint8))
                mask = a[top:top+h, left:left+w, 0]
                np.testing.assert_array_equal(mask, (score>=.5)*255)
                p2 = Path(d) / "panel2.png"
                score[0, 0] = 99
                save_panel(x, score, .5, p2)
                with Image.open(p2) as image:
                    b = np.asarray(image)
                cx, cy = 12+w+12+20, top+10
                np.testing.assert_array_equal(a[cy,cx], b[cy,cx])

    def test_aligned_standard_metrics(self):
        y = np.array([[1,0,0],[0,0,0]],dtype=np.float32)
        rows = [dict(label=1,y=y,equipment="a"),dict(label=0,y=y*0,equipment="a")]
        maps = np.stack([y*.9,y*0+.1])
        val = metrics(rows,maps,.5,"val")
        test = metrics(rows,maps,.5,"test")
        self.assertAlmostEqual(val["image_auroc"],1)
        self.assertAlmostEqual(val["pixel_auroc"],1)
        self.assertNotIn("pixel_aupro",val)
        self.assertAlmostEqual(test["image_f1"],1)
        self.assertAlmostEqual(test["pixel_aupro"],1)
        self.assertAlmostEqual(metrics(rows,maps,.95,"test")["image_f1"],0)
        tied = pixel_metrics(rows,np.zeros_like(maps),True)
        self.assertAlmostEqual(tied["pixel_auroc"],.5)
        self.assertAlmostEqual(tied["pixel_aupro"],.15)
        self.assertIsNone(pixel_metrics(rows[1:],maps[1:],True)["pixel_aupro"])

    def test_aupro_weights_regions_equally(self):
        # Two separate components: 1 pixel and 3 pixels.
        # Only the small component ranks above every background pixel.
        y = np.array([[1,0,0,0,1,1,1]],dtype=np.float32)
        z = np.array([[[.9,.5,.5,.5,.1,.1,.1]]],dtype=np.float32)
        result = pixel_metrics([dict(y=y)],z,True)
        self.assertAlmostEqual(result["pixel_aupro"],.5)
        self.assertAlmostEqual(result["pixel_auroc"],.25)

    def test_copy_paste(self):
        x=np.full((H,W),.3,np.float32)
        y=np.zeros_like(x)
        y[30,40]=1
        donor=x.copy()
        donor[30,40]=.8
        a,b=copy_void(x,dict(x=donor,y=y))
        self.assertEqual(b.sum(),1)
        self.assertGreater(a[30,40],x[30,40])
        self.assertTrue(np.array_equal(a[b==0],x[b==0]))

if __name__=="__main__":
    unittest.main()
