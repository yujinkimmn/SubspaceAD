import unittest
import numpy as np
import torch
from run import VoidNet, MixStyle, loss_fn, threshold_for, metrics, copy_void, H, W

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
