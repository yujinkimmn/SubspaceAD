import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from padim_backbone import CrossMix, preprocess, layers, consistency, batch_rows, operating_points
from run import create_padim_backbone, padim_input, PADIM_BACKBONE

class TestPadimBackbone(unittest.TestCase):
    def test_cross_mix_and_eval_identity(self):
        torch.manual_seed(42); random.seed(42)
        x=torch.randn(2,3,8,8)
        x[1]+=10
        mix=CrossMix(p=1)
        self.assertTrue(torch.equal(x,mix(x,["A","A"])))
        y=mix(x,["A","B"])
        self.assertEqual(mix.applied,1)
        self.assertFalse(torch.equal(x,y))
        mix.eval()
        self.assertTrue(torch.equal(x,mix(x,["A","B"])))

    def test_letterbox_both_orientations(self):
        for shape in ((81,61),(61,81)):
            z,(t,l,h,w)=preprocess(torch.ones(2,1,*shape),64)
            self.assertEqual(tuple(z.shape),(2,3,64,64))
            self.assertEqual(max(h,w),64)
            self.assertEqual((t,l),((64-h)//2,(64-w)//2))

    def test_native_factor_four_and_depth_two(self):
        from run import restore_padim_map
        from padim_backbone import LocalHead
        net=create_padim_backbone(pretrained=False).eval()
        for shape in ((81,61),(61,81)):
            x=torch.ones(1,1,*shape)
            z,crop=preprocess(x,None,4)
            self.assertEqual(tuple(z.shape[-2:]),tuple(v*4 for v in shape))
            self.assertEqual(crop,(0,0,shape[0]*4,shape[1]*4))
            with torch.no_grad():
                fs=layers(net,z,depth=2)
                self.assertEqual(len(fs),2)
                self.assertEqual(tuple(fs[0].shape),(1,64,*shape))
                self.assertEqual(fs[1].shape[1],128)
                logits=LocalHead(2)(fs,crop,None,shape,4)
                self.assertEqual(tuple(logits.shape),(1,1,*shape))
            peak=torch.zeros(1,1,*shape)
            peak[:,:,5,5]=1
            self.assertTrue(torch.equal(peak,restore_padim_map(peak,crop,shape,None,4)))

    def test_transform_matches_original_dtype_resize_order(self):
        from torchvision.transforms import v2
        for dtype, maximum in ((np.uint8,255),(np.uint16,65535)):
            raw=np.arange(61*81,dtype=np.int64).reshape(61,81)% (maximum+1)
            raw=raw.astype(dtype)
            expected=v2.Compose([v2.ToImage(),v2.Resize((244,324),antialias=True),
                                 v2.ToDtype(torch.float32,scale=True)])(torch.from_numpy(raw[None]).repeat(3,1,1))
            z,crop=preprocess([padim_input(dict(raw=raw,x=raw.astype(np.float32)/maximum))],None,4)
            self.assertTrue(torch.equal(z[0],expected))
            self.assertTrue((z>=0).all() and (z<=1).all())
            self.assertEqual(crop,(0,0,244,324))

    def test_timm_local_checkpoint(self):
        net=create_padim_backbone(pretrained=False)
        self.assertEqual(net.pretrained_cfg["architecture"],"resnet18")
        self.assertEqual(net.pretrained_cfg["tag"],"a1_in1k")
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"a1_test.pth"
            torch.save(net.state_dict(),path)
            loaded=create_padim_backbone(path)
            self.assertTrue(all(torch.equal(v,loaded.state_dict()[k]) for k,v in net.state_dict().items()))
            old=Path(d)/"resnet18-f37072fd.pth"
            old.write_bytes(b"test")
            with self.assertRaisesRegex(ValueError,"not resnet18.a1_in1k"):
                create_padim_backbone(old)

    def test_matched_sampling(self):
        rows=[dict(label=i%2,equipment="A" if i<5 else "B",id=i) for i in range(10)]
        a=list(batch_rows(rows,4,3,True,42))
        random.seed(99)
        b=list(batch_rows(rows,4,3,True,42))
        self.assertEqual(a,b)
        self.assertTrue(all(len({r["equipment"] for r in batch})==2 for batch in a))

    def test_validation_thresholds_and_ties(self):
        normal=dict(label=0,y=np.zeros((2,2),np.float32))
        positive=dict(label=1,y=np.array([[1,0],[0,0]],np.float32))
        rows=[normal,normal,positive]
        val=np.array([[[1,0],[0,0]],[[1,0],[0,0]],[[2,0],[0,0]]],np.float32)
        test=np.array([[[3,0],[0,0]],[[3,0],[0,0]],[[4,0],[0,0]]],np.float32)
        result=operating_points(rows,val,rows,test,1.)
        zero=next(p for p in result["points"] if p["kind"]=="fpr_ceiling" and p["target"]==0.)
        self.assertEqual(zero["val"]["fpr"],0.)
        self.assertEqual(zero["test"]["fpr"],1.)
        self.assertFalse((val[0]>=zero["threshold"]).any())
        self.assertEqual(zero["val"]["region_1px"],1.)
        self.assertIsNone(zero["val"]["region_2_4px"])
        other=operating_points(rows,val,rows,test*100,1.)
        self.assertEqual([p["threshold"] for p in result["points"]],
                         [p["threshold"] for p in other["points"]])

if __name__=="__main__":
    unittest.main()
