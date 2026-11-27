import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from padim_backbone import CrossMix, preprocess, layers, consistency, batch_rows, operating_points, adaptation_domains
from run import create_padim_backbone, padim_input, PADIM_BACKBONE, domain_aug, padim_prediction_maps

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

    def test_mix_rng_does_not_change_photometric_rng(self):
        x=np.linspace(.2,.8,20,dtype=np.float32).reshape(4,5)
        a,b=random.Random(42),random.Random(42)
        mix=CrossMix(p=1,seed=42)
        replica=CrossMix(p=1,seed=42)
        features=torch.arange(40,dtype=torch.float32).view(2,1,4,5)
        for _ in range(3):
            np.testing.assert_array_equal(domain_aug(x,a),domain_aug(x,b))
            torch.testing.assert_close(mix(features,["A","B"]),replica(features,["A","B"]))

    def test_preflight_counts_only_eligible_training_domains(self):
        rows=[dict(label=0,equipment="A"),dict(label=1,equipment="B"),dict(label=-1,equipment="C")]
        self.assertEqual(adaptation_domains("p2_classification_mixstyle",rows),["A","B"])
        self.assertEqual(adaptation_domains("p4_consistency_mixstyle",rows),["A"])
        self.assertIsNone(adaptation_domains("p5_pixel_no_mixstyle",rows))

    def test_direct_native_map_preserves_model_input_score(self):
        peak=torch.zeros(1,1,5,5)
        peak[0,0,2,2]=1
        crop=(0,0,20,20)
        restored, restored_score=padim_prediction_maps(peak,crop,(5,5),None,4)
        direct, direct_score=padim_prediction_maps(peak,crop,(5,5),None,4,direct_native=True)
        torch.testing.assert_close(restored_score,direct_score)
        torch.testing.assert_close(direct,peak)
        self.assertLess(restored.max().item(),direct.max().item())

    def test_letterbox_both_orientations(self):
        for shape in ((81,61),(61,81)):
            z,(t,l,h,w)=preprocess(torch.ones(2,1,*shape),64)
            self.assertEqual(tuple(z.shape),(2,3,64,64))
            self.assertEqual(max(h,w),64)
            self.assertEqual((t,l),((64-h)//2,(64-w)//2))

    def test_native_factor_four_and_depth_two(self):
        from run import restore_padim_map, restore_head_logits
        from torch.nn.functional import interpolate
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
            # The final anomaly score passes through the actual 4x input grid.
            input_map=interpolate(peak,size=z.shape[-2:],mode="bilinear",align_corners=False)
            expected=interpolate(input_map,size=shape,mode="bilinear",align_corners=False)
            actual=restore_padim_map(peak,crop,shape,None,4)
            torch.testing.assert_close(actual,expected)
            self.assertEqual(tuple(actual.shape),(1,1,*shape))
            self.assertAlmostEqual(float(actual[0,0,5,5]),.765625)
            self.assertGreater(float(actual[0,0,5,6]),0.)
            self.assertFalse(torch.equal(peak,actual))
            # The temporary supervision head does not receive this extra spatial smoothing.
            self.assertTrue(torch.equal(peak,restore_head_logits(peak,crop,shape,None,4)))

    def test_anomaly_map_letterbox_restoration_unchanged(self):
        from run import restore_padim_map
        from torch.nn.functional import interpolate
        shape=(61,81)
        _,crop=preprocess(torch.ones(1,1,*shape),64)
        top,left,h,w=crop
        score=torch.arange(16*16,dtype=torch.float32).reshape(1,1,16,16)
        enlarged=interpolate(score,size=(64,64),mode="bilinear",align_corners=False)
        expected=interpolate(enlarged[:,:,top:top+h,left:left+w],size=shape,
                             mode="bilinear",align_corners=False)
        torch.testing.assert_close(restore_padim_map(score,crop,shape,64,None),expected)

    def test_pred_score_is_max_before_native_downsample(self):
        from run import padim_prediction_maps
        from torch.nn.functional import interpolate
        for shape in ((61,81),(81,61)):
            distance=torch.zeros(2,1,*shape)
            distance[0,0,5,5]=1
            distance[0,0,5,6]=.8
            distance[1,0,10,10]=2
            distance[1,0,11,10]=.9
            crop=(0,0,shape[0]*4,shape[1]*4)
            high=interpolate(distance,size=crop[-2:],mode="bilinear",align_corners=False)
            expected=interpolate(high,size=shape,mode="bilinear",align_corners=False)
            native,score=padim_prediction_maps(distance,crop,shape,None,4,0.)
            torch.testing.assert_close(native,expected)
            torch.testing.assert_close(score,high.amax(dim=(-2,-1)).squeeze(1))
            self.assertTrue(torch.all(score>native.amax(dim=(-2,-1)).squeeze(1)))

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

        explicit=operating_points(rows,val,rows,test,None,
                                  val_image_scores=[3.,3.,8.],test_image_scores=[4.,4.,9.])
        target=next(p for p in explicit["points"] if p["kind"]=="image_recall" and p["target"]==.95)
        self.assertEqual(target["threshold"],8.)
        self.assertEqual(target["test"]["recall"],1.)
        self.assertEqual(target["test"]["fpr"],0.)

if __name__=="__main__":
    unittest.main()
