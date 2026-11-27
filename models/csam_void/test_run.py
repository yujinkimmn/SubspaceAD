import unittest
import tempfile
import csv
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
import numpy as np
import torch
from run import (VoidNet, MixStyle, loss_fn, threshold_for, metrics, copy_void, H, W,
                 save_panel, pixel_metrics, calibrate_post_processor, post_process, normalize_scores, jet_heatmap,
                 load_manifest, panel_metadata_lines, save_eval, select_experiment, validation_constraints)

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
                scale = 4
                ph, pw = h*scale, w*scale
                expand = lambda v: np.repeat(np.repeat(v,scale,axis=0),scale,axis=1)
                self.assertEqual(a.shape, (ph+96, pw*3+48, 3))
                left, top = 12+2*(pw+12), 32
                original = a[top:top+ph, 12:12+pw, 0]
                np.testing.assert_array_equal(original, expand((x*255).astype(np.uint8)))
                mask = a[top:top+ph, left:left+pw, 0]
                np.testing.assert_array_equal(mask, expand((score>.5)*255))
                np.testing.assert_array_equal(a[top:top+ph,12+pw+12:12+2*pw+12],expand(jet_heatmap(score)))
                p2 = Path(d) / "panel2.png"
                score[0, 0] = 99
                save_panel(x, score, .5, p2)
                with Image.open(p2) as image:
                    b = np.asarray(image)
                cx, cy = 12+pw+12+20*scale, top+10*scale
                np.testing.assert_array_equal(a[cy,cx], b[cy,cx])

    def test_manifest_optional_wafer_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            Image.fromarray(np.zeros((H,W),np.uint8)).save(root/"image.png")
            for extra, values, expected in (([], [], ""), (["wafer"], ["007"], "007"),
                                            (["wafer_id"], ["008"], "008"),
                                            (["wafer", "wafer_id"], ["", "009"], "009")):
                path=root/"manifest.csv"
                with path.open("w",newline="",encoding="utf-8") as f:
                    writer=csv.writer(f)
                    writer.writerow(["image","mask","label","split","equipment","lot",*extra])
                    writer.writerow(["image.png","",0,"test","EQ-1","LOT-1",*values])
                row=load_manifest(path)[0]
                self.assertEqual(row["wafer"],expected)
                self.assertEqual(panel_metadata_lines(row)[1],f"Wafer: {expected or 'N/A'}")

    def test_panel_metadata_and_panel_only_save(self):
        for h,w in ((61,81),(81,61)):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)
                x=np.full((h,w),.3,np.float32)
                y=np.zeros_like(x); y[10,20]=1; y[10,21]=1
                raw=y*.8
                cal=dict(image_threshold=.4,image_min=0.,image_max=1.,pixel_threshold=.5,pixel_min=0.,pixel_max=1.)
                row=dict(id="7",image="example.png",x=x,y=y,label=1,equipment="EQ-1",lot="L"*90,wafer="007")
                lines=panel_metadata_lines(row,True,.9)
                self.assertEqual(lines,["Lot: "+"L"*90,"Wafer: 007","Equipment: EQ-1",
                                        "GT: VOID (1) | GT pixels: 2","Image prediction: VOID (1) | Score: 0.9"])
                save_eval([row],raw[None],cal,root,"test",panel_scale=1,collect_comparison=True)
                self.assertEqual({p.name for p in (root/"test").iterdir()},{"scores.csv","7_panel.png"})
                self.assertTrue((root/"comparison_stats.json").is_file())
                with (root/"test"/"scores.csv").open(newline="",encoding="utf-8") as f:
                    rec=next(csv.DictReader(f))
                self.assertEqual(rec["wafer"],"007")
                self.assertEqual(rec["lot"],row["lot"])
                with Image.open(root/"test"/"7_panel.png") as im:
                    a=np.asarray(im)
                top=a.shape[0]-h-64
                self.assertGreater(top,32)
                self.assertGreater(a.shape[1],3*w+48)  # long metadata expands only the canvas
                self.assertTrue((a[:top-32]!=255).any())
                norm=post_process(raw[None],cal)[1][0]
                np.testing.assert_array_equal(a[top:top+h,12:12+w,0],(x*255).astype(np.uint8))
                np.testing.assert_array_equal(a[top:top+h,24+w:24+2*w],jet_heatmap(norm))
                left=12+2*(w+12)
                np.testing.assert_array_equal(a[top:top+h,left:left+w,0],(norm>.5)*255)
                save_eval([row],raw[None],cal,root,"val",save_raw_maps=True,panel_scale=1)
                self.assertEqual({p.name for p in (root/"val").iterdir()},
                                 {"scores.csv","7_panel.png","7.npy","7_normalized.npy"})
                np.testing.assert_array_equal(np.load(root/"val"/"7.npy"),raw)

    def test_anomalib_normalization_and_independent_thresholds(self):
        y=np.array([[1,0],[0,0]],np.float32)
        rows=[dict(label=1,y=y,equipment="a"),dict(label=0,y=y*0,equipment="b")]
        maps=np.array([[[6,4],[4,2]],[[5,1],[1,1]]],np.float32)
        cal=calibrate_post_processor(rows,maps)
        self.assertEqual((cal["image_min"],cal["image_max"]),(5,6))
        self.assertEqual((cal["pixel_min"],cal["pixel_max"]),(1,6))
        self.assertEqual(cal["image_threshold"],6.)
        self.assertEqual(cal["pixel_threshold"],6.)
        # Test calibration stays frozen; map and image score are normalized independently.
        test=np.array([[[6.2,6],[4,1]],[[7,3],[2,0]]],np.float32)
        scores,pixels,labels,masks=post_process(test,cal)
        np.testing.assert_allclose(scores,[.7,1.],atol=1e-6)
        np.testing.assert_allclose(pixels[0],[[.54,.5],[.1,0]],atol=1e-6)
        self.assertTrue(labels[0]); self.assertTrue(masks[0,0,0]); self.assertTrue(masks[0,0,1])
        self.assertNotEqual(float(scores[0]),float(pixels[0].max()))
        np.testing.assert_array_equal(normalize_scores([0,2,4],2,2,2),[0,.5,1])
        # The raw equality included by the F1 curve is also included in inference.
        np.testing.assert_array_equal(post_process(maps,cal)[2],[True,False])
        self.assertEqual(metrics(rows,test,.5,"test",cal)["tp"],1)

    def test_eval_rank_metrics_ignore_display_clipping(self):
        y=np.array([[1,0]],np.float32)
        rows=[dict(label=1,y=y,equipment="E"),dict(label=0,y=y*0,equipment="E")]
        val=np.array([[[1.,0.]],[[0.5,0.]]],np.float32)
        calibration=calibrate_post_processor(rows,val,raw_image_scores=[1.,0.])
        test=np.array([[[3.,2.]],[[2.,2.]]],np.float32)
        image_scores=[3.,2.]
        display, pixels, _, _=post_process(test,calibration,raw_image_scores=image_scores)
        np.testing.assert_array_equal(display, [1.,1.])
        np.testing.assert_array_equal(pixels,np.ones_like(test))
        for split in ("val","test"):
            result=metrics(rows,test,.5,split,calibration,raw_image_scores=image_scores)
            self.assertEqual(result["image_auroc"],1.)
            self.assertEqual(result["pixel_auroc"],1.)
            if split == "test":
                self.assertEqual(result["pixel_aupro"],1.)

    def test_validation_constraints_control_selection(self):
        args=SimpleNamespace(padim_suite=True,region_recall=.9)
        goals=dict(image_recall=.95,region_recall=.9,pixel_fpr_cap=.05)
        def candidate(name, recall, fpr, pixel_fpr=.04, small=.9):
            val=dict(recall=recall,fpr=fpr,worst_equipment_fpr=fpr,dice=.9,pixel_fpr=pixel_fpr,
                     regions={k:dict(recall=small) for k in ("1px","2-4px")})
            return dict(experiment=name,status="ok",val=val,validation_constraints=validation_constraints(val,goals))
        ineligible=[candidate("low recall",.70,.01),candidate("too much pixel FP",1.,.02,.08),
                    candidate("missing GT",1.,.03,small=None)]
        self.assertIsNone(select_experiment(ineligible,args))
        self.assertEqual(select_experiment([*ineligible,candidate("eligible",.95,.05)],args),"eligible")

    def test_jet_uses_opencv_lut_without_per_image_normalization(self):
        import cv2
        values=np.array([[0,.25,.5,.75,1]],np.float32)
        expected=cv2.applyColorMap((values*255).astype(np.uint8),cv2.COLORMAP_JET)[:,:,::-1]
        np.testing.assert_array_equal(jet_heatmap(values),expected)

    def test_image_and_pixel_thresholds_can_differ(self):
        y=np.array([[1,0],[0,0]],np.float32)
        rows=[dict(label=1,y=y,equipment="a"),dict(label=0,y=y*0,equipment="b")]
        maps=np.array([[[6,9],[4,2]],[[5,1],[1,1]]],np.float32)
        cal=calibrate_post_processor(rows,maps)
        self.assertEqual(cal["image_threshold"],9.)
        self.assertEqual(cal["pixel_threshold"],6.)
        scores,pixels,labels,masks=post_process(np.array([[[7,1],[1,1]]],np.float32),cal)
        self.assertFalse(labels[0])
        self.assertTrue(masks[0,0,0])

    def test_padim_explicit_image_score_reaches_metrics_csv_and_evidence(self):
        y=np.array([[1,0],[0,0]],np.float32)
        rows=[dict(id="2",image="void.png",label=1,y=y,x=y,equipment="A",lot="L1",wafer="01"),
              dict(id="3",image="normal.png",label=0,y=y*0,x=y*0,equipment="A",lot="L2",wafer="02")]
        maps=np.array([[[1,.1],[.1,.1]],[[3,.1],[.1,.1]]],np.float32)
        cal=calibrate_post_processor(rows,maps,raw_image_scores=[4.,3.])
        self.assertEqual(metrics(rows,maps,.5,"val",cal,raw_image_scores=[4.,3.])["image_auroc"],1.)
        test=metrics(rows,maps,.5,"test",cal,raw_image_scores=[5.,3.])
        self.assertEqual((test["tp"],test["fp"],test["image_f1"]),(1,0,1.))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            save_eval(rows,maps,cal,root,"test",panel_scale=1,raw_image_scores=[5.,3.],
                      collect_comparison=True)
            with (root/"test"/"scores.csv").open(newline="",encoding="utf-8") as f:
                rec=list(csv.DictReader(f))
            self.assertEqual(float(rec[0]["raw_score"]),5.)
            self.assertEqual(float(rec[0]["native_map_max"]),1.)
            self.assertEqual(rec[0]["prediction"],"1")
            self.assertEqual(rec[1]["prediction"],"0")

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
