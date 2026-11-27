"""Score processing tests independent of the Torch backbone/runtime."""
import unittest
import numpy as np

from score_processing import (PADIM_IMAGE_SCORE_SOURCE, NATIVE_IMAGE_SCORE_SOURCE,
                              calibrate_post_processor, get_raw_image_scores, post_process, validation_constraints)


class TestImageScoreProcessing(unittest.TestCase):
    def setUp(self):
        y=np.array([[1,0],[0,0]],np.float32)
        self.rows=[dict(label=1,y=y),dict(label=0,y=y*0)]
        # Native-map peaks rank the normal image higher; model-grid peaks rank correctly.
        self.maps=np.array([[[1,.1],[.1,.1]],[[3,.1],[.1,.1]]],np.float32)
        self.raw=np.array([4,3],np.float32)

    def test_calibration_uses_explicit_model_resolution_scores(self):
        cal=calibrate_post_processor(self.rows,self.maps,raw_image_scores=self.raw)
        self.assertEqual(cal["image_score_source"],PADIM_IMAGE_SCORE_SOURCE)
        self.assertEqual((cal["image_min"],cal["image_max"],cal["image_threshold"]),(3.,4.,4.))
        self.assertEqual(cal["pixel_max"],3.)
        scores,_,labels,_=post_process(self.maps,cal,raw_image_scores=[5.,3.])
        np.testing.assert_array_equal(scores,[1.,0.])
        np.testing.assert_array_equal(labels,[True,False])
        with self.assertRaisesRegex(ValueError,"model-input-resolution pred_score"):
            post_process(self.maps,cal)

    def test_recall_constrained_pixel_threshold_and_conflicting_cap(self):
        gt=np.array([[1,0,1,1,0]], dtype=np.float32)
        rows=[dict(label=1,y=gt),dict(label=0,y=gt*0)]
        maps=np.array([[[4.,3.,2.,1.,0.]], [[2.,0.,0.,0.,0.]]], dtype=np.float32)
        goals=dict(image_recall=.95,region_recall=.9,pixel_fpr_cap=.05)
        c=calibrate_post_processor(rows,maps,raw_image_scores=[8.,7.],goals=goals)
        self.assertEqual((c["image_threshold"],c["pixel_threshold"]),(8.,2.))
        _, _, labels, predicted=post_process(maps,c,raw_image_scores=[8.,7.])
        np.testing.assert_array_equal(labels,[True,False])
        self.assertTrue(predicted[0,0,2])  # The tied pixel is required to hit the 2px region.
        masks=np.stack([r["y"]>0 for r in rows])
        pixel_fpr=float(predicted[~masks].mean())
        checks=validation_constraints(dict(recall=1.,pixel_fpr=pixel_fpr,
                                           regions={"1px":{"recall":1.},"2-4px":{"recall":1.}}),goals)
        self.assertEqual(checks,{"image_recall":True,"1px":True,"2-4px":True,"pixel_fpr":False})

    def test_recall_constraint_can_disable_region_goals_and_respect_pixel_cap(self):
        goals=dict(image_recall=.95,region_recall=0,pixel_fpr_cap=0.)
        cal=calibrate_post_processor(self.rows,self.maps,raw_image_scores=self.raw,goals=goals)
        _, _, _, prediction=post_process(self.maps,cal,raw_image_scores=self.raw)
        self.assertFalse(prediction.any())
        self.assertGreater(cal["pixel_threshold"],3.)

    def test_native_fallback_only_for_non_padim_calibration(self):
        cal=calibrate_post_processor(self.rows,self.maps)
        self.assertEqual(cal["image_score_source"],NATIVE_IMAGE_SCORE_SOURCE)
        self.assertEqual(cal["image_max"],3.)
        np.testing.assert_array_equal(get_raw_image_scores(self.maps),[1.,3.])
        with self.assertRaises(ValueError):
            get_raw_image_scores(self.maps,[1.])
        with self.assertRaises(ValueError):
            get_raw_image_scores(self.maps,[np.nan,1.])


if __name__=="__main__":
    unittest.main()
