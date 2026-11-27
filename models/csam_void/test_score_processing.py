"""Score processing tests independent of the Torch backbone/runtime."""
import unittest
import numpy as np

from score_processing import (PADIM_IMAGE_SCORE_SOURCE, NATIVE_IMAGE_SCORE_SOURCE,
                              calibrate_post_processor, get_raw_image_scores, post_process)


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
