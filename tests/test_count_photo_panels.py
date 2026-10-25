"""Synthetic geometry checks for private photo panel preparation/stitching."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from count_photo_panels import (assemble_panel, band_geometry, dimension_plan,
                                effective_dimensions, photo_count_summary,
                                photo_tile_windows, stitch_photo_cores)


class PhotoPanelGeometryTests(unittest.TestCase):
    def test_crop_at_right_bottom_without_inventing_padding(self):
        plan=dimension_plan(5159,3959)
        self.assertEqual(plan["crop_dimensions"],[5156,3956])
        self.assertEqual(plan["cropped_right_bottom_px"],[3,3])
        self.assertEqual(plan["padded_right_bottom_px"],[0,0])
        self.assertEqual(plan["effective_dimensions"],[5156,3956])
        self.assertEqual(effective_dimensions(5159,3959),(5156,3956))

    def test_panorama_bands_cover_once_with_aligned_starts_and_final_four_pixel_edge(self):
        width,height=5200,12479
        bands=band_geometry(width,height)
        self.assertEqual(len(bands),7)
        self.assertEqual(bands[-1]["core_xyxy"][3],12476)
        coverage=np.zeros(12476,dtype=np.uint8)
        for band in bands:
            x0,y0,x1,y1=band["core_xyxy"]
            cx0,cy0,cx1,cy1=band["context_xyxy"]
            self.assertEqual((x0,x1),(0,width))
            self.assertEqual((y0%8,cy0%8),(0,0))
            self.assertEqual((y1%4,cy1%4),(0,0))
            if y1 < 12476: self.assertEqual((y1%8,cy1%8),(0,0))
            self.assertLessEqual((cx1-cx0)*(cy1-cy0),24_000_000)
            self.assertLessEqual(y0-cy0,96)
            self.assertLessEqual(cy1-y1,96)
            coverage[y0:y1]+=1
        self.assertTrue(np.all(coverage==1))

    def test_tile_context_stitch_preserves_global_coordinate_map_once(self):
        width,height=400,240
        yy,xx=np.indices((height,width))
        rgb=np.zeros((height,width,3),dtype=np.uint8)
        rgb[:,:,0]=xx.astype(np.uint8); rgb[:,:,1]=yy.astype(np.uint8)
        windows=list(photo_tile_windows(width,height))
        self.assertGreater(len(windows),1)
        for _,(x0,y0,x1,y1) in windows:
            self.assertLessEqual(x1-x0,512); self.assertLessEqual(y1-y0,320)
        def predict(context):
            return context[::2,::2,0].astype(np.float32)+256*context[::2,::2,1].astype(np.float32)
        stitched,count=assemble_panel(rgb,predict)
        expected=rgb[::2,::2,0].astype(np.float32)+256*rgb[::2,::2,1].astype(np.float32)
        self.assertEqual(count,len(windows))
        self.assertTrue(np.array_equal(stitched,expected))

    def test_global_quadrants_clamp_after_all_core_bands_are_stitched(self):
        top_negative=np.full((2,4),-1,dtype=np.float32)
        top_positive=np.full((2,4),2,dtype=np.float32)
        bottom_negative=np.full((4,4),-1,dtype=np.float32)
        full=stitch_photo_cores(8,16,[([0,0,8,4],top_negative),
            ([0,4,8,8],top_positive),([0,8,8,16],bottom_negative)])
        summary=photo_count_summary(full)
        self.assertEqual(summary["raw_total"],-8.0)
        self.assertEqual(summary["raw_quadrant_sums"],[4.0,4.0,-8.0,-8.0])
        self.assertAlmostEqual(summary["count"],0.08)
        with self.assertRaisesRegex(ValueError,"overlap"):
            stitch_photo_cores(8,16,[([0,0,8,8],np.ones((4,4),np.float32)),
                ([0,4,8,16],np.ones((6,4),np.float32))])


if __name__=="__main__":
    unittest.main()
