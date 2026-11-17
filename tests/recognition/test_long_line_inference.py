from PIL import Image
from src.recognition.long_line_inference import bucket_image,window_ranges,owned_tokens


def test_real_crop_preserves_height_in_768_bucket():
    im,meta=bucket_image(Image.new('RGB',(374,34)),768)
    assert im.size==(768,64) and meta['valid_width']==704 and meta['scaled_height']==64
    _,small=bucket_image(Image.new('RGB',(374,34)),512)
    assert small['scaled_height']==47


def test_balanced_windows_cover_full_line():
    for width in (1,374,512,704,768,1024,1600):
        ranges=window_ranges(width)
        assert ranges[0][0]==0 and ranges[-1][1]==width
        assert all(0<r-l<=512 for l,r in ranges)
        assert all(b[0]<=a[1] for a,b in zip(ranges,ranges[1:]))


def test_seam_ownership_preserves_legitimate_repeats():
    # Overlap x=16..32, ownership boundary x=24. Two distinct '1' runs survive.
    left=[0,1,0,1,0,0,0,0];right=[0,0,1,0,0,0,0,0]
    assert owned_tokens([left,right],[(0,32),(16,48)])==[1,1,1]


def test_same_overlap_run_has_one_owner():
    left=[0,0,0,0,0,1,0,0];right=[0,1,0,0,0,0,0,0]
    # Both runs have source center22; left owns it because the seam is24.
    assert owned_tokens([left,right],[(0,32),(16,48)])==[1]
