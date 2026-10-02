import numpy as np
from PIL import Image
import pytest
import torch

from src.explicit_region.geometry import apply_geometry, sample_geometry, SampleRejected
from src.explicit_region.masks import pixel_mask_to_token_mask


def test_shared_geometry_and_nearest_mask():
    image = Image.fromarray(np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3))
    mask_array = np.zeros((12, 20), dtype=np.uint8)
    mask_array[3:9, 7:13] = 255
    mask = Image.fromarray(mask_array)
    geometry = sample_geometry(
        mask, resolution=8, base_seed=1, global_sample_index=2, sample_key="x",
        minimum_mask_retention=0.5, random_flip=True,
    )
    source = apply_geometry(image, geometry, is_mask=False)
    target = apply_geometry(image, geometry, is_mask=False)
    transformed_mask = apply_geometry(mask, geometry, is_mask=True)
    assert source.size == target.size == transformed_mask.size == (8, 8)
    assert set(np.unique(transformed_mask)).issubset({0, 255})
    assert np.array_equal(source, target)


def test_empty_mask_rejected():
    with pytest.raises(SampleRejected, match="post_resize_empty_mask"):
        sample_geometry(
            Image.new("L", (10, 10)), resolution=8, base_seed=0,
            global_sample_index=0, sample_key="empty",
        )


def test_any_overlap_keeps_one_pixel():
    mask = torch.zeros(1, 16, 16)
    mask[0, 15, 15] = 1
    tokens = pixel_mask_to_token_mask(mask, (4, 4), mode="any_overlap")
    assert tokens.sum().item() == 1
    assert tokens[0, -1, -1]


def test_zero_token_fails():
    with pytest.raises(ValueError, match="minimum_edit_tokens"):
        pixel_mask_to_token_mask(torch.zeros(1, 8, 8), (2, 2))


def test_contain_center_pad_fallback_is_exact_and_shared():
    source_array=np.full((4,20,3),10,np.uint8);target_array=np.full((4,20,3),20,np.uint8)
    mask_array=np.zeros((4,20),np.uint8);mask_array[:,0:2]=255;mask_array[:,18:20]=255
    source,target,mask=map(Image.fromarray,(source_array,target_array,mask_array))
    geometry=sample_geometry(mask,resolution=8,base_seed=42,global_sample_index=99,
                             sample_key="wide",minimum_mask_retention=.75,max_resample_attempts=8)
    assert geometry.geometry_mode=="contain_center_pad_fallback"
    assert geometry.mask_retention==1.0 and not geometry.flip
    transformed=[apply_geometry(image,geometry,is_mask=(i==2)) for i,image in enumerate((source,target,mask))]
    assert all(image.size==(8,8) for image in transformed)
    assert np.array_equal(np.asarray(transformed[0])[0],np.full((8,3),127,np.uint8))
    assert set(np.unique(transformed[2])).issubset({0,255})
    assert (np.asarray(transformed[2])>0).any()
