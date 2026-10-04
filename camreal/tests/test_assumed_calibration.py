import numpy as np
import pytest
import yaml
from camsim import camera, config, render
from camreal.preprocessing import CameraPreprocessor
from camreal.tools.make_assumed_calibration import make_assumed


@pytest.mark.parametrize('width,height',[(640,400),(1920,1200)])
def test_assumed_projection_and_preview_only(tmp_path,width,height):
    c=make_assumed(width,height)
    path=tmp_path/'assumed.yaml';path.write_text(yaml.safe_dump(c))
    cfg=config.load();mask=render.bev_visibility_mask(camera.build(cfg)[0],cfg)
    pre=CameraPreprocessor(path,cfg,mask,'rear_axle')
    # At 1 metre forward, camera height 0.2m gives v=cy+fy*0.2.
    uv=np.array([[width/2,height/2+width/2*.2]])
    np.testing.assert_allclose(camera.project(pre.H,uv),[[1.,0.]],atol=1e-12)
    assert pre.bev(np.zeros((height,width,3),np.uint8)).shape==(*render.bev_size(cfg),3)
    with pytest.raises(ValueError,match='미리보기 전용'):pre.require_driving_calibration()


def test_assumed_invalid_parameters():
    for params in [dict(width=0),dict(hfov_deg=180),dict(height_m=0),dict(pitch_deg=float('nan'))]:
        with pytest.raises(ValueError):make_assumed(**params)
