from pathlib import Path
import re
import shutil
import subprocess
import pytest

PAGE = Path(__file__).with_name('calibration_page.js')
SCENARIOS = re.findall(r'^  async (\w+)\(\)\{$', PAGE.read_text(), re.M)


def test_every_scenario_is_found():
    assert len(SCENARIOS) == 9


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
@pytest.mark.parametrize('scenario', SCENARIOS)
def test_calibration_page(scenario):
    run = subprocess.run(['node', str(PAGE), scenario], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr


def test_page_says_to_skip_markers_outside_the_image():
    assert '영상에 안 보이는 마커는 건너뛰기' in (Path(__file__).parents[1]/'calibration'/'web'/'index.html').read_text()
