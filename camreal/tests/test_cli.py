from pathlib import Path
import subprocess
import sys
import pytest
import yaml
from camreal.__main__ import load_course, session_name


def test_help_does_not_import_ros_or_torch():
    subprocess.run([sys.executable,'-c',
        'import sys; from camreal.__main__ import main; '
        '\ntry: main(["--help"])\nexcept SystemExit as e: assert e.code == 0\n'
        'assert "torch" not in sys.modules; assert "rclpy" not in sys.modules'],check=True)


@pytest.mark.parametrize('value',['../escape','','a/b','a b'])
def test_session_names_are_paths_only_under_configured_roots(value):
    with pytest.raises(ValueError):session_name(value)


def test_config_rejects_split_overlap_typos_and_old_format(tmp_path):
    c=yaml.safe_load(Path('camreal/config/course.yaml').read_text())
    c['sessions']['val']=c['sessions']['train']
    path=tmp_path/'course.yaml';path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='같은 세션'):load_course(path)
    c['sessions']['val']=['other'];c['sampling']['interval']=5
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='sampling'):load_course(path)
    del c['sampling']['interval'];c['bundle']=c.pop('model')
    path.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='이전 버전'):load_course(path)
