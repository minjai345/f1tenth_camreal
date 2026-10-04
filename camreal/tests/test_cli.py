import ast
from pathlib import Path
import subprocess
import sys
import pytest
import yaml
from camreal.__main__ import load_course, main, session_name


def test_help_does_not_import_ros_or_torch():
    subprocess.run([sys.executable,'-c',
        'import sys; from camreal.__main__ import main\n'
        'for argv in (["--help"],["calibrate","--help"]):\n'
        '    try: main(argv)\n    except SystemExit as e: assert e.code == 0\n'
        'assert "torch" not in sys.modules; assert "rclpy" not in sys.modules'],check=True)


def test_ros_package_installs_every_camreal_package():
    # colcon installs camreal from camsim_driver/setup.py: python3 -m camreal has to work there outside the repo too.
    setup=next(n for n in ast.walk(ast.parse(Path('camreal/ros2/camsim_driver/setup.py').read_text()))
               if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='setup')
    args={k.arg:ast.literal_eval(k.value) for k in setup.keywords if k.arg in ('packages','package_data')}
    for init in Path('camreal').rglob('__init__.py'):
        if 'tests' in init.parts or 'ros2' in init.parts:continue
        package='.'.join(init.parent.parts)
        assert package in args['packages'],package
        if (init.parent/'web').is_dir():assert {'web/*.html','web/*.js'}<=set(args['package_data'].get(package,[])),package


@pytest.mark.parametrize('argv',[[],['calib','--image','frame.png']])
def test_calibrate_takes_a_bag_session_or_an_image(argv,capsys):
    with pytest.raises(SystemExit) as error:main(['calibrate',*argv,'--config','missing.yaml'])
    assert error.value.code==2 and '하나만' in capsys.readouterr().err


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
