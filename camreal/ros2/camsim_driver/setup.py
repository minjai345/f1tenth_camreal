from glob import glob
from setuptools import setup

setup(
    name='camsim_driver', version='0.1.0',
    packages=['camsim_driver', 'camsim', 'camreal', 'camreal.labeling', 'camreal.tools'],
    package_dir={'camsim': '../../../camsim', 'camreal': '../..'},
    package_data={'camsim': ['config.yaml'], 'camreal': ['config/*.yaml'],
                  'camreal.labeling': ['web/*.html', 'web/*.js']},
    data_files=[('share/ament_index/resource_index/packages', ['resource/camsim_driver']),
                ('share/camsim_driver', ['package.xml', 'README.md']),
                ('share/camsim_driver/config', glob('config/*.yaml')),
                ('share/camsim_driver/launch', glob('launch/*.launch.py'))],
    install_requires=['setuptools', 'numpy', 'PyYAML'],
    zip_safe=False, maintainer='camsim maintainers', maintainer_email='maintainers@example.com',
    description='Camera BEV waypoint inference and pure pursuit in one ROS 2 node',
    license='MIT', entry_points={'console_scripts': ['camsim_driver_node = camsim_driver.node:main']},
)
