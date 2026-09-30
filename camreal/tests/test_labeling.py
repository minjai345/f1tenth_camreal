import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
import yaml
from camreal.labeling.core import Project, ConflictError, ring_point, export_dataset, LABEL_HEADER
from conftest import make_model_dir

BEV={'x_range_m':[.2,4.],'y_range_m':[-1.5,1.5],'resolution_m':.05}


def test_click_is_snapped_onto_the_ring_around_the_rear_axle():
    np.testing.assert_allclose(ring_point([2.,0.],1.,BEV),[1.,0.])
    np.testing.assert_allclose(ring_point([.5,.5],1.,BEV),[np.sqrt(.5)]*2)
    np.testing.assert_allclose(ring_point([.9,-.2],1.5,BEV),np.array([.9,-.2])*1.5/np.hypot(.9,-.2))


@pytest.mark.parametrize('point',[[0.,0.],[float('nan'),0.],[-1.,0.],[.1,1.],[1.],[1.,2.,3.]])
def test_invalid_or_out_of_bev_clicks_rejected(point):
    with pytest.raises(ValueError):ring_point(point,1.,BEV)


def test_label_header_is_camsims():
    from camsim.dataset import LABEL_HEADER as camsim_header
    assert LABEL_HEADER==camsim_header


@pytest.fixture
def bag_projects(tmp_path):
    rosbag=pytest.importorskip('rosbag2_py')
    from rclpy.serialization import serialize_message
    from cv_bridge import CvBridge
    from camreal.labeling.bag import extract_bag
    from camreal.tools.make_assumed_calibration import make_assumed
    make_model_dir(tmp_path)
    (tmp_path/'camera.yaml').write_text(yaml.safe_dump(make_assumed(640,400)))
    projects=[]
    for session in ('session_a','session_b'):
        bag=tmp_path/(session+'_bag')
        writer=rosbag.SequentialWriter()
        writer.open(rosbag.StorageOptions(uri=str(bag),storage_id='sqlite3'),rosbag.ConverterOptions('',''))
        writer.create_topic(rosbag.TopicMetadata(name='/camera/image',type='sensor_msgs/msg/Image',serialization_format='cdr'))
        bridge=CvBridge()
        for i in range(3):
            raw=np.full((400,640,3),100+i+(20 if session.endswith('b') else 0),np.uint8)
            msg=bridge.cv2_to_imgmsg(raw,encoding='bgr8');msg.header.frame_id='camera_optical'
            msg.header.stamp.sec=10;msg.header.stamp.nanosec=i*100000000
            writer.write('/camera/image',serialize_message(msg),10000000000+i*100000000+1000)
        del writer
        project=tmp_path/session
        extract_bag(bag,'/camera/image',tmp_path/'model',tmp_path/'camera.yaml',project,session,interval_s=.15)
        projects.append(project)
    return projects


def course_config(projects):
    root=projects[0].parent
    c=yaml.safe_load(Path('camreal/config/course.yaml').read_text())
    c.update(model=str(root/'model'),calibration=str(root/'camera.yaml'),image_topic='/camera/image')
    c['paths']={k:str(v) for k,v in dict(bags=root,projects=root,datasets=root/'datasets').items()}
    c['sessions']={'train':['session_a'],'val':['session_b']}
    c['sampling']={'interval_s':.15,'max_frames':100}
    path=root/'course.yaml'
    path.write_text(yaml.safe_dump(c))
    return path


def payload(revision=0,status='accepted',waypoint=(1.,.3)):
    return dict(revision=revision,status=status,waypoint_m=list(waypoint),note='')


def test_real_rosbag_extraction_timestamps_and_sampling(bag_projects):
    p=Project(bag_projects[0])
    assert len(p.frames)==2 and p.ahead_m==1.
    f=p.frames['000001']
    assert f['header_stamp_ns']=='10200000000' and f['bag_stamp_ns']=='10200001000'
    assert f['source_sequence']==3 and f['camera_frame']=='camera_optical'
    assert (p.root/'raw/000001.png').is_file() and (p.root/'model/checkpoint.json').is_file()
    assert p.meta['calibration_status']=='assumed' and p.meta['model']['git_commit']=='test-commit'


def test_annotations_snap_revision_history_and_reopen(bag_projects):
    p=Project(bag_projects[0])
    first=p.save('000000',payload(waypoint=(2.,.6)))
    assert first['revision']==1
    np.testing.assert_allclose(first['waypoint_m'],np.array([2.,.6])/np.hypot(2.,.6))
    with pytest.raises(ConflictError):p.save('000000',payload())
    with pytest.raises(ValueError,match='원 위에'):p.save('000000',dict(payload(1),waypoint_m=None))
    p.save('000000',payload(1,'rejected'))
    assert Project(p.root).annotation('000000')['status']=='rejected'
    assert len(list((p.root/'annotation_history/000000').glob('*.json')))==2
    assert p.counts()=={'accepted':0,'rejected':1,'unlabeled':1}


def test_export_is_a_camsim_dataset_that_trains(bag_projects,tmp_path):
    import torch
    from camsim import train
    from camsim.dataset import DiskDataset
    from camreal.evaluate import evaluate
    for path in bag_projects:Project(path).save('000000',payload())
    Project(bag_projects[0]).save('000001',payload(status='rejected'))   # never a training target
    out=tmp_path/'export'
    result=export_dataset([bag_projects[0]],[bag_projects[1]],out)
    assert result['counts']=={'train':1,'val':1}
    cfg=Project(bag_projects[0]).cfg
    ds=DiskDataset(str(out/'train'),cfg,'all')
    assert len(ds)==1 and np.isnan(ds.poses).all() and ds.files==['session_a_000000.png']
    np.testing.assert_allclose(ds.wps[0],np.array([1.,.3])/np.hypot(1.,.3),atol=1e-4)
    # Students train on it with camsim as-is.
    net,_=train.train(None,cfg,steps=2,batch_size=1,dataset=ds,val_dataset=DiskDataset(str(out/'val'),cfg,'all'),log_every=1)
    assert isinstance(net,torch.nn.Module)
    metrics=evaluate(tmp_path/'model',out,tmp_path/'eval')
    assert set(metrics['splits']['val']['sessions'])=={'session_b'}
    assert (tmp_path/'eval/index.html').is_file() and (tmp_path/'eval/000.png').is_file()


def test_same_bag_cannot_cross_splits(bag_projects,tmp_path):
    original=bag_projects[0]
    copied=tmp_path/'renamed_project';shutil.copytree(original,copied)
    meta=json.loads((copied/'project.json').read_text());meta['session_id']='renamed'
    (copied/'project.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='rosbag'):export_dataset([original],[copied],tmp_path/'bad')


def test_export_detects_image_changes(bag_projects,tmp_path):
    for path in bag_projects:Project(path).save('000000',payload())
    with (bag_projects[0]/'bev/000000.png').open('ab') as stream:stream.write(b'x')
    with pytest.raises(ValueError,match='BEV image changed'):
        export_dataset([bag_projects[0]],[bag_projects[1]],tmp_path/'bad')


def test_export_command_writes_dataset_zip_and_baseline(bag_projects,tmp_path):
    for path in bag_projects:Project(path).save('000000',payload())
    course=course_config(bag_projects)
    command=[sys.executable,'-m','camreal','export','week3_real','--config',str(course)]
    done=subprocess.run(command,check=True,timeout=120,capture_output=True,text=True)
    assert 'train 1장 · val 1장' in done.stdout and 'session_b' in done.stdout
    assert (tmp_path/'datasets/week3_real.zip').is_file()
    assert (tmp_path/'datasets/week3_real/baseline/index.html').is_file()
    again=subprocess.run(command,timeout=60,capture_output=True,text=True)
    assert again.returncode==2 and '이미 있습니다' in again.stderr


def test_local_editor_api_save_reload_and_conflict(bag_projects):
    import select
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    course=course_config(bag_projects)
    proc=subprocess.Popen([sys.executable,'-u','-m','camreal','label','session_a','--config',str(course),'--port','0'],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        ready,_,_=select.select([proc.stdout],[],[],15)
        assert ready,'server did not start'
        line=proc.stdout.readline()
        if not line:raise RuntimeError(proc.stderr.read())
        base=line.split('Label editor: ')[1].split(' |')[0]
        with urlopen(base+'/api/project',timeout=5) as response:info=json.load(response)
        assert info['ahead_m']==1. and info['calibration_status']=='assumed'
        with urlopen(base+'/',timeout=5) as response:assert 'BEV'.encode() in response.read()
        body=json.dumps(payload()).encode()
        headers={'Content-Type':'application/json','X-Label-Token':info['token']}
        request=Request(base+'/api/annotation/000000',data=body,headers=headers)
        with urlopen(request,timeout=5) as response:assert json.load(response)['status']=='accepted'
        with pytest.raises(HTTPError) as error:urlopen(request,timeout=5)
        assert error.value.code==409
        with urlopen(base+'/api/frame/000000',timeout=5) as response:assert json.load(response)['revision']==1
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base+'/api/annotation/000001',data=body,headers={'Content-Type':'application/json'}),timeout=5)
        assert error.value.code==403
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_prepare_command_reads_real_bag_and_reuses_it(bag_projects,tmp_path):
    course=course_config(bag_projects)
    c=yaml.safe_load(course.read_text())
    c['paths']['projects']=str(tmp_path/'prepared')
    course.write_text(yaml.safe_dump(c))
    command=[sys.executable,'-m','camreal','prepare','session_a_bag','--config',str(course)]
    first=subprocess.run(command,check=True,timeout=30,capture_output=True,text=True)
    assert '입력:' in first.stdout and '출력:' in first.stdout
    p=Project(tmp_path/'prepared/session_a_bag')
    assert len(p.frames)==2
    annotation=p.root/'annotations/000000.json'
    annotation.write_text('{"keep": true}')
    again=subprocess.run(command,check=True,timeout=30,capture_output=True,text=True)
    assert '이미 준비됨' in again.stdout
    assert annotation.read_text()=='{"keep": true}'
    c['sampling']['interval_s']=.7
    course.write_text(yaml.safe_dump(c))
    changed=subprocess.run(command,timeout=30,capture_output=True,text=True)
    assert changed.returncode==2 and '추출 설정' in changed.stderr
    assert annotation.read_text()=='{"keep": true}'
    c['sampling']['interval_s']=p.meta['sampling']['interval_s']
    course.write_text(yaml.safe_dump(c))
    (p.root/'bev/000000.png').unlink()
    broken=subprocess.run(command,timeout=30,capture_output=True,text=True)
    assert broken.returncode==2 and '재사용할 수 없습니다' in broken.stderr
