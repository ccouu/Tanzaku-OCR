"""Upload reviewed snapshot, train through colab-cli, retrieve results and release GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import re
from src.common.paths import DATASET_V1, TRAINING_OUTPUT, resolve_path


class SessionUnavailable(RuntimeError):
    pass


def transfer_cli(mode, local, remote, session):
    import os
    import shutil
    configured=os.environ.get('COLAB_CLI_PYTHON')
    if configured:
        interpreter=Path(configured)
    elif shutil.which('uv'):
        result=subprocess.run(['uv','tool','dir'],capture_output=True,text=True,check=True)
        base=Path(result.stdout.strip())/'google-colab-cli'
        interpreter=base/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    else:
        raise RuntimeError('Set COLAB_CLI_PYTHON to the installed Colab CLI Python interpreter')
    adapter=Path(__file__).with_name('colab_transfer.py')
    process=subprocess.Popen([str(interpreter),str(adapter),mode,str(local),remote,'--session',session],
                             stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    for line in process.stdout:
        print(redact(line.rstrip()),flush=True)
    code=process.wait()
    if code==3:raise SessionUnavailable('Colab session is no longer available')
    if code:raise RuntimeError('Colab CLI file transfer failed')


def redact(text):
    return re.sub(r'(?i)(token|key|authorization|credential)([=:\s]+)[^\s&\"\']+', r'\1\2[REDACTED]', text)


def call(*args, timeout=600):
    result=subprocess.run(['colab',*map(str,args)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'colab {args[0]} failed:\n{redact(result.stdout)}\n{redact(result.stderr)}')
    return redact(result.stdout)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def pipeline(dataset, output, session, epochs, uploaded=False):
    dataset=resolve_path(dataset);output=resolve_path(output);output.mkdir(parents=True,exist_ok=True)
    archive=dataset/'dataset.zip'
    if not archive.exists():
        archive=dataset.with_suffix('.zip')
    expected=file_sha(archive)
    info=json.loads((dataset/'training_manifest.json').read_text(encoding='utf-8'))
    if info.get('reviewed_only') is not True:raise ValueError('Dataset must be reviewed-only')
    worker=Path(__file__).with_name('train_yolo_obb.py')
    remote='/content/menu_obb_training'
    print('Uploading reviewed-only archive...',flush=True)
    if not uploaded:
        transfer_cli('upload',archive,'/content/stage2_training.zip',session)
    transfer_cli('upload',worker,'/content/train_yolo_obb.py',session)
    launch=output/'launch_remote.py'
    launch.write_text(
        'import runpy, sys, os, contextlib\n'
        f'os.makedirs({remote!r},exist_ok=True)\n'
        'os.environ["WANDB_DISABLED"]="true"\n'
        f'sys.argv=["/content/train_yolo_obb.py","--archive","/content/stage2_training.zip","--sha256",{expected!r},"--work",{remote!r},"--epochs",{str(epochs)!r}]\n'
        f'with open({(remote+"/training.log")!r},"a",buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):\n'
        ' runpy.run_path("/content/train_yolo_obb.py",run_name="__main__")\n',encoding='utf-8')
    console=(output/'training_console.log').open('w',encoding='utf-8')
    execution=subprocess.Popen(['colab','exec','-s',session,'-f',str(launch),'--timeout','14400'],
                               stdout=console,stderr=subprocess.STDOUT)
    (output/'execution_process.json').write_text(json.dumps(dict(pid=execution.pid,session=session,kernel_training=True)),encoding='utf-8')
    print('Launched foreground kernel training; console:',output/'training_console.log',flush=True)
    from .monitor_colab import monitor
    monitor(session,output,execution=execution)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATASET_V1)
    parser.add_argument('--output',type=Path,default=TRAINING_OUTPUT)
    parser.add_argument('--session',default='japanese-menu-yolo-stage2')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--uploaded',action='store_true',help='Use an already-uploaded archive; the worker still verifies its checksum')
    args=parser.parse_args();pipeline(args.dataset,args.output,args.session,args.epochs,args.uploaded)
