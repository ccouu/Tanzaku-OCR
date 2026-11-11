"""Reattach to a running Stage 2 worker through the CLI's Contents API session."""
import json
from pathlib import Path
import shutil
import time
from .colab_pipeline import transfer_cli, file_sha, call, SessionUnavailable
from src.common import dump_json
from src.common.paths import TRAINING_OUTPUT, ROOT, resolve_path


def monitor(session='japanese-menu-yolo-stage2',output=TRAINING_OUTPUT,execution=None):
    output=resolve_path(output);output.mkdir(parents=True,exist_ok=True)
    remote='/content/menu_obb_training'
    recovered=0;failures=0
    while True:
        try:
            transfer_cli('status',output/'remote_status.json',remote,session)
            status=json.loads((output/'remote_status.json').read_text())
        except SessionUnavailable:
            dump_json(output/'monitor_error.json',dict(state='session_lost',session=session,recovered_epoch=recovered))
            raise
        except RuntimeError:
            failures+=1
            if failures>=5:
                dump_json(output/'monitor_error.json',dict(state='transport_failed',recovered_epoch=recovered))
                raise
            print('Status transport temporarily unavailable; retrying. Training worker is independent.',flush=True)
            time.sleep(20);continue
        failures=0
        recovery=status.get('recovery') or {}
        if recovery.get('epoch',0)>recovered:
            directory=output/'recovery';directory.mkdir(exist_ok=True)
            for kind,info in recovery.get('files',{}).items():
                target=directory/f'latest_{kind}.pt';incoming=directory/f'incoming_{kind}.pt'
                if not info['path'].startswith(remote+'/recovery/'):
                    raise ValueError('Unexpected recovery checkpoint path')
                transfer_cli('download',incoming,info['path'],session)
                if file_sha(incoming)!=info['sha256']:raise ValueError('Recovery checkpoint checksum mismatch')
                incoming.replace(target)
            dump_json(directory/'checkpoint_info.json',recovery)
            recovered=recovery['epoch']
            print('Saved and verified local recovery checkpoints at epoch',recovered,flush=True)
        if execution is not None and execution.poll() not in [None,0] and status['state'] not in ['completed','failed']:
            dump_json(output/'execution_warning.json',dict(returncode=execution.returncode,note='Kernel execution client exited; checking worker status through API.'))
        if status['state']=='completed':break
        if status['state']=='failed':raise RuntimeError(status.get('error',status.get('log_tail','Training failed')))
        time.sleep(25)
    for filename in ['yolo11n_menu_obb_best.pt','training_results.zip','evaluation_report.json','training.log','run_config.json']:
        transfer_cli('download',output/filename,remote+'/'+filename,session)
    report=json.loads((output/'evaluation_report.json').read_text())
    if file_sha(output/'yolo11n_menu_obb_best.pt')!=report['checkpoint_sha256']:
        raise RuntimeError('Downloaded checkpoint hash mismatch')
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    shutil.copy2(output/'yolo11n_menu_obb_best.pt',ROOT/'checkpoints/yolo11n_menu_obb_best.pt')
    try:
        print(call('log','-s',session,'-n','8','-o',output/'colab_execution.ipynb'),flush=True)
    finally:
        print(call('stop','-s',session),flush=True)
    print('Training and evaluation completed; checkpoint verified and GPU released.',flush=True)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',default='japanese-menu-yolo-stage2')
    parser.add_argument('--output',type=Path,default=TRAINING_OUTPUT)
    args=parser.parse_args();monitor(args.session,args.output)
