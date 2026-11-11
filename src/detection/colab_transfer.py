"""Workspace-local chunked Contents API adapter using the installed CLI session."""
import argparse
import base64
import json
from pathlib import Path
import time
from urllib.parse import quote


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['upload','download','status'])
    parser.add_argument('local');parser.add_argument('remote');parser.add_argument('--session',required=True)
    args=parser.parse_args()
    import requests
    from colab_cli.common import state
    name=state.resolve_session(args.session)
    if not name:raise SystemExit(3)
    session=state.get_session(name)
    client=requests.Session()
    client.headers.update({'X-Colab-Client-Agent':'colab-cli','X-Colab-Runtime-Proxy-Token':session.token})
    url=session.url.rstrip('/')+'/api/contents/'+quote(args.remote.strip('/'),safe='/')
    params={'authuser':'0','colab-runtime-proxy-token':session.token}
    def request(method,remote_override=None,**kwargs):
        for attempt in range(4):
            try:
                target=url if remote_override is None else session.url.rstrip('/')+'/api/contents/'+quote(remote_override.strip('/'),safe='/')
                response=client.request(method,target,params=params,timeout=(20,90),**kwargs)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                # Exceptions can contain proxy credentials in URLs. Never surface them.
                if attempt==3:raise RuntimeError('Colab file transport failed after four attempts: '+type(exc).__name__) from None
                time.sleep(2+attempt*2)
    local=Path(args.local)
    if args.mode=='upload':
        size=local.stat().st_size;offset=0;chunk=1
        with local.open('rb') as source:
            while True:
                block=source.read(4*1024*1024)
                if not block:break
                final=offset+len(block)==size
                request('PUT',json=dict(name=Path(args.remote).name,path=args.remote,type='file',format='base64',
                        content=base64.b64encode(block).decode('ascii'),chunk=1 if offset==0 else (-1 if final else chunk)))
                offset+=len(block);chunk+=1
                print(f'TRANSFER {offset}/{size}',flush=True)
    elif args.mode=='download':
        result=request('GET')
        temporary=local.with_suffix(local.suffix+'.tmp')
        if result.get('format')=='base64':temporary.write_bytes(base64.b64decode(result['content']))
        else:temporary.write_text(result['content'],encoding='utf-8')
        temporary.replace(local)
        print('Downloaded',local.name,local.stat().st_size,'bytes',flush=True)
    else:
        def text(path):
            result=request('GET',path)
            return base64.b64decode(result['content']).decode() if result.get('format')=='base64' else result['content']
        doc=json.loads(text(args.remote+'/status.json'))
        if doc['state'] in ['evaluating','completed'] or (doc['state']=='training' and doc.get('epoch')):
            lines=text(args.remote+'/runs/menu_obb/results.csv').splitlines()
            if len(lines)>1:
                doc['latest_epoch_csv']=lines[-1];doc['csv_header']=lines[0]
        if doc['state']=='failed':doc['log_tail']=text(args.remote+'/training.log')[-4000:]
        local.write_text(json.dumps(doc,indent=2),encoding='utf-8')
        print('State:',doc['state'],'Epoch:',doc.get('latest_epoch_csv','').split(',')[0],flush=True)


if __name__=='__main__':
    main()
