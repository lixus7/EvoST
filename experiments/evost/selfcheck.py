"""Small CPU checks of the released calibration implementation; no traffic training."""
from pathlib import Path
import sys
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'g37'))
import g28_common as g

def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    x=torch.randn(90,3,12,dtype=torch.float64)
    b=torch.randn_like(x)
    y=torch.randn_like(x)+10
    valid=torch.ones(90,3,dtype=torch.bool)
    valid[20:30,1]=False
    batch=g.NodeRLS(3,12,2.,1.,1.,'cpu')
    batch.fit(x,b,y,valid)
    online=g.NodeRLS(3,12,2.,1.,1.,'cpu')
    online.fit(x[:15],b[:15],y[:15],valid[:15])
    for i in range(15,len(x)):
        online.update(x[i],b[i],y[i],valid[i])
    torch.testing.assert_close(online.theta,batch.theta,rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(online.P,batch.P,rtol=1e-9,atol=1e-9)

    def stream(targets):
        model=g.NodeRLS(3,12,2.,1.,1.,'cpu')
        preds=[]
        def save(t,pred,_,count):
            latest=(t//8)*8-12
            expected=valid[:max(0,latest+1)].sum(0).to(torch.float64)
            torch.testing.assert_close(count,expected,rtol=0,atol=0)
            preds.append(pred.clone())
        g.rls_stream(model,len(x),g.array_chunks(x,b,targets,valid),delay=12,cadence=8,on_window=save,chunk=16)
        return torch.stack(preds)
    original=stream(y)
    changed=y.clone()
    changed[30:]+=1000
    perturbed=stream(changed)
    # The first altered label window matures at 42 and is first absorbed at cadence 48.
    torch.testing.assert_close(original[:48],perturbed[:48],rtol=0,atol=0)
    assert not torch.equal(original[48:],perturbed[48:])
    errors=np.arange(1.,13.)[None]
    metric=g.summarize(errors,errors**2,np.ones_like(errors))
    assert metric['H3_MAE']==2. and metric['H12_MAE']==6.5 and metric['Avg_MAE']==3.75
    print('PASS: recursive/batch ridge agreement, invalid-window exclusion, delayed-feedback causality, and cumulative metric definitions (CPU).')

if __name__=='__main__':
    main()
