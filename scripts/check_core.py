"""Small mathematical and input-contract checks; no ecological validation data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tree_dynamics.object_metrics import components, selected, match_matrix, score
from tree_dynamics.photometric import transform

def rejects(call):
    try:call()
    except ValueError:return
    raise AssertionError('Invalid input was not rejected')

def main():
    # A maximum-IoU-only assignment would retain one match here. Cardinality comes first.
    pairs=match_matrix(np.array([[.99,.5],[.5,0]]),.5)
    assert {(a,b) for a,b,_ in pairs}=={(0,1),(1,0)}
    assert match_matrix(np.empty((0,2)),.5)==[]
    m=np.zeros((256,256),bool);m[30:33,30:34]=1
    objects,edge=selected(components(m));assert len(objects)==1 and objects[0]['area']==3 and edge==0
    big=np.zeros_like(m);big[40:60,40:80]=1
    assert len(selected(components(big))[0])==0 # upper boundary is exclusive: 200 m2
    diag=np.zeros_like(m);diag[50,50]=1;diag[51,51]=1
    assert len(components(diag))==2 # four-neighbour connectivity
    counts,_=score(objects,objects,.5);assert counts['tp']==1 and counts['fn']==counts['fp']==0
    rejects(lambda:selected([np.zeros((512,512),bool)]))
    rng=np.random.default_rng(310);a=rng.uniform(1,1000,(4,32,32)).astype('float32');a[:,0,:]=0
    np.testing.assert_array_equal(transform(a,'base',1),a)
    np.testing.assert_allclose(transform(a,'gain',1.2),a*1.2)
    for kind,amount in [('base',1),('gain',.8),('nir',1.1),('offset',.05),('gamma',.8)]:
        np.testing.assert_array_equal(transform(np.zeros_like(a),kind,amount),np.zeros_like(a))
        assert not transform(a,kind,amount)[:,0,:].any()
    rejects(lambda:transform(a,'gamam',.8))
    rejects(lambda:transform(a,'gamma',-1))
    rejects(lambda:transform(a[:3],'base',1))
    invalid=a.copy();invalid[0,3,3]=np.nan
    rejects(lambda:transform(invalid,'base',1))
    print('PASS: matching objective, connectivity, area boundaries, mask shape, scenario and NoData handling.')

if __name__=='__main__':main()
