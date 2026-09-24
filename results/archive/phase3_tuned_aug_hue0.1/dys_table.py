import pickle, glob, sys, collections, os
d=sys.argv[1]; D='Dysfunctional'
print(f"{'run':38s} {'acc':>6s} {'D_rec':>6s} {'D_pred':>6s} {'D_fp':>5s}")
for f in sorted(glob.glob(os.path.join(d,'results_clarifier_*.pkl'))):
    r=pickle.load(open(f,'rb'))
    ps=r['per_sample']; t=[s['true'] for s in ps]; p=[s['pred'] for s in ps]
    acc=sum(a==b for a,b in zip(t,p))/len(t); nd=t.count(D)
    rec=sum(a==D==b for a,b in zip(t,p))/nd if nd else float('nan')
    fp=sum(a!=D and b==D for a,b in zip(t,p))
    print(f"{os.path.basename(f)[18:-4]:38s} {acc:6.3f} {rec:6.2f} {p.count(D):6d} {fp:5d}")
