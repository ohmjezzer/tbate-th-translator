import pytesseract, json, collections, re
from PIL import Image

def overlap(a,b):
    ax0,ay0,ax1,ay1=a; bx0,by0,bx1,by1=b
    ix=max(0,min(ax1,bx1)-max(ax0,bx0)); iy=max(0,min(ay1,by1)-max(ay0,by0))
    inter=ix*iy
    if inter==0: return 0
    return inter/ min((ax1-ax0)*(ay1-ay0)+1,(bx1-bx0)*(by1-by0)+1)

def blocks_psm(im, psm):
    df=pytesseract.image_to_data(im, lang='eng', config=f'--psm {psm}', output_type=pytesseract.Output.DATAFRAME)
    df=df.dropna(subset=['text']); df['text']=df['text'].astype(str)
    df=df[df['text'].str.strip()!='']
    words=[]
    for _,r in df.iterrows():
        t=r['text'].strip()
        conf=float(r['conf'])
        if conf < 25: continue
        if len(t)<2 and not t.isalpha(): continue
        words.append({'b':int(r['block_num']),'p':int(r['par_num']),'l':int(r['line_num']),
                      'x':int(r['left']),'y':int(r['top']),'w':int(r['width']),'h':int(r['height']),'t':t,'c':conf})
    # group by (b,)
    groups=collections.OrderedDict()
    for w in words:
        groups.setdefault(w['b'],[]).append(w)
    out=[]
    for b,ws in groups.items():
        ws.sort(key=lambda z:(z['y'],z['x']))
        x0=min(w['x'] for w in ws); y0=min(w['y'] for w in ws)
        x1=max(w['x']+w['w'] for w in ws); y1=max(w['y']+w['h'] for w in ws)
        txt=' '.join(w['t'] for w in ws)
        out.append({'box':[x0,y0,x1,y1],'text':txt})
    return out

def clean(t):
    core=re.sub(r'[^A-Za-z]','',t)
    if len(core)<3: return False
    if len(core)/max(1,len(t)) < 0.5: return False
    return True

merged={}
for i in range(1,42):
    im=Image.open(f'orig/page-{i:04d}.webp').convert('RGB')
    b3=[x for x in blocks_psm(im,3) if clean(x['text'])]
    b11=[x for x in blocks_psm(im,11) if clean(x['text'])]
    final=list(b3)
    for x in b11:
        if any(overlap(x['box'],y['box'])>0.35 for y in final): continue
        final.append(x)
    final.sort(key=lambda z:(z['box'][1],z['box'][0]))
    merged[i]=final
json.dump(merged, open('blocks2.json','w'), ensure_ascii=False, indent=1)
print('total', sum(len(v) for v in merged.values()))
for p in sorted(merged,key=lambda x:int(x)):
    print(f'=== P{p} ({len(merged[p])}) ===')
    for b in merged[p]:
        print('  ',b['box'],b['text'])
