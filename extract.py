import pytesseract, json, collections
from PIL import Image
out={}
for i in range(1,42):
    f=f'orig/page-{i:04d}.webp'
    im=Image.open(f).convert('RGB')
    df=pytesseract.image_to_data(im, lang='eng', config='--psm 3', output_type=pytesseract.Output.DATAFRAME)
    df=df.dropna(subset=['text'])
    df['text']=df['text'].astype(str)
    df=df[df['text'].str.strip()!='']
    blocks=collections.OrderedDict()
    for _,r in df.iterrows():
        b=int(r['block_num']); p=int(r['par_num']); l=int(r['line_num'])
        key=(b,)
        blocks.setdefault(key,{'x0':1e9,'y0':1e9,'x1':-1,'y1':-1,'lines':collections.OrderedDict()})
        B=blocks[key]
        x0,y0,w,h=int(r['left']),int(r['top']),int(r['width']),int(r['height'])
        B['x0']=min(B['x0'],x0); B['y0']=min(B['y0'],y0); B['x1']=max(B['x1'],x0+w); B['y1']=max(B['y1'],y0+h)
        B['lines'].setdefault((p,l),[]).append(r['text'])
    res=[]
    for k,B in blocks.items():
        text=' '.join(' '.join(v) for v in B['lines'].values())
        if len(text.replace(' ',''))>=3:
            res.append({'box':[int(B['x0']),int(B['y0']),int(B['x1']),int(B['y1'])],'text':text})
    out[i]=res
json.dump(out, open('blocks.json','w'), ensure_ascii=False, indent=1)
print('pages', len(out), 'total blocks', sum(len(v) for v in out.values()))
