import pytesseract, sys
from PIL import Image
f=sys.argv[1]
im=Image.open(f).convert('RGB')
df=pytesseract.image_to_data(im, lang='eng', config='--psm 3', output_type=pytesseract.Output.DATAFRAME)
df=df.dropna(subset=['text'])
df['text']=df['text'].astype(str)
df=df[df['text'].str.strip()!='']
# group by block
import collections
blocks=collections.OrderedDict()
for _,r in df.iterrows():
    b=int(r['block_num'])
    blocks.setdefault(b,{'x0':1e9,'y0':1e9,'x1':-1,'y1':-1,'words':[]})
    B=blocks[b]
    x0,y0,w,h=int(r['left']),int(r['top']),int(r['width']),int(r['height'])
    B['x0']=min(B['x0'],x0); B['y0']=min(B['y0'],y0); B['x1']=max(B['x1'],x0+w); B['y1']=max(B['y1'],y0+h)
    B['words'].append(r['text'])
for b,B in blocks.items():
    if len(B['words'])>=2:
        print(f"block{b} box=({B['x0']},{B['y0']},{B['x1']},{B['y1']}) : {' '.join(B['words'])}")
