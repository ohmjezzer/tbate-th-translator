import pytesseract, re
from PIL import Image
for p in range(1,42):
    im=Image.open(f'out/page-{p:04d}.png').convert('RGB')
    cfg='--psm 11'
    try:
        df=pytesseract.image_to_data(im, lang='eng', config=cfg, output_type=pytesseract.Output.DATAFRAME)
    except Exception as e:
        print(p,'ERR',e); continue
    df=df.dropna(subset=['text']); df['text']=df['text'].astype(str)
    words=[]
    for _,r in df.iterrows():
        t=r['text'].strip(); c=float(r['conf'])
        if len(re.sub(r'[^A-Za-z]','',t))>=3 and c>55:
            words.append(t)
    if words: print(p, words)
