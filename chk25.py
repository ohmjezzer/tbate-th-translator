import pytesseract,re
from PIL import Image
im=Image.open('out/page-0025.png').convert('RGB')
df=pytesseract.image_to_data(im, lang='eng', config='--psm 11', output_type=pytesseract.Output.DATAFRAME)
df=df.dropna(subset=['text'])
for _,r in df.iterrows():
    t=str(r['text']).strip()
    if len(re.sub(r'[^A-Za-z]','',t))>=3 and float(r['conf'])>60:
        print(int(r['left']),int(r['top']),repr(t))
