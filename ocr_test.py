import pytesseract
from PIL import Image
import pandas as pd, sys

im = Image.open('orig/page-0001.webp').convert('RGB')
df = pytesseract.image_to_data(im, lang='eng', config='--psm 3', output_type=pytesseract.Output.DATAFRAME)
df = df.dropna(subset=['text'])
df['text']=df['text'].astype(str)
df = df[df['text'].str.strip()!='']
print("rows", len(df))
for _,r in df.iterrows():
    if float(r['conf'])>40 and len(r['text'].strip())>=2:
        print(f"{int(r['left'])},{int(r['top'])},{int(r['width'])},{int(r['height'])} conf={int(float(r['conf']))} '{r['text']}'")
