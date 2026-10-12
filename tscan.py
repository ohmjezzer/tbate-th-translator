import pytesseract, os
from PIL import Image
for i in range(1,42):
    f=f'orig/page-{i:04d}.webp'
    im=Image.open(f).convert('RGB')
    txt=pytesseract.image_to_string(im, lang='eng', config='--psm 11')
    cleaned=' | '.join([t.strip() for t in txt.splitlines() if len(t.strip())>=3])
    print(f'--- P{i} ---')
    print(cleaned[:600])
