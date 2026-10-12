# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pythainlp.tokenize import word_tokenize
import data

SRC='orig'; OUT='out'
REG='fonts/Sarabun-Regular.ttf'; BLD='fonts/Sarabun-Bold.ttf'
os.makedirs(OUT, exist_ok=True)

_fontcache={}
def F(path,size):
    k=(path,size)
    if k not in _fontcache: _fontcache[k]=ImageFont.truetype(path,size)
    return _fontcache[k]
def lum(c): return 0.299*c[0]+0.587*c[1]+0.114*c[2]

def dom_color(im, box):
    x0,y0,x1,y1=[int(v) for v in box]
    reg=im.crop((x0,y0,x1,y1))
    cols=reg.getcolors(3000000)
    if cols:
        cols.sort(key=lambda t:-t[0]); return cols[0][1]
    q=reg.convert('P',palette=Image.ADAPTIVE,colors=8).convert('RGB')
    rr=q.getcolors(200000); rr.sort(key=lambda t:-t[0]); return rr[0][1]

def tokenize(t):
    try:
        toks=[x for x in word_tokenize(t,keep_whitespace=False) if x.strip()!='']
        return toks if toks else list(t)
    except Exception:
        return list(t)

_tmpimg=Image.new('RGB',(10,10)); _tmpdraw=ImageDraw.Draw(_tmpimg)

def _chunk(tok,font,maxw,draw):
    pieces=[]; cur=''
    for ch in tok:
        if draw.textlength(cur+ch,font=font)<=maxw or cur=='':
            cur+=ch
        else:
            pieces.append(cur); cur=ch
    if cur: pieces.append(cur)
    return pieces

def wrap(text,font,maxw,draw):
    out=[]
    for para in text.split('\n'):
        toks=[]
        for tk in tokenize(para):
            if draw.textlength(tk,font=font)>maxw: toks.extend(_chunk(tk,font,maxw,draw))
            else: toks.append(tk)
        cur=''
        for tk in toks:
            if draw.textlength(cur+tk,font=font)<=maxw or cur=='': cur+=tk
            else: out.append(cur); cur=tk
        out.append(cur)
    return [l for l in out if l!='']

def fit(text,bw,bh,fontpath,maxsize=220):
    bw=max(bw,10)
    for strict in (True,False):
        for size in range(int(maxsize),9,-1):
            f=F(fontpath,size)
            if strict:
                toks=[t for p in text.split('\n') for t in tokenize(p)]
                if toks and max(_tmpdraw.textlength(t,font=f) for t in toks)>bw: continue
            lines=wrap(text,f,bw,_tmpdraw)
            lh=int(size*1.30)
            mw=max((_tmpdraw.textlength(l,font=f) for l in lines), default=0)
            if len(lines)*lh<=bh and mw<=bw*1.02: return f,lines,lh
    f=F(fontpath,10); return f,wrap(text,f,bw,_tmpdraw),13

def ring_bg(img, box, m=9):
    W,H=img.size
    x0,y0,x1,y1=[int(v) for v in box]
    X0=max(0,x0-m); Y0=max(0,y0-m); X1=min(W,x1+m); Y1=min(H,y1+m)
    arr=np.array(img.crop((X0,Y0,X1,Y1)).convert('RGB')).astype(np.int32)
    h,w,_=arr.shape
    mask=np.ones((h,w),bool)
    mask[(y0-Y0):(y1-Y0),(x0-X0):(x1-X0)]=False
    if mask.sum()<10: 
        mask[:]=True
    px=arr[mask]
    med=np.median(px,axis=0)
    return tuple(int(v) for v in med)

def fit_line(th, bw, bh, maxsize=220):
    maxsize=min(maxsize, int(bh*1.2))
    for size in range(max(11,int(maxsize)),9,-1):
        f=F(BLD,size)
        if _tmpdraw.textlength(th,font=f)<=bw:
            return f
    return F(BLD,11)

def inpaint_region(img, eb, bg, thr=72, radius=6):
    x0,y0,x1,y1=eb
    reg=np.array(img.crop((x0,y0,x1,y1)).convert('RGB'))
    diff=np.linalg.norm(reg.astype(np.int16)-np.array(bg,dtype=np.int16),axis=2)
    mask=(diff>thr).astype(np.uint8)*255
    if mask.sum()==0: return
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask=cv2.dilate(mask,k,iterations=2)
    out=cv2.inpaint(reg, mask, radius, cv2.INPAINT_TELEA)
    img.paste(Image.fromarray(out),(x0,y0))

def process(page):
    img=Image.open(f'{SRC}/page-{page:04d}.webp').convert('RGB')
    W,H=img.size
    draw=ImageDraw.Draw(img)
    for it in data.DATA.get(page,[]):
        x0,y0,x1,y1=it['b']; k=it['k']; th=it['th']
        x0=max(0,min(W-1,x0)); x1=max(1,min(W,x1)); y0=max(0,min(H-1,y0)); y1=max(1,min(H,y1))
        if x1<=x0 or y1<=y0: continue
        bg=dom_color(img,(x0,y0,x1,y1))
        cx=(x0+x1)//2; cy=(y0+y1)//2
        if k=='f':
            m=14
            eb=(max(0,x0-m),max(0,y0-m),min(W,x1+m),min(H,y1+m))
            bg2=ring_bg(img,(x0,y0,x1,y1),m=9)
            inpaint_region(img, eb, bg2, thr=60, radius=6)
            draw=ImageDraw.Draw(img)
            if '\n' in th:
                f,lines,lh=fit(th,(x1-x0)*1.05,(y1-y0)*1.7,BLD)
            else:
                f=fit_line(th,(x1-x0)*1.08,(y1-y0)*1.2)
                lines=[th]; lh=int(f.size*1.15)
            thh=len(lines)*lh; ty=cy-thh//2
            color=(255,255,255) if lum(bg2)<128 else (25,25,25)
            stroke=(0,0,0) if lum(bg2)<128 else (255,255,255)
            sw=max(2,int(f.size*0.13))
            for l in lines:
                lw=draw.textlength(l,font=f)
                draw.text((cx-lw/2,ty),l,font=f,fill=color,stroke_width=sw,stroke_fill=stroke)
                ty+=lh
        else:
            EX=4
            x0=max(0,x0-EX); y0=max(0,y0-EX); x1=min(W,x1+EX); y1=min(H,y1+EX)
            f,lines,lh=fit(th,(x1-x0)*0.92,(y1-y0)*1.35,REG)
            tw=max(draw.textlength(l,font=f) for l in lines)
            thh=len(lines)*lh
            pad=max(8,int(f.size*0.45))
            rw=max((x1-x0),tw)+pad*2
            rh=max((y1-y0),thh)+pad*2
            rx0=int(max(0,cx-rw/2)); rx1=int(min(W,cx+rw/2))
            ry0=int(max(0,cy-rh/2)); ry1=int(min(H,cy+rh/2))
            rad=max(2,min(14,(ry1-ry0)//3,(rx1-rx0)//3))
            draw.rounded_rectangle([rx0,ry0,rx1,ry1],radius=rad,fill=bg)
            tcol=(15,15,15) if lum(bg)>120 else (245,245,245)
            ty=cy-thh//2
            for l in lines:
                lw=draw.textlength(l,font=f)
                draw.text((cx-lw/2,ty),l,font=f,fill=tcol); ty+=lh
    img.save(f'{OUT}/page-{page:04d}.png')

if __name__=='__main__':
    pages=[int(a) for a in sys.argv[1:]] or list(range(1,42))
    for p in pages:
        process(p); print('done',p)
