"""Procedural thumbnails for feature tiles without illustrations (conceptual, not data)."""
import numpy as np, sys
from PIL import Image, ImageDraw, ImageFont
W,H=1536,1024
yy,xx=np.mgrid[0:H,0:W].astype(float)
GOLD=np.array([1.0,0.62,0.22]); GOLDW=np.array([1.0,0.85,0.55]); BLUE=np.array([0.15,0.65,1.0]); CYAN=np.array([0.4,0.9,1.0])
def canvas():
    r=np.hypot(xx-W/2,yy-H/2)/np.hypot(W/2,H/2); img=np.zeros((H,W,3)); img+=np.array([2,4,16])/255*(1.1-0.6*r[...,None]); return img
def blob(img,x0,y0,a,b,ang,col,amp=1.0):
    R=int(max(a,b)*5)+3; x1,x2=max(0,int(x0-R)),min(W,int(x0+R)); y1,y2=max(0,int(y0-R)),min(H,int(y0+R))
    if x2<=x1 or y2<=y1: return
    X=xx[y1:y2,x1:x2]-x0; Y=yy[y1:y2,x1:x2]-y0; c,s=np.cos(ang),np.sin(ang)
    u=(X*c+Y*s)/a; v=(-X*s+Y*c)/b; img[y1:y2,x1:x2]+=amp*np.exp(-0.5*(u*u+v*v))[...,None]*col
def lens(img,cx,cy,sc,rng,arcs=True):
    blob(img,cx,cy,60*sc,45*sc,0.4,GOLD,0.35); blob(img,cx,cy,18*sc,13*sc,0.4,GOLDW,1.0); blob(img,cx,cy,5*sc,4*sc,0.4,np.array([1,1,.9]),0.8)
    if arcs:
        for th0,span in [(rng.uniform(0,6.3),1.0),(rng.uniform(0,6.3),0.7)]:
            for t in np.linspace(th0-span/2,th0+span/2,60):
                f=max(np.cos((t-th0)/span*np.pi),0)**0.7
                blob(img,cx+110*sc*np.cos(t),cy+100*sc*np.sin(t),(8*f+2)*sc,(3.5*f+1)*sc,t+np.pi/2,BLUE,0.14)
def stars(img,rng,n=200):
    for _ in range(n): blob(img,rng.uniform(0,W),rng.uniform(0,H),0.7,0.7,0,np.array([.8,.85,1]),rng.uniform(.15,.5))
def finish(img): return Image.fromarray((np.clip(1-np.exp(-1.6*img),0,1)**0.9*255).astype(np.uint8))
def line(d,p,q,col,w=4,dash=None):
    if dash is None: d.line([p,q],fill=col,width=w); return
    p=np.array(p,float); q=np.array(q,float); L=np.linalg.norm(q-p); u=(q-p)/L; t=0
    while t<L: d.line([tuple(p+u*t),tuple(p+u*min(t+dash,L))],fill=col,width=w); t+=2*dash
font=lambda s: ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',s)

def mass_models(path):
    rng=np.random.default_rng(3); img=canvas(); stars(img,rng); lens(img,W/2,H/2,2.0,rng); im=finish(img); d=ImageDraw.Draw(im,'RGBA')
    cx,cy=W/2,H/2
    for k,r0 in enumerate([90,150,215,290,375]):
        t=np.linspace(0,2*np.pi,400); q=0.78; phi=0.4
        r=r0*(1+0.06*np.cos(4*(t-0.2))*(k/4))
        x=r*np.cos(t); y=r*q*np.sin(t); X=cx+x*np.cos(phi)-y*np.sin(phi); Y=cy+x*np.sin(phi)+y*np.cos(phi)
        pts=list(zip(X,Y)); col=(255,190,110,150-20*k)
        for i in range(0,len(pts)-1,8): d.line(pts[i:i+5],fill=col,width=4)
    im.save(path)

def slam(path):
    rng=np.random.default_rng(5); img=canvas(); stars(img,rng,120)
    xs=[W*0.2,W*0.5,W*0.8]; cy=H*0.47
    # Source: blurry source blob
    for _ in range(9): blob(img,xs[0]+rng.normal(0,35),cy+rng.normal(0,30),rng.uniform(15,30),rng.uniform(10,22),rng.uniform(0,3),CYAN,0.35)
    # Light: lens light only
    blob(img,xs[1],cy,90,68,0.4,GOLD,0.4); blob(img,xs[1],cy,26,19,0.4,GOLDW,1.0)
    # Mass: lens with arcs
    lens(img,xs[2],cy,1.3,rng)
    im=finish(img); d=ImageDraw.Draw(im,'RGBA')
    for i,x in enumerate(xs):
        d.rounded_rectangle([x-200,cy-210,x+200,cy+210],radius=28,outline=(120,170,255,140),width=5)
        if i<2:
            d.line([(x+215,cy),(xs[i+1]-225,cy)],fill=(200,220,255,220),width=8)
            d.polygon([(xs[i+1]-215,cy),(xs[i+1]-245,cy-18),(xs[i+1]-245,cy+18)],fill=(200,220,255,220))
    f=font(64)
    for x,t in zip(xs,['Source','Light','Mass']):
        w=d.textlength(t,font=f); d.text((x-w/2,cy+240),t,font=f,fill=(225,235,255,235))
    # mass contours on last panel
    for r0 in [70,120,170]:
        d.ellipse([xs[2]-r0,cy-r0*0.8,xs[2]+r0,cy+r0*0.8],outline=(255,190,110,110),width=3)
    im.save(path)

def graphical(path,kind):
    rng=np.random.default_rng(11); img=canvas(); stars(img,rng,150)
    cx,cy=W/2,H/2; n=8; pos=[(cx+430*np.cos(a),cy+330*np.sin(a)) for a in np.linspace(0,2*np.pi,n,endpoint=False)+0.2]
    for (x,y) in pos:
        if kind=='lens': lens(img,x,y,0.75,rng)
        else:
            ang=rng.uniform(0,3); q=rng.uniform(.4,.9)
            blob(img,x,y,55,55*q,ang,BLUE,0.35); blob(img,x,y,30,30*q,ang,GOLD,0.5); blob(img,x,y,9,9*q,ang,GOLDW,0.9)
    im=finish(img); d=ImageDraw.Draw(im,'RGBA')
    for (x,y) in pos:
        v=np.array([cx-x,cy-y]); L=np.linalg.norm(v); u=v/L
        line(d,tuple(np.array([x,y])+u*120),tuple(np.array([cx,cy])-u*95),(170,200,255,170),w=5,dash=18)
    d.ellipse([cx-85,cy-85,cx+85,cy+85],fill=(124,77,255,70),outline=(170,140,255,230),width=6)
    for k,dx in enumerate([-38,0,38]): d.ellipse([cx+dx-12,cy-12,cx+dx+12,cy+12],fill=(225,215,255,235))
    im.save(path)

out=sys.argv[1]; which=sys.argv[2]
{'mass':lambda:mass_models(out),'slam':lambda:slam(out),'graph_lens':lambda:graphical(out,'lens'),'graph_galaxy':lambda:graphical(out,'galaxy')}[which]()
