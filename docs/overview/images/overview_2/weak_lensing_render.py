import numpy as np, sys
from PIL import Image
rng=np.random.default_rng(7)
W,H=1536,1024; img=np.zeros((H,W,3))
yy,xx=np.mgrid[0:H,0:W].astype(float); cx,cy=W/2,H/2
# background: dark navy with soft vignette
r=np.hypot(xx-cx,yy-cy)/np.hypot(cx,cy)
bg=np.array([2,4,16])/255; img+=bg*(1.1-0.6*r[...,None])
def blob(x0,y0,a,b,ang,col,amp=1.0,prof='gauss'):
    R=int(max(a,b)*5)+3; x1,x2=max(0,int(x0-R)),min(W,int(x0+R)); y1,y2=max(0,int(y0-R)),min(H,int(y0+R))
    X=xx[y1:y2,x1:x2]-x0; Y=yy[y1:y2,x1:x2]-y0; c,s=np.cos(ang),np.sin(ang)
    u=(X*c+Y*s)/a; v=(-X*s+Y*c)/b; q=u*u+v*v
    f=np.exp(-0.5*q) if prof=='gauss' else np.exp(-2.0*q**0.4)
    img[y1:y2,x1:x2]+=amp*f[...,None]*np.array(col)
# faint stars
for _ in range(260):
    blob(rng.uniform(0,W),rng.uniform(0,H),0.7,0.7,0,np.array([0.8,0.85,1.0]),rng.uniform(0.15,0.6))
thE=150.0  # Einstein radius in px
gold=np.array([1.0,0.62,0.22]); goldw=np.array([1.0,0.85,0.55])
# background (weakly sheared) galaxies
n=0
while n<400:
    x,y=rng.uniform(-20,W+20),rng.uniform(-20,H+20); dx,dy=x-cx,y-cy; rr=np.hypot(dx,dy)
    if rr<thE*1.35: continue
    g=min(0.55,thE/(2*rr))*1.6  # exaggerated SIS shear for visibility
    e_int=rng.normal(0,0.12,2); phi=np.arctan2(dy,dx)+np.pi/2
    e1=g*np.cos(2*phi)+e_int[0]; e2=g*np.sin(2*phi)+e_int[1]; e=min(np.hypot(e1,e2),0.8)
    ang=0.5*np.arctan2(e2,e1); size=rng.lognormal(np.log(4.2),0.35)
    a=size*np.sqrt((1+e)/(1-e)); b=size*np.sqrt((1-e)/(1+e))
    hue=rng.uniform(0,1); col=np.array([0.25+0.35*hue,0.65+0.25*hue,1.0])
    blob(x,y,a,b,ang,col,rng.uniform(0.3,0.85)); n+=1
# strong arcs near Einstein radius
for th0,span,w,amp in [(0.35,0.9,4.5,1.0),(2.55,0.75,4.0,0.9),(4.25,0.6,3.5,0.8),(1.6,0.25,3.0,0.6)]:
    rad=thE*rng.uniform(0.97,1.05)
    for t in np.linspace(th0-span/2,th0+span/2,140):
        fall=max(np.cos((t-th0)/span*np.pi),0)**0.7
        blob(cx+rad*np.cos(t),cy+rad*np.sin(t)*0.92,7*fall+2,w*fall*0.8+0.8,t+np.pi/2,np.array([0.15,0.6,1.0]),0.07*amp)
    for t in rng.uniform(th0-span*0.4,th0+span*0.4,int(14*span)+3):
        blob(cx+rad*np.cos(t),cy+rad*np.sin(t)*0.92,rng.uniform(2,3.5),rng.uniform(1.5,2.5),t+np.pi/2,np.array([0.4,0.9,1.0]),rng.uniform(0.5,1.0)*amp)
# cluster members
blob(cx,cy,70,52,0.4,gold,0.35); blob(cx,cy,22,16,0.4,goldw,1.1); blob(cx,cy,6,5,0.4,np.array([1,1,0.9]),0.9)
for _ in range(16):
    t=rng.uniform(0,2*np.pi); rr=rng.uniform(40,230)
    s=rng.uniform(3.5,8); blob(cx+rr*np.cos(t)*1.2,cy+rr*np.sin(t)*0.85,s*1.2,s,rng.uniform(0,3),gold,rng.uniform(0.5,0.9))
    blob(cx+rr*np.cos(t)*1.2,cy+rr*np.sin(t)*0.85,s*3,s*2.5,0,gold,0.08)
out=1-np.exp(-1.6*img)  # soft tone map
Image.fromarray((np.clip(out,0,1)**0.9*255).astype(np.uint8)).save(sys.argv[1])
