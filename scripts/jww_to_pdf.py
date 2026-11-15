import sys,math,os
sys.path.insert(0,'scripts'); sys.stdout.reconfigure(encoding='utf-8')
import ezjww
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
MM=72/25.4
def hidden_set(d):
    """ファイルに保存された表示状態：大レイヤ・小レイヤの state 0 = 非表示（印刷しない）"""
    out=set()
    for gi,g in enumerate(d['header']['layer_groups']):
        for li in range(16):
            ls=g.get('layers',[{}]*16)[li].get('state',0) if g.get('layers') else 0
            if g['state']==0 or ls==0: out.add((gi,li))
    return out
HIDDEN=set()
def aux(e):
    b=e['base']; return b['pen_color']==9 or b['pen_style']==9
def draw(src,out):
    d=ezjww.read_document(src); HIDDEN.clear(); HIDDEN.update(hidden_set(d)); E=[e for e in d['entities'] if not aux(e) and (e['base'].get('layer_group',0),e['base'].get('layer',0)) not in HIDDEN]  # 非表示レイヤは印刷しない（Jw_cad と同じ）
    pens=d['header']['pens']['printer']; lt={x['number']:x['segments_mm'] for x in d['header']['line_types']['standard']}
    W,H=420*MM,297*MM
    c=canvas.Canvas(out,pagesize=(W,H)); c.setStrokeColorRGB(0,0,0); c.setFillColorRGB(0,0,0)
    P=lambda x,y:((x+210)*MM,(y+148.5)*MM)
    def setpen(b):
        w=pens[b['pen_color']]['width']*0.01 if b['pen_color']<len(pens) else 0.1
        c.setLineWidth(max(0.1,w)*MM)
        st=b['pen_style']
        if st in lt and st!=1: c.setDash([max(0.15,v)*MM for v in lt[st]])
        else: c.setDash([])
    n=0
    for e in E:
        t=e['type']; b=e['base']
        if t=='LINE':
            setpen(b); a1=P(e['start_x'],e['start_y']); a2=P(e['end_x'],e['end_y']); c.line(*a1,*a2); n+=1
        elif t=='POINT':
            x,y=P(e['x'],e['y']); c.circle(x,y,0.6*MM,stroke=0,fill=1); n+=1
        elif t=='CIRCLE':
            setpen(b); x,y=P(e['center_x'],e['center_y']); c.circle(x,y,e['radius']*MM,stroke=1,fill=0); n+=1
        elif t=='ARC':
            # 傾き（tilt_angle）と扁平率（flatness＝だ円）も使う。使わないと tilt=π の開閉線が反転して見えた（11/14 参考図面）
            setpen(b); r=e['radius']; f=e.get('flatness',1.0) or 1.0; tl=e.get('tilt_angle',0.0) or 0.0
            if e.get('is_full_circle'): a0,sw=0.0,2*math.pi
            else: a0=e['start_angle']; sw=e.get('arc_angle',0)
            k=max(12,int(abs(sw)*r/0.5)); ct,st=math.cos(tl),math.sin(tl)
            def pt(th):
                u=r*math.cos(th); v=r*f*math.sin(th)
                return P(e['center_x']+u*ct-v*st, e['center_y']+u*st+v*ct)
            pts=[pt(a0+sw*i/k) for i in range(k+1)]
            p=c.beginPath(); p.moveTo(*pts[0])
            for q in pts[1:]: p.lineTo(*q)
            c.drawPath(p,stroke=1,fill=0); n+=1
        elif t=='TEXT' and e.get('content'):
            try:
                size=max(2.0,abs(e.get('size_y') or 3))*MM
                x,y=P(e['start_x'],e['start_y']); c.saveState(); c.translate(x,y); c.rotate(float(e.get('angle',0) or 0))
                sx=(abs(e.get('size_x') or 3)/max(0.01,abs(e.get('size_y') or 3))); c.scale(sx,1); c.setFont('HeiseiKakuGo-W5',size); c.drawString(0,0,str(e['content'])); c.restoreState()
            except Exception: pass
    c.showPage(); c.save(); print(out,n)
draw(sys.argv[1],sys.argv[2])
