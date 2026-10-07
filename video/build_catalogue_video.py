import os, subprocess, math, random
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from kokoro_onnx import Kokoro

D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out"); os.makedirs(OUT, exist_ok=True)
W, H, FPS = 1080, 1920, 30
DARK, CREAM, BEIGE, GOLD, INK, GREY = (20,20,20), (242,239,233), (228,221,208), (168,138,98), (24,24,24), (110,105,98)
FD = os.path.join(D, "fonts")
def F(kind, w, size):
    p = {"j": f"{FD}/fontsource-jost-5.3.0/files/jost-latin-{w}-normal.woff",
         "c": f"{FD}/fontsource-cormorant-garamond-5.3.0/files/cormorant-garamond-latin-{w}-normal.woff",
         "ci": f"{FD}/fontsource-cormorant-garamond-5.3.0/files/cormorant-garamond-latin-{w}-italic.woff"}[kind]
    return ImageFont.truetype(p, size)

# ---------- PDF crops (coords measured on 827px-wide preview, x3 for 300 dpi) ----------
pages = {i: Image.open(f"{D}/pg-{i:02d}.png").convert("RGB") for i in (1, 2, 6, 7, 8)}
def crop(p, x0, y0, x1, y1, s=3.0):
    return pages[p].crop((int(x0*s), int(y0*s), int(x1*s), int(y1*s)))
logo_dark = crop(1, 300, 370, 527, 590)
logo_light = crop(2, 340, 720, 487, 860)
homme = [("Costumes","Coupes ajustées, tissus de qualité"),("Chemises","Coton, cols nets, coloris classiques"),("Jeans","Denim brut, coupes droites"),
         ("Chaussures","Cuir, modèles habillés"),("Ceintures","Cuir, boucles sobres"),("Cravates","Soie, motifs discrets"),
         ("Pochettes","La touche finale"),("Montres","L'élégance au poignet"),("Parfums","Une signature olfactive")]
cols6, rows6 = [(75,282),(309,518),(545,752)], [(199,376),(470,647),(740,917)]
for i,(n,d) in enumerate(homme):
    (x0,x1),(y0,y1) = cols6[i%3], rows6[i//3]
    homme[i] = (n, d, crop(6, x0+2, y0+2, x1-2, y1-2))
maison = [("Fauteuils","Velours, cuir, assise généreuse"),("Armoires","Bois massif, finitions laiton"),("Rideaux","Lin et tissus lourds"),("Tapis","Tissés, motifs sobres"),
          ("Lampes en bois","Bois sculpté, abat-jour lin"),("Lustres","Cristal et laiton"),("Œuvres en bois","Pièces d'art uniques"),("Sur mesure","Votre idée, visualisée avant fabrication")]
cols7, rows7 = [(75,230),(249,404),(423,578),(597,752)], [(199,354),(466,621)]
for i,(n,d) in enumerate(maison):
    (x0,x1),(y0,y1) = cols7[i%4], rows7[i//4]
    maison[i] = (n, d, crop(7, x0+2, y0+2, x1-2, y1-2))
phone_cm = crop(8, 72, 232, 401, 892)
phone_home = crop(8, 427, 232, 756, 892)

# ---------- voice ----------
k = Kokoro(f"{D}/tts/kokoro-v1.0.onnx", f"{D}/tts/voices-v1.0.bin")
SR = 24000
def tts(text, speed=0.9):
    s, sr = k.create(text, voice="ff_siwis", speed=speed, lang="fr-fr")
    assert sr == SR
    return s.astype(np.float32)

voice_h = {"Costumes":"Des costumes aux coupes ajustées, dans des tissus de qualité.","Chemises":"Des chemises en coton, aux cols nets et aux coloris classiques.",
  "Jeans":"Des jeans en denim brut, en coupe droite.","Chaussures":"Des chaussures en cuir, aux modèles habillés.","Ceintures":"Des ceintures en cuir, aux boucles sobres.",
  "Cravates":"Des cravates en soie, aux motifs discrets.","Pochettes":"La pochette : la touche finale.","Montres":"La montre : l'élégance au poignet.",
  "Parfums":"Et nos parfums : une signature olfactive."}
voice_m = {"Fauteuils":"Des fauteuils en velours ou en cuir, à l'assise généreuse.","Armoires":"Des armoires en bois massif, aux finitions laiton.",
  "Rideaux":"Des rideaux en lin et en tissus lourds.","Tapis":"Des tapis tissés, aux motifs sobres.","Lampes en bois":"Des lampes en bois sculpté, avec abat-jour en lin.",
  "Lustres":"Des lustres en cristal et laiton.","Œuvres en bois":"Des œuvres en bois : des pièces d'art uniques.",
  "Sur mesure":"Et le sur mesure : votre idée, visualisée avant fabrication."}

# ---------- drawing helpers ----------
def ease(t): t = max(0.0, min(1.0, t)); return 1 - (1 - t) ** 3
def ctext(dr, y, txt, font, fill, spacing=0, alpha=1.0, bg=None, x=None):
    if spacing:
        widths = [dr.textlength(c, font=font) for c in txt]; tw = sum(widths) + spacing*(len(txt)-1)
        cx = (W - tw)/2 if x is None else x
        for c, w_ in zip(txt, widths):
            dr.text((cx, y), c, font=font, fill=mix(fill, bg, alpha)); cx += w_ + spacing
    else:
        tw = dr.textlength(txt, font=font)
        dr.text(((W - tw)/2 if x is None else x, y), txt, font=font, fill=mix(fill, bg, alpha))
def mix(c, bg, a):
    if bg is None or a >= 1: return c
    return tuple(int(bg[i] + (c[i]-bg[i])*max(0,a)) for i in range(3))
def paste_fade(base, im, xy, a):
    if a <= 0: return
    if a >= 1: base.paste(im, xy); return
    region = base.crop((xy[0], xy[1], xy[0]+im.width, xy[1]+im.height))
    base.paste(Image.blend(region, im, a), xy)
def scaled(im, w): return im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)

# ---------- scenes: each is (voice_text, render(t, dur) -> Image) ----------
scenes = []
def scene(text, pad=0.9, min_dur=0):
    def deco(fn): scenes.append((text, fn, pad, min_dur)); return fn
    return deco

LOGO_D = scaled(logo_dark, 520)
@scene("Ascension. L'art de s'élever.", pad=1.4, min_dur=4.0)
def s_intro(t, d):
    im = Image.new("RGB", (W, H), DARK); dr = ImageDraw.Draw(im)
    a = ease(t/1.2); paste_fade(im, LOGO_D, ((W-LOGO_D.width)//2, 560 - int(40*(1-a))), a)
    ctext(dr, 1080, "ASCENSION", F("j",300,92), CREAM, spacing=34, alpha=ease((t-0.8)/1.0), bg=DARK)
    ctext(dr, 1230, "L'art de s'élever", F("ci",400,72), GOLD, alpha=ease((t-1.6)/1.0), bg=DARK)
    ctext(dr, 1700, "CATALOGUE", F("j",400,30), (150,145,138), spacing=14, alpha=ease((t-2.2)/1.0), bg=DARK)
    return im

def chapter(title, sub, tagline, bg, fg, logo):
    L = scaled(logo, 380)
    def r(t, d):
        im = Image.new("RGB", (W, H), bg); dr = ImageDraw.Draw(im)
        a = ease(t/1.0); paste_fade(im, L, ((W-L.width)//2, 600), a)
        ctext(dr, 960, title, F("j",300,80), fg, spacing=28, alpha=ease((t-0.5)/0.9), bg=bg)
        if sub: ctext(dr, 1080, sub, F("j",400,46), GOLD, spacing=22, alpha=ease((t-0.8)/0.9), bg=bg)
        ctext(dr, 1200 if sub else 1100, tagline, F("j",400,30), mix(fg, bg, 0.6), spacing=10, alpha=ease((t-1.1)/0.9), bg=bg)
        return im
    return r

scene("Découvrez notre catalogue. Ascension : le prêt-à-porter homme haut de gamme, au style classique chic.", pad=1.0)(
    chapter("ASCENSION", None, "PRÊT-À-PORTER HOMME", DARK, CREAM, logo_dark))

def product(section, idx, total, name, desc, tile):
    T = scaled(tile, 860)
    def r(t, d):
        im = Image.new("RGB", (W, H), CREAM); dr = ImageDraw.Draw(im)
        ctext(dr, 150, section, F("j",400,30), GOLD, spacing=12, x=110)
        dr.text((W-110-dr.textlength(f"{idx:02d} / {total:02d}", font=F("j",300,30)), 150), f"{idx:02d} / {total:02d}", font=F("j",300,30), fill=GREY)
        a = ease(t/0.7)
        z = 1.0 + 0.05 * (t / max(d, 0.1))          # slow Ken Burns zoom
        Z = T.resize((int(T.width*z), int(T.height*z)), Image.LANCZOS)
        Z = Z.crop(((Z.width-T.width)//2, (Z.height-T.height)//2, (Z.width-T.width)//2+T.width, (Z.height-T.height)//2+T.height))
        paste_fade(im, Z, ((W-T.width)//2, 300 + int(50*(1-a))), a)
        y = 300 + T.height + 90
        b = ease((t-0.35)/0.7)
        dr.text((110 + int(30*(1-b)), y), name, font=F("c",500,110), fill=mix(INK, CREAM, b))
        lw = int((W-220) * ease((t-0.5)/0.8)); dr.line((110, y+150, 110+lw, y+150), fill=GOLD, width=3)
        c = ease((t-0.8)/0.7)
        dr.text((110, y+185), desc, font=F("j",300,44), fill=mix(GREY, CREAM, c))
        return im
    return r

for i, (n, d_, tile) in enumerate(homme, 1):
    scene(voice_h[n], pad=0.7)(product("ASCENSION · L'HOMME", i, len(homme), n, d_, tile))

scene("Ascension Hôme prolonge cet univers dans votre maison : mobilier, décoration, et créations sur mesure.", pad=1.0)(
    chapter("ASCENSION", "HOME", "MOBILIER · DÉCORATION · SUR MESURE", BEIGE, INK, logo_light))

for i, (n, d_, tile) in enumerate(maison, 1):
    scene(voice_m[n], pad=0.7)(product("ASCENSION HOME", i, len(maison), n, d_, tile))

@scene("En quatre étapes : votre idée ; le rendu par intelligence artificielle ; vos ajustements ; puis la fabrication et la livraison.", pad=1.0)
def s_steps(t, d):
    im = Image.new("RGB", (W, H), DARK); dr = ImageDraw.Draw(im)
    ctext(dr, 330, "LE SUR MESURE", F("j",400,34), GOLD, spacing=14, alpha=ease(t/0.6), bg=DARK)
    ctext(dr, 400, "en 4 étapes", F("ci",400,96), CREAM, alpha=ease(t/0.8), bg=DARK)
    steps = [("01","Votre idée"),("02","Le rendu IA"),("03","Vos ajustements"),("04","Fabrication et livraison")]
    for j, (num, lab) in enumerate(steps):
        a = ease((t - 0.8 - j*(d-2.0)/4) / 0.6)
        y = 700 + j*250
        dr.text((170 + int(40*(1-a)), y), num, font=F("c",500,120), fill=mix(GOLD, DARK, a))
        dr.text((380 + int(40*(1-a)), y+40), lab, font=F("j",300,56), fill=mix(CREAM, DARK, a))
        if j < 3: dr.line((170, y+200, 910, y+200), fill=mix((70,66,60), DARK, a), width=2)
    return im

PH = [scaled(phone_cm, 470), scaled(phone_home, 470)]
@scene("Tout se passe sur Ouatsape. Au Cameroun comme en République démocratique du Congo, votre conseiller vous accompagne : conseils, photos, rendez-vous, paiement par Mobile Money ou à la livraison.", pad=1.0)
def s_whatsapp(t, d):
    im = Image.new("RGB", (W, H), CREAM); dr = ImageDraw.Draw(im)
    ctext(dr, 170, "LA RELATION CLIENT", F("j",400,32), GOLD, spacing=12, alpha=ease(t/0.6), bg=CREAM)
    ctext(dr, 240, "Tout se passe", F("c",500,104), INK, alpha=ease(t/0.8), bg=CREAM)
    ctext(dr, 350, "sur WhatsApp.", F("c",500,104), INK, alpha=ease((t-0.3)/0.8), bg=CREAM)
    for j, p in enumerate(PH):
        a = ease((t - 0.6 - j*0.4)/0.8)
        paste_fade(im, p, (50 + j*510, 560 + int(80*(1-a))), a)
    ctext(dr, 1580, "CAMEROUN  ·  RD CONGO", F("j",400,40), INK, spacing=10, alpha=ease((t-2.0)/0.8), bg=CREAM)
    ctext(dr, 1660, "Un numéro par pays, un conseiller dédié", F("j",300,38), GREY, alpha=ease((t-2.4)/0.8), bg=CREAM)
    ctext(dr, 1730, "Paiement Mobile Money ou à la livraison", F("j",300,38), GREY, alpha=ease((t-2.8)/0.8), bg=CREAM)
    return im

LOGO_S = scaled(logo_dark, 360)
@scene("Ascension. Ascension Hôme. L'art de s'élever. Écrivez-nous dès aujourd'hui sur Ouatsape.", pad=2.0)
def s_outro(t, d):
    im = Image.new("RGB", (W, H), DARK); dr = ImageDraw.Draw(im)
    paste_fade(im, LOGO_S, ((W-LOGO_S.width)//2, 560), ease(t/1.0))
    ctext(dr, 900, "ASCENSION  |  ASCENSION HOME", F("j",300,38), CREAM, spacing=8, alpha=ease((t-0.4)/0.9), bg=DARK)
    ctext(dr, 1010, "L'art de s'élever", F("ci",400,80), GOLD, alpha=ease((t-1.0)/0.9), bg=DARK)
    a = ease((t-2.0)/0.8)
    bw, bh, by = 640, 120, 1300
    dr.rounded_rectangle(((W-bw)//2, by, (W+bw)//2, by+bh), radius=60, fill=mix(GOLD, DARK, a))
    ctext(dr, by+30, "Commandez sur WhatsApp", F("j",400,46), DARK)
    ctext(dr, 1500, "CAMEROUN  ·  RD CONGO", F("j",400,32), (150,145,138), spacing=12, alpha=ease((t-2.6)/0.8), bg=DARK)
    return im

# ---------- synthesize voice, compute timeline ----------
FADE = 0.35
voice_parts, durs = [], []
lead = 0.5
for text, fn, pad, mn in scenes:
    v = tts(text)
    d = max(lead + len(v)/SR + pad, mn)
    seg = np.zeros(int(d*SR), np.float32); seg[int(lead*SR):int(lead*SR)+len(v)] = v
    voice_parts.append(seg); durs.append(d)
voice = np.concatenate(voice_parts)
total = len(voice)/SR
print("scenes", len(scenes), "total %.1fs" % total)

# ---------- music bed: soft kalimba (thumb piano) + shaker, synthesized ----------
rng = np.random.default_rng(7)
N = len(voice) + SR; mus = np.zeros(N, np.float32)
def kalimba(f, dur=1.6):
    tt = np.arange(int(dur*SR))/SR
    env = np.exp(-tt*3.2) * (1-np.exp(-tt*400))
    return (np.sin(2*np.pi*f*tt) + 0.25*np.sin(2*np.pi*f*5.4*tt)*np.exp(-tt*18) + 0.12*np.sin(2*np.pi*f*2*tt)) * env
scale = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33]   # pentatonic C
bpm = 92; beat = 60/bpm
chords = [[0,2,4],[1,3,5],[0,3,5],[2,4,6]]
pattern = [0,2,1,2, 0,1,2,1]
b = 0
while b*beat/2*SR < N - 2*SR:
    pos = int(b*beat/2*SR); ch = chords[(b//16) % 4]
    if rng.random() < 0.85:
        n = kalimba(scale[ch[pattern[b % 8]]] * (0.5 if b % 8 == 0 else 1)) * (0.9 if b % 2 == 0 else 0.55)
        mus[pos:pos+len(n)] += n[:N-pos]
    sh = rng.standard_normal(int(0.06*SR)) * np.exp(-np.arange(int(0.06*SR))/SR*60)
    sh = np.convolve(sh, [1,-1], "same") * (0.12 if b % 2 else 0.06)
    mus[pos:pos+len(sh)] += sh[:N-pos]
    b += 1
mus = mus[:len(voice)]
mus /= np.abs(mus).max() + 1e-9
# duck music under voice
envv = np.convolve(np.abs(voice), np.ones(int(0.25*SR))/(0.25*SR), "same")
duck = np.where(envv > 0.01, 0.10, 0.22).astype(np.float32)
duck = np.convolve(duck, np.ones(int(0.3*SR))/(0.3*SR), "same")
fade = np.ones_like(mus); fl = int(1.5*SR); fade[:fl] = np.linspace(0,1,fl); fade[-fl*2:] = np.linspace(1,0,fl*2)
vn = voice / (np.abs(voice).max() + 1e-9) * 0.85
mixa = vn + mus*duck*fade
mixa /= max(1.0, np.abs(mixa).max()/0.95)
sf.write(f"{OUT}/audio.wav", mixa, SR)

# ---------- render video ----------
ff = subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",f"{W}x{H}","-r",str(FPS),"-i","-",
    "-i",f"{OUT}/audio.wav","-c:v","libx264","-preset","medium","-crf","20","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k",
    "-shortest","-movflags","+faststart",f"{OUT}/Ascension_Catalogue.mp4"], stdin=subprocess.PIPE)
prev_last = None
for si, ((text, fn, pad, mn), d) in enumerate(zip(scenes, durs)):
    nf = int(round(d*FPS))
    for f in range(nf):
        t = f/FPS
        fr = fn(t, d)
        if prev_last is not None and t < FADE:
            fr = Image.blend(prev_last, fr, t/FADE)
        if si == len(scenes)-1 and t > d-1.0:
            fr = Image.blend(fr, Image.new("RGB",(W,H),(0,0,0)), min(1,(t-(d-1.0))/1.0))
        ff.stdin.write(fr.tobytes())
    prev_last = fn(d, d)
    if si == 0:
        prev_last.save(f"{OUT}/thumb_intro.png")
    if si in (2, 13, len(scenes)-2): prev_last.save(f"{OUT}/thumb_{si}.png")
ff.stdin.close(); ff.wait()
print("done")
