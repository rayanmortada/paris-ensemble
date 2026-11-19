"""A gold line drawing of Paris, in the spirit of the château drawing in the Versailles app. Generated, so it stays small."""

W, H = 1600, 420
BASE = 410


def _rect(x, y, w, h):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}"/>'


def _line(x1, y1, x2, y2):
    return f'<path d="M{x1:.1f} {y1:.1f}L{x2:.1f} {y2:.1f}"/>'


def _window(x, y, w, h, arched=False):
    top = f'A{w / 2:.1f} {w / 2:.1f} 0 0 1 {x + w:.1f} {y:.1f}' if arched else f'L{x + w:.1f} {y:.1f}'
    return (f'<path d="M{x:.1f} {y + h:.1f}L{x:.1f} {y:.1f}{top}L{x + w:.1f} {y + h:.1f}Z"/>'
            f'<path d="M{x + w / 2:.1f} {y:.1f}V{y + h:.1f}"/>')


def haussmann(x, w, floors=5, floor_h=26):
    """A Paris apartment block: shop front, stone floors, balcony lines, cornice, mansard roof, chimneys."""
    out = []
    height = floors * floor_h + 22
    y0 = BASE - height
    out.append(_rect(x, y0, w, height))
    bays = max(3, int(w // 26))
    bay = w / bays
    for f in range(floors):
        fy = y0 + f * floor_h + 6
        for b in range(bays):
            out.append(_window(x + b * bay + bay * 0.26, fy, bay * 0.48, floor_h * 0.62))
        if f in (1, 4):                                   # the long balconies Haussmann blocks are known for
            out.append(_line(x, fy + floor_h * 0.66, x + w, fy + floor_h * 0.66))
            out.append(_line(x, fy + floor_h * 0.74, x + w, fy + floor_h * 0.74))
    for b in range(bays):                                 # shop fronts
        out.append(f'<path d="M{x + b * bay + 3:.1f} {BASE}V{BASE - 15:.1f}H{x + (b + 1) * bay - 3:.1f}V{BASE}"/>')
    out.append(_line(x - 3, y0, x + w + 3, y0))
    roof = 30
    out.append(f'<path d="M{x:.1f} {y0:.1f}L{x + 10:.1f} {y0 - roof:.1f}H{x + w - 10:.1f}L{x + w:.1f} {y0:.1f}"/>')
    for b in range(1, bays, 2):                           # dormers
        cx = x + b * bay + bay / 2
        out.append(f'<path d="M{cx - 6:.1f} {y0 - 4:.1f}V{y0 - 16:.1f}L{cx:.1f} {y0 - 23:.1f}L{cx + 6:.1f} {y0 - 16:.1f}V{y0 - 4:.1f}Z"/>')
    for cx in (x + w * 0.2, x + w * 0.78):                # chimneys
        out.append(_rect(cx, y0 - roof - 12, 12, 12))
    return "".join(out)


def eiffel(cx):
    """Four levels, curved legs, lattice bracing, the great arch at the base."""
    out = []
    y_base, y1, y2, y3, y_top = BASE, BASE - 72, BASE - 148, BASE - 300, BASE - 392
    w_base, w1, w2, w3 = 112, 72, 42, 13
    # outer and inner legs
    for s in (-1, 1):
        out.append(f'<path d="M{cx + s * w_base:.1f} {y_base}Q{cx + s * (w1 + 8):.1f} {(y_base + y1) / 2:.1f} {cx + s * w1:.1f} {y1}'
                   f'L{cx + s * w2:.1f} {y2}Q{cx + s * (w3 + 8):.1f} {(y2 + y3) / 2:.1f} {cx + s * w3:.1f} {y3}L{cx + s * 2.5:.1f} {y_top + 18}"/>')
        out.append(f'<path d="M{cx + s * (w_base - 30):.1f} {y_base}Q{cx + s * (w1 - 14):.1f} {(y_base + y1) / 2:.1f} {cx + s * (w1 - 20):.1f} {y1}"/>')
        out.append(f'<path d="M{cx + s * (w1 - 20):.1f} {y1}L{cx + s * (w2 - 12):.1f} {y2}"/>')
    # the great arch
    out.append(f'<path d="M{cx - 84:.1f} {y_base - 6}A84 62 0 0 1 {cx + 84:.1f} {y_base - 6}"/>')
    out.append(f'<path d="M{cx - 74:.1f} {y_base - 6}A74 54 0 0 1 {cx + 74:.1f} {y_base - 6}"/>')
    # platforms
    for y, half, h in ((y1, w1 + 8, 9), (y2, w2 + 6, 8), (y3, w3 + 5, 7)):
        out.append(_rect(cx - half, y - h, 2 * half, h))
        for k in range(-int(half) + 6, int(half), 8):
            out.append(_line(cx + k, y - h, cx + k, y))
    # lattice on the legs, first stage
    for s in (-1, 1):
        for t in range(6):
            a = t / 6
            b = (t + 1) / 6
            xa = w_base + (w1 - w_base) * a
            xb = w_base + (w1 - w_base) * b
            ya = y_base + (y1 - y_base) * a
            yb = y_base + (y1 - y_base) * b
            out.append(_line(cx + s * xa, ya, cx + s * (xb - 28), yb))
            out.append(_line(cx + s * (xa - 28), ya, cx + s * xb, yb))
    # lattice, second stage
    for s in (-1, 1):
        for t in range(6):
            a, b = t / 6, (t + 1) / 6
            ya, yb = y1 + (y2 - y1) * a, y1 + (y2 - y1) * b
            xa, xb = w1 + (w2 - w1) * a, w1 + (w2 - w1) * b
            ia, ib = (w1 - 20) + ((w2 - 12) - (w1 - 20)) * a, (w1 - 20) + ((w2 - 12) - (w1 - 20)) * b
            out.append(_line(cx + s * xa, ya, cx + s * ib, yb))
            out.append(_line(cx + s * ia, ya, cx + s * xb, yb))
    # lattice, upper stage: X bracing inside the narrowing shaft
    steps = 13
    for t in range(steps):
        a, b = t / steps, (t + 1) / steps
        ya, yb = y2 + (y3 - y2) * a, y2 + (y3 - y2) * b
        ha = w2 + (w3 - w2) * (a ** 0.55)
        hb = w2 + (w3 - w2) * (b ** 0.55)
        out.append(_line(cx - ha, ya, cx + hb, yb))
        out.append(_line(cx + ha, ya, cx - hb, yb))
    steps = 9
    for t in range(steps):
        a, b = t / steps, (t + 1) / steps
        ya, yb = y3 + (y_top + 18 - y3) * a, y3 + (y_top + 18 - y3) * b
        ha = w3 + (2.5 - w3) * a
        hb = w3 + (2.5 - w3) * b
        out.append(_line(cx - ha, ya, cx + hb, yb))
        out.append(_line(cx + ha, ya, cx - hb, yb))
    # top and antenna
    out.append(_rect(cx - 5, y_top + 8, 10, 10))
    out.append(_line(cx, y_top + 8, cx, y_top))
    return "".join(out)


def arc_de_triomphe(cx, scale=1.0):
    w, h = 132 * scale, 112 * scale
    x, y = cx - w / 2, BASE - h
    out = [_rect(x, y, w, h)]
    out.append(_rect(x - 3, y - 7, w + 6, 7))                               # attic
    out.append(_rect(x + 6, y - 17, w - 12, 10))
    aw, ah = 28 * scale, 78 * scale
    out.append(f'<path d="M{cx - aw:.1f} {BASE}V{BASE - ah + aw:.1f}A{aw:.1f} {aw:.1f} 0 0 1 {cx + aw:.1f} {BASE - ah + aw:.1f}V{BASE}"/>')
    out.append(f'<path d="M{cx - aw - 5:.1f} {BASE}V{BASE - ah + aw:.1f}A{aw + 5:.1f} {aw + 5:.1f} 0 0 1 {cx + aw + 5:.1f} {BASE - ah + aw:.1f}V{BASE}"/>')
    out.append(_line(x, y + 16 * scale, x + w, y + 16 * scale))
    out.append(_line(x, y + 22 * scale, x + w, y + 22 * scale))
    for s in (-1, 1):                                                       # reliefs on the piers
        px = cx + s * (aw + (w / 2 - aw) / 2) - 13 * scale
        out.append(_rect(px, BASE - h + 34 * scale, 26 * scale, 28 * scale))
        out.append(_rect(px, BASE - 34 * scale, 26 * scale, 24 * scale))
    return "".join(out)


def notre_dame(cx):
    out = []
    tw, th = 52, 182                                                         # the two square towers
    gap = 54
    for s in (-1, 1):
        x = cx + s * (gap / 2 + tw / 2) - tw / 2
        out.append(_rect(x, BASE - th, tw, th))
        out.append(_rect(x - 2, BASE - th - 5, tw + 4, 5))
        for k in (0, 1):                                                     # the tall twin bays of the belfries
            out.append(_window(x + 7 + k * 21, BASE - th + 14, 16, 44, arched=True))
        out.append(_line(x, BASE - th + 68, x + tw, BASE - th + 68))
        out.append(_line(x, BASE - 62, x + tw, BASE - 62))
        out.append(f'<path d="M{x + 8:.1f} {BASE}V{BASE - 34}A18 18 0 0 1 {x + tw - 8:.1f} {BASE - 34}V{BASE}"/>')
    x = cx - gap / 2
    out.append(_rect(x, BASE - 150, gap, 150))
    out.append(_line(x - tw, BASE - 96, x + gap + tw, BASE - 96))            # gallery of kings
    out.append(_line(x - tw, BASE - 88, x + gap + tw, BASE - 88))
    for k in range(9):
        out.append(_line(x + 3 + k * 6, BASE - 96, x + 3 + k * 6, BASE - 88))
    ry = BASE - 122                                                          # the rose window
    out.append(f'<circle cx="{cx}" cy="{ry}" r="20"/><circle cx="{cx}" cy="{ry}" r="7"/>')
    for k in range(12):
        import math
        a = k * math.pi / 6
        out.append(_line(cx + 7 * math.cos(a), ry + 7 * math.sin(a), cx + 20 * math.cos(a), ry + 20 * math.sin(a)))
    out.append(f'<path d="M{cx - 18} {BASE}V{BASE - 40}A18 18 0 0 1 {cx + 18} {BASE - 40}V{BASE}"/>')
    # nave and spire behind
    out.append(f'<path d="M{cx + gap / 2 + tw} {BASE - 92}L{cx + gap / 2 + tw + 18} {BASE - 92}V{BASE}"/>')
    out.append(f'<path d="M{cx - 2} {BASE - 150}L{cx + 8} {BASE - 262}L{cx + 18} {BASE - 150}"/>')
    return "".join(out)


def sacre_coeur(cx):
    y0 = BASE - 62
    out = [f'<path d="M{cx - 130} {BASE}Q{cx - 60} {y0 + 8} {cx - 56} {y0}H{cx + 56}Q{cx + 60} {y0 + 8} {cx + 130} {BASE}"/>']   # the hill
    for k in range(5):
        y = BASE - 8 - k * 11
        half = 110 - k * 11
        out.append(_line(cx - half, y, cx + half, y))
    out.append(_rect(cx - 46, y0 - 74, 92, 74))
    out.append(f'<path d="M{cx - 15} {y0}V{y0 - 30}A15 15 0 0 1 {cx + 15} {y0 - 30}V{y0}"/>')

    def dome(x, base, w, h):
        return (f'<path d="M{x - w} {base}C{x - w} {base - h * 0.8} {x - w * 0.25} {base - h} {x} {base - h - 8}'
                f'C{x + w * 0.25} {base - h} {x + w} {base - h * 0.8} {x + w} {base}Z"/>'
                + _line(x, base - h - 8, x, base - h - 22) + _rect(x - w, base, 2 * w, 10))

    out.append(_rect(cx - 22, y0 - 110, 44, 36))
    out.append(dome(cx, y0 - 110, 22, 62))
    for s in (-1, 1):
        out.append(dome(cx + s * 46, y0 - 84, 12, 30))
    return "".join(out)


def pyramid(cx):
    half, h = 62, 74
    out = [f'<path d="M{cx - half} {BASE}L{cx} {BASE - h}L{cx + half} {BASE}Z"/>']
    for k in range(1, 6):
        t = k / 6
        out.append(_line(cx - half * t, BASE - h + h * t, cx + half * t, BASE - h + h * t))
    for k in range(-5, 6):
        out.append(_line(cx, BASE - h, cx + half * k / 5, BASE))
    return "".join(out)


def dome_pantheon(cx):
    out = [_rect(cx - 64, BASE - 76, 128, 76)]
    out.append(f'<path d="M{cx - 72} {BASE - 76}L{cx} {BASE - 106}L{cx + 72} {BASE - 76}Z"/>')
    for k in range(-3, 4):
        out.append(_line(cx + k * 18, BASE, cx + k * 18, BASE - 76))
    out.append(_rect(cx - 34, BASE - 138, 68, 34))
    for k in range(-3, 4):
        out.append(_line(cx + k * 9.5, BASE - 138, cx + k * 9.5, BASE - 104))
    out.append(f'<path d="M{cx - 30} {BASE - 138}C{cx - 30} {BASE - 178} {cx - 8} {BASE - 192} {cx} {BASE - 192}C{cx + 8} {BASE - 192} {cx + 30} {BASE - 178} {cx + 30} {BASE - 138}"/>')
    out.append(_line(cx, BASE - 192, cx, BASE - 208))
    return "".join(out)


def tree(cx, r=17):
    h = BASE - r * 2 - 12
    return (f'<path d="M{cx} {BASE}V{h + r}"/>'
            f'<path d="M{cx - r} {h}a{r} {r * 0.85:.1f} 0 1 1 {2 * r} 0a{r} {r * 0.85:.1f} 0 1 1 {-2 * r} 0Z"/>')


def lamp(cx):
    return f'<path d="M{cx} {BASE}V{BASE - 34}"/><circle cx="{cx}" cy="{BASE - 38}" r="4"/>'


def svg(color: str = "#BA9E66", opacity: float = 0.62) -> str:
    parts = [
        haussmann(-16, 104, 5), tree(112), haussmann(132, 84, 6, 24),
        arc_de_triomphe(304), lamp(392), tree(416, 15),
        haussmann(444, 72, 5), pyramid(578), tree(652, 15),
        eiffel(800),
        tree(944, 15), haussmann(966, 76, 5), dome_pantheon(1118),
        tree(1200, 15), notre_dame(1296), lamp(1392),
        sacre_coeur(1486),
    ]
    ground = (f'<path d="M0 {BASE}H{W}"/>'
              + "".join(f'<path d="M{x} {BASE + 5}q10 -3 20 0t20 0"/>' for x in range(10, W, 112)))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMax slice" fill="none" '
            f'stroke="{color}" stroke-opacity="{opacity}" stroke-width="1" stroke-linejoin="round" stroke-linecap="round">'
            + "".join(parts) + ground + "</svg>")


EMBLEM = ('<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
          '<path d="M16 2v4"/><path d="M14.6 6h2.8"/><path d="M15 6 12.8 18M17 6l2.2 12"/><path d="M11.4 18h9.2"/>'
          '<path d="M12.2 18c-.6 3.6-2.6 8-6.2 12M19.8 18c.6 3.6 2.6 8 6.2 12"/><path d="M10.2 24h11.6"/>'
          '<path d="M11 30c.6-3 2.4-5 5-5s4.4 2 5 5"/><path d="M13.4 12h5.2"/></svg>')
