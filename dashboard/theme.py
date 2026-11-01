"""Look and feel: one stylesheet and the small HTML pieces the pages share.

The theme is light only (pinned in .streamlit/config.toml): Streamlit's own widgets cannot follow a custom dark palette reliably. Chart colours (blue = generation, observed or central
estimate; orange = the high-emissions scenario or a forecast) were checked with the
dataviz palette validator.
"""
from __future__ import annotations

import html

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9aa3a7"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Source+Sans+3:wght@400;600&display=swap');
:root{--ink:#12303a;--muted:#4f6670;--paper:#f3f7f7;--card:#ffffff;--line:#d3e0e3;--sea:#1f6f8b;--soft:#e4eff1;--warn:#8a5a00;--warnbg:#fff4dc}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--paper)!important;color:var(--ink)}
.stApp,.stApp p,.stApp li,.stApp label,.stApp span,.stMarkdown{font-family:'Source Sans 3',system-ui,sans-serif;color:var(--ink)}
h1,h2,h3,h4{font-family:'Bricolage Grotesque','Source Sans 3',system-ui,sans-serif!important;color:var(--ink)!important;letter-spacing:-0.01em}
[data-testid="stSidebar"],[data-testid="collapsedControl"],#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
[data-testid="stHeader"]{background:transparent;height:0}
.block-container{max-width:1080px;padding-top:1.2rem;padding-bottom:4rem}
.brand{display:flex;align-items:baseline;gap:.6rem;margin:0 0 .4rem}
.brand b{font-family:'Bricolage Grotesque',sans-serif;font-size:1.15rem;color:var(--ink)}
.brand span{color:var(--muted);font-size:.95rem}
[data-testid="stSegmentedControl"] button,[data-testid="stBaseButton-segmented_control"],[data-testid="stBaseButton-segmented_controlActive"]{font-family:'Source Sans 3',sans-serif}
.hero{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:2rem;align-items:center;padding:1.2rem 0 1.6rem}
.hero h1{font-size:2.35rem;line-height:1.1;margin:0 0 .8rem;font-weight:700}
.hero p{font-size:1.1rem;line-height:1.5;color:var(--muted);max-width:34rem;margin:0}
@media(max-width:820px){.hero{grid-template-columns:1fr}.hero h1{font-size:1.9rem}}
.ribbon{width:100%;height:auto;display:block}
.ribbon .band{fill:var(--soft)}
.ribbon .area{fill:var(--sea);opacity:.13}
.ribbon .line{fill:none;stroke:var(--sea);stroke-width:3;stroke-linecap:round;stroke-dasharray:1;stroke-dashoffset:1;animation:draw 2.2s cubic-bezier(.3,.1,.2,1) .2s forwards}
.ribbon .dot{fill:var(--sea);opacity:0;animation:pop .4s ease-out 2.3s forwards}
.ribbon text{font-family:'Source Sans 3',sans-serif;font-size:15px;fill:var(--muted)}
.ribbon text.cap{fill:var(--ink);font-size:17px;font-weight:600}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes pop{to{opacity:1}}
@media(prefers-reduced-motion:reduce){.ribbon .line{animation:none;stroke-dashoffset:0}.ribbon .dot{animation:none;opacity:1}}
.answers{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin:.4rem 0 1.4rem}
@media(max-width:820px){.answers{grid-template-columns:1fr}}
.answer{border-top:3px solid var(--sea);padding:.8rem .2rem 0}
.answer .q{color:var(--muted);font-size:.95rem;margin:0 0 .3rem}
.answer .a{font-family:'Bricolage Grotesque',sans-serif;font-size:1.3rem;line-height:1.25;font-weight:600;margin:0 0 .5rem;color:var(--ink)}
.answer .d{font-size:.98rem;line-height:1.45;color:var(--muted);margin:0 0 .6rem}
.sure{display:flex;align-items:center;gap:.45rem;font-size:.92rem;color:var(--ink)}
.sure i{width:.62rem;height:.62rem;border-radius:50%;background:var(--line);display:inline-block}
.sure i.on{background:var(--sea)}
.note{background:var(--warnbg);color:var(--warn);border-radius:6px;padding:.8rem 1rem;font-size:.97rem;line-height:1.45;margin:.8rem 0}
.note b{color:var(--warn)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:1rem 1.2rem;margin:.6rem 0}
.big{font-family:'Bricolage Grotesque',sans-serif;font-size:2rem;font-weight:700;color:var(--ink);line-height:1.1}
.small{color:var(--muted);font-size:.95rem}
.verdict{font-family:'Bricolage Grotesque',sans-serif;font-size:1.25rem;line-height:1.3;font-weight:600;margin:.2rem 0 .4rem}
.stApp [data-testid="stMetricValue"]{font-family:'Bricolage Grotesque',sans-serif}
.stApp [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:6px}
[data-testid="stSlider"] *,[data-testid="stTickBarMin"],[data-testid="stTickBarMax"],[data-testid="stThumbValue"]{font-family:'Source Sans 3',system-ui,sans-serif!important}
h2{font-size:1.5rem!important;margin-top:1.6rem!important}
h3{font-size:1.15rem!important}
"""


def inject(st) -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def sure(level: int, label: str) -> str:
    """Three-dot confidence meter with a plain label: level 1 = low, 3 = high."""
    dots = "".join(f'<i class="{"on" if i < level else ""}"></i>' for i in range(3))
    return f'<div class="sure">{dots}<span>{esc(label)}</span></div>'


def answer(question: str, headline: str, detail: str, level: int, label: str) -> str:
    return (f'<div class="answer"><p class="q">{esc(question)}</p><p class="a">{esc(headline)}</p>'
            f'<p class="d">{esc(detail)}</p>{sure(level, label)}</div>')


def note(text: str) -> str:
    return f'<div class="note">{text}</div>'


def ribbon_svg(monthly: list[float], months: list[str], hi_from: int, hi_to: int, caption: str) -> str:
    """Seasonal curve drawn once on load. monthly = 12 mean values; hi_from..hi_to = highlighted months (1-based)."""
    W, H, L, R, T, B = 620, 250, 28, 14, 34, 34
    top = max(monthly) * 1.12
    xs = [L + (W - L - R) * i / 11 for i in range(12)]
    ys = [T + (H - T - B) * (1 - v / top) for v in monthly]
    d = f"M{xs[0]:.1f},{ys[0]:.1f}"
    for i in range(11):  # smooth cubic through the points
        cx = (xs[i] + xs[i + 1]) / 2
        d += f" C{cx:.1f},{ys[i]:.1f} {cx:.1f},{ys[i + 1]:.1f} {xs[i + 1]:.1f},{ys[i + 1]:.1f}"
    area = d + f" L{xs[-1]:.1f},{H - B} L{xs[0]:.1f},{H - B} Z"
    bx0, bx1 = xs[hi_from - 1] - 14, xs[hi_to - 1] + 14
    labels = "".join(f'<text x="{x:.1f}" y="{H - 12}" text-anchor="middle">{esc(m)}</text>' for x, m in zip(xs, months))
    peak = max(range(12), key=lambda i: monthly[i])
    return (
        f'<svg class="ribbon" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(caption)}">'
        f'<rect class="band" x="{bx0:.1f}" y="{T - 14}" width="{bx1 - bx0:.1f}" height="{H - B - T + 14}" rx="6"/>'
        f'<text class="cap" x="{(bx0 + bx1) / 2:.1f}" y="{T - 20}" text-anchor="middle">{esc(caption)}</text>'
        f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="var(--line)"/>'
        f'<path class="area" d="{area}"/><path class="line" pathLength="1" d="{d}"/>'
        f'<circle class="dot" cx="{xs[peak]:.1f}" cy="{ys[peak]:.1f}" r="5"/>{labels}</svg>'
    )
