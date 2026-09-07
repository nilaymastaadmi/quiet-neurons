# Contrast measurement, 2026-09-07

The README claims WCAG AA in both themes. This is how that was measured, so the claim can be
re-checked rather than believed.

## Result

| theme | text elements | failures | floor |
|---|---|---|---|
| dark | 305 | **0** | **5.12:1** |
| light | 305 | **0** | **4.71:1** |

**The element count is state-dependent and is not the claim.** It moves with which controls are
active: injecting a surprise and switching layers add and remove readouts. An independent re-run
counted **315** against our 305 and reached identical floors and the same zero. Reproduce it in the
state described below and expect a count in that range. What must hold every time is the zero and
the floor.

## Rerun, 2026-09-08 04:4x, on the final build

Same script, same state, both themes, on a quiet machine after the n=8,192 training job finished.

| theme | text elements | failures | floor |
|---|---|---|---|
| dark | 451 | **0** | **5.12:1** |
| light | 451 | **0** | **4.71:1** |

**Both floors reproduce to two decimals** and both themes are still at zero failures. The dark floor
is the counterexample panel's "Same letter, no surprise" label; the light floor is a numeric
readout in the same panel.

The element count moved 305 to 451 and that is expected, not drift: the page gained four primer
boxes, the eight-word batch panel and its output table, and the live neuron counter in the
standfirst between the two measurements. The count is a function of how much text is on screen and
in what state. **The floor and the zero are the claim; the count is not.**

Threshold 4.5:1 for normal text, 3:1 for large (≥24 px, or ≥18.66 px at weight ≥700).

## What was wrong before

An earlier version of the README claimed a 4.69:1 floor. It was not reproducible on the shipped
build. Two real failures existed:

| theme | element | colours | ratio |
|---|---|---|---|
| dark | **"Inject a surprise" button** and pressed size buttons | `#FFFFFF` on `#FF9E3D` | **2.06:1** |
| dark | `.k` stat labels on the accent panel | `#868DA1` on `#37270F` | 4.34:1 |
| light | `.k` stat labels on the accent panel | `#666D82` on `#F7E7D6` | 4.26:1 |

The first is the worst kind: the falsification control, the single most important button on the
page, was the least readable thing on it.

Fixes: a new `--on-fire` token supplies the foreground for anything sitting on the accent
(`#0F1118` in dark, 9.17:1; white in light). The light accent darkened `#B45911` → `#9A4A0E`
(5.17:1 on the accent panel, up from 3.96:1). `--muted` moved `#666D82` → `#5A6070` in light and
`#868DA1` → `#939AAE` in dark.

## To reproduce

Serve `web/`, open `index.html`, click **Inject a surprise** so the active button states render,
then paste this in the console. Run it once per theme (emulate `prefers-color-scheme`, or use the
OS setting).

```js
const L = c => { const v = c/255; return v <= .04045 ? v/12.92 : Math.pow((v+.055)/1.055, 2.4); };
const lum = a => .2126*L(a[0]) + .7152*L(a[1]) + .0722*L(a[2]);
const parse = s => { const m = s.match(/rgba?\(([^)]+)\)/); if (!m) return null;
  const p = m[1].split(',').map(Number);
  if (p.length > 3 && p[3] === 0) return null; return p.slice(0,3); };
// resolve against the nearest OPAQUE ancestor, not the element's own transparent background
const bgOf = el => { let n = el; while (n && n !== document.documentElement) {
  const c = parse(getComputedStyle(n).backgroundColor); if (c) return c; n = n.parentElement; }
  return parse(getComputedStyle(document.body).backgroundColor); };
const ratio = (a,b) => { const la = lum(a), lb = lum(b);
  return (Math.max(la,lb) + .05) / (Math.min(la,lb) + .05); };

const out = [];
for (const el of document.querySelectorAll('body *')) {
  if (!el.offsetParent && el.tagName !== 'BODY') continue;      // skip hidden
  const txt = [...el.childNodes].filter(n => n.nodeType === 3)
                .map(n => n.textContent.trim()).join('');
  if (txt.length < 2) continue;                                  // own text only, no inherited
  const cs = getComputedStyle(el), fg = parse(cs.color);
  if (!fg) continue;
  const px = parseFloat(cs.fontSize);
  const large = px >= 24 || ((parseInt(cs.fontWeight,10)||400) >= 700 && px >= 18.66);
  const r = ratio(fg, bgOf(el));
  out.push({ r: +r.toFixed(2), need: large ? 3 : 4.5, pass: r >= (large ? 3 : 4.5), txt: txt.slice(0,30) });
}
out.sort((a,b) => a.r - b.r);
console.log({ total: out.length, failures: out.filter(o => !o.pass), floor: out.filter(o => o.pass)[0] });
```

Two details that matter, because getting either wrong produces a flattering number: it counts only
each element's **own** text nodes, so a container does not inherit its children's contrast; and it
walks up for the background, so elements with a transparent background are scored against what is
actually behind them rather than against white.

---

# Non-text contrast, WCAG 1.4.11

The section above covers **1.4.3, text only**, which is what the README's claim was scoped to.
1.4.11 is a separate criterion: visual information required to **identify a user interface
component** needs 3:1 against what is adjacent to it. Measured 2026-09-07, both themes.

## Result

| theme | failing controls | floor |
|---|---|---|
| dark | **0** | **3.36:1** |
| light | **0** | **4.01:1** |

A control passes if it is identifiable by *either* its fill or its border, whichever is stronger,
against the nearest opaque background behind it. Buttons on the accent are identified by fill
(8.54:1 dark, 6.25:1 light); the text input and the two segmented controls by border.

## What was wrong

The text input's boundary measured **1.34:1** in dark mode. It used `var(--line)`, which is a
decorative hairline chosen to be nearly invisible, and was never meant to carry the job of making
a component identifiable. On the page's primary control that is a real 1.4.11 failure. The
segmented layer and size selectors used the same token.

Fixed by adding a dedicated token, `--control-edge`, rather than lightening `--line` and coarsening
every divider on the page to fix one control: `#606B92` in dark, `#767F94` in light, each verified
against `--ground`, `--surface` and `--surface-2` so it holds wherever a control sits.

## What was checked and deliberately NOT changed

The first scan flagged 36 SVG elements. **35 are exempt and changing them would be wrong:**

- `<line>` **fill** at `rgb(0,0,0)`: a line element's fill never renders. Pure artefact of scanning
  computed styles rather than painted pixels.
- **Gridlines** at `rgb(34,38,50)` (`--line-soft`): 1.4.11 explicitly exempts decoration. Gridlines
  do not identify a component and are not required to perceive the data.
- **Phase-band shading** behind the trace (`--line-soft`, `--fire-soft`, `--pass-soft`): these tint
  regions that are *also* labelled in text, so colour is not the sole carrier of the information.

Reporting all 36 as failures and "fixing" them would have made the chart worse and the claim less
honest, not more. The distinction between a decorative hairline and a component boundary is the
whole point of 1.4.11, and a script that cannot draw it produces noise.

## To reproduce

Serve `web/`, open `index.html`, run once per theme:

```js
const L=c=>{const v=c/255;return v<=.04045?v/12.92:Math.pow((v+.055)/1.055,2.4)};
const lum=a=>.2126*L(a[0])+.7152*L(a[1])+.0722*L(a[2]);
const P=s=>{const m=String(s).match(/rgba?\(([^)]+)\)/); if(!m)return null;
  const q=m[1].split(',').map(Number); if(q.length>3&&q[3]===0)return null; return q.slice(0,3);};
const bgOf=el=>{let n=el.parentElement; while(n&&n!==document.documentElement){
  const c=P(getComputedStyle(n).backgroundColor); if(c)return c; n=n.parentElement;}
  return P(getComputedStyle(document.body).backgroundColor);};
const R=(a,b)=>{const x=lum(a),y=lum(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05);};

const res=[];
document.querySelectorAll('.ctl input[type=text], .seg, button').forEach(el=>{
  const cs=getComputedStyle(el), behind=bgOf(el);
  const bg=P(cs.backgroundColor), bc=P(cs.borderTopColor), bw=parseFloat(cs.borderTopWidth)||0;
  let ident=null;                                  // fill first, border as fallback
  if(bg&&behind&&R(bg,behind)>=3) ident=R(bg,behind);
  if(ident===null&&bc&&bw>0&&behind) ident=R(bc,behind);
  if(ident!==null) res.push({el:el.id||el.className, r:+ident.toFixed(2), pass:ident>=3});
});
console.log({failing:res.filter(x=>!x.pass), floor:Math.min(...res.map(x=>x.r))});
```
