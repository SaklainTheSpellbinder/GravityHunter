from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from gravityhunter.analysis.real_pipeline import RealCaseResult
from gravityhunter.dsp.templates import instantaneous_frequency

G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30


@dataclass
class AnimationPayload:
    data: dict

    def to_json(self) -> str:
        # Avoid accidentally terminating the script element if metadata ever contains </script>.
        return json.dumps(self.data, separators=(",", ":")).replace("</", "<\\/")


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 1800) -> tuple[list[float], list[float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return [], []
    if x.size > max_points:
        idx = np.linspace(0, x.size - 1, max_points).astype(int)
        x, y = x[idx], y[idx]
    return x.tolist(), y.tolist()


def _normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    scale = float(np.nanpercentile(np.abs(y), 99.5))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanmax(np.abs(y))) if y.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(y)
    return np.clip(y / scale, -1.0, 1.0)


def build_animation_payload(
    result: RealCaseResult,
    *,
    detector_name: str = "H1",
    sync_to: str = "detected",
) -> AnimationPayload:
    """Build a compact JSON payload for the browser-side merger animation.

    The orbital scene is schematic. Timing, waveform phase/frequency, detector
    strain, candidate time and GPS clock are taken from the selected real case.
    """
    spec = result.spec
    detector = result.h1 if detector_name == "H1" else result.l1
    record = detector.record
    duration = float(record.duration)
    catalog_t = result.event_time_s
    detected_t = detector.detection.peak_time_s

    if spec.expected_signal and catalog_t is not None:
        if sync_to == "detected" and detected_t is not None:
            collision_t = float(detected_t)
            sync_label = "GravityHunter detected time"
        else:
            collision_t = float(catalog_t)
            sync_label = "catalog reference time"
    else:
        collision_t = None
        sync_label = "no merger"

    phase_intervals: dict[str, list[float]] = {}
    if collision_t is not None and catalog_t is not None:
        shift = collision_t - catalog_t
        for name, (a, b) in result.phase_intervals_absolute.items():
            phase_intervals[name] = [float(a + shift), float(b + shift)]

    # Real processed detector strain for the full-record live strip.
    tx, white = _downsample_xy(record.time, _normalize(detector.white), max_points=1900)

    # SNR candidate-time axis for a second synchronized strip.
    candidate_t = (detector.lags + result.template_reference_index) / record.fs
    valid = (candidate_t >= 0) & (candidate_t <= duration) & np.isfinite(detector.snr)
    sx, sy = _downsample_xy(candidate_t[valid], detector.snr[valid], max_points=1900)

    # Reference waveform dynamics. Map the template peak to whichever time drives
    # the animation, so the collision is truly synchronized to the selected clock.
    tr = result.template
    rel_t = tr.time_from_reference
    amp = np.abs(tr.complex_template)
    amp_scale = max(float(np.max(amp)), np.finfo(float).tiny)
    amp_n = amp / amp_scale
    phase = np.unwrap(np.angle(tr.complex_template))
    phase = phase - phase[tr.peak_index]
    f_gw = instantaneous_frequency(tr)

    sl = tr.compact_slice(pre_pad_s=0.02, post_pad_s=0.05)
    rt = rel_t[sl]
    rp = phase[sl]
    rf = f_gw[sl]
    ra = amp_n[sl]

    finite = np.isfinite(rt) & np.isfinite(rp) & np.isfinite(ra)
    rt, rp, ra = rt[finite], rp[finite], ra[finite]
    rf = rf[finite]
    # Smooth browser interpolation by retaining enough points around the very short merger.
    if rt.size > 1800:
        idx = np.linspace(0, rt.size - 1, 1800).astype(int)
        rt, rp, rf, ra = rt[idx], rp[idx], rf[idx], ra[idx]

    total_mass = None
    if spec.m1_source is not None and spec.m2_source is not None:
        total_mass = float(spec.m1_source + spec.m2_source)

    separation_km = np.full_like(rf, np.nan, dtype=float)
    if total_mass is not None:
        f_ok = np.isfinite(rf) & (rf > 0)
        separation_km[f_ok] = (
            G_SI * total_mass * M_SUN_KG / (np.pi * rf[f_ok]) ** 2
        ) ** (1.0 / 3.0) / 1000.0

    # A clean reference waveform strip mapped into record time.
    ref_t = (collision_t if collision_t is not None else duration / 2.0) + rt
    ref_y = _normalize(tr.plus[sl][finite])
    if ref_t.size > 1800:
        idx = np.linspace(0, ref_t.size - 1, 1800).astype(int)
        ref_t, ref_y = ref_t[idx], ref_y[idx]

    gps_start = None if record.start_time is None else float(record.start_time)
    detected_gps = None if detected_t is None or gps_start is None else gps_start + float(detected_t)
    catalog_gps = None if spec.gps_event is None else float(spec.gps_event)

    payload = {
        "caseKey": spec.key,
        "caseLabel": spec.label,
        "detector": detector_name,
        "duration": duration,
        "fs": float(record.fs),
        "expectedSignal": bool(spec.expected_signal),
        "gpsStart": gps_start,
        "catalogGps": catalog_gps,
        "catalogUtc": spec.utc_event,
        "catalogTime": catalog_t,
        "detectedTime": detected_t,
        "detectedGps": detected_gps,
        "collisionTime": collision_t,
        "syncLabel": sync_label,
        "threshold": float(detector.detection.threshold),
        "peakSnr": float(detector.detection.peak_snr),
        "phases": phase_intervals,
        "masses": {
            "m1": spec.m1_source,
            "m2": spec.m2_source,
            "final": spec.final_mass_source,
            "radiated": spec.radiated_energy,
        },
        "realWave": {"t": tx, "y": white},
        "referenceWave": {"t": ref_t.tolist(), "y": ref_y.tolist()},
        "snr": {"t": sx, "y": sy},
        "dynamics": {
            "relT": rt.tolist(),
            "phase": rp.tolist(),
            "fGw": np.nan_to_num(rf, nan=0.0, posinf=0.0, neginf=0.0).tolist(),
            "amp": ra.tolist(),
            "separationKm": np.nan_to_num(separation_km, nan=0.0, posinf=0.0, neginf=0.0).tolist(),
        },
    }
    return AnimationPayload(payload)


def merger_animation_html(payload: AnimationPayload, *, height_px: int = 930) -> str:
    """Return a self-contained Canvas/SVG-free HTML animation for Streamlit components.html."""
    data_json = payload.to_json()
    # JS intentionally uses only local data. No network requests or external libraries.
    return f"""
<div id="gh-merger" style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#F1E6C5;background:linear-gradient(155deg,#081827,#040b13);border:1px solid rgba(221,229,236,.11);border-radius:18px;padding:16px;box-sizing:border-box;overflow:hidden;">
  <style>
    #gh-merger *{{box-sizing:border-box}}
    #gh-merger .top{{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:start;margin-bottom:12px}}
    #gh-merger .kicker{{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:#d1c3a2;font-weight:800}}
    #gh-merger .title{{font-size:20px;font-weight:750;margin-top:3px}}
    #gh-merger .sub{{font-size:12px;color:#afc2d3;margin-top:3px;max-width:760px}}
    #gh-merger .clock{{text-align:right;background:rgba(13,29,48,.78);border:1px solid rgba(221,229,236,.10);border-radius:12px;padding:9px 12px;min-width:220px}}
    #gh-merger .clock strong{{font-variant-numeric:tabular-nums;font-size:17px}}
    #gh-merger .clock div{{font-size:11px;color:#b8c9d8;margin-top:2px}}
    #gh-merger .stagebar{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:10px 0 12px}}
    #gh-merger .stage{{padding:7px 9px;border-radius:999px;text-align:center;border:1px solid rgba(221,229,236,.09);font-size:11px;color:#91a6b9;background:rgba(13,29,48,.58)}}
    #gh-merger .stage.on{{color:#07111c;background:#f1e6c5;font-weight:760}}
    #gh-merger .canvaswrap{{position:relative;border:1px solid rgba(221,229,236,.10);border-radius:15px;overflow:hidden;background:radial-gradient(circle at 50% 35%,rgba(31,65,94,.24),rgba(3,10,17,.97) 68%)}}
    #gh-merger canvas{{display:block;width:100%;height:auto}}
    #gh-merger .badge{{position:absolute;left:13px;top:12px;background:rgba(2,10,18,.72);border:1px solid rgba(244,231,194,.14);border-radius:12px;padding:8px 10px;font-size:11px;color:#d6e0e8}}
    #gh-merger .badge b{{color:#f4e7c2}}
    #gh-merger .controls{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0 8px}}
    #gh-merger button,#gh-merger select{{border-radius:11px;border:1px solid rgba(244,231,194,.17);background:#102941;color:#f1e6c5;padding:8px 11px;font-weight:700;font-size:12px}}
    #gh-merger button:hover{{background:#173653;cursor:pointer}}
    #gh-merger button:disabled{{opacity:.35;cursor:not-allowed}}
    #gh-merger label{{font-size:11px;color:#b8c9d8}}
    #gh-merger input[type=range]{{width:100%;accent-color:#d3b76b}}
    #gh-merger .timeline{{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;font-size:11px;color:#aebfd0}}
    #gh-merger .plots{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}}
    #gh-merger .panel{{background:rgba(7,27,49,.70);border:1px solid rgba(244,231,194,.10);border-radius:14px;padding:8px}}
    #gh-merger .panelhead{{font-size:11px;color:#c8d5e0;margin:0 0 5px 3px}}
    #gh-merger .readouts{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}}
    #gh-merger .readout{{background:rgba(7,27,49,.62);border:1px solid rgba(244,231,194,.10);border-radius:12px;padding:8px 10px}}
    #gh-merger .readout span{{display:block;color:#91a9bc;font-size:10px;text-transform:uppercase;letter-spacing:.08em}}
    #gh-merger .readout b{{display:block;margin-top:3px;font-size:14px;font-variant-numeric:tabular-nums}}
    #gh-merger .note{{font-size:10.5px;color:#91a9bc;margin-top:8px}}
    @media(max-width:720px){{#gh-merger .top{{grid-template-columns:1fr}}#gh-merger .clock{{text-align:left}}#gh-merger .plots{{grid-template-columns:1fr}}#gh-merger .readouts{{grid-template-columns:1fr 1fr}}}}
  </style>
  <div class="top">
    <div><div class="kicker">Event timeline</div><div class="title" id="gh-title"></div><div class="sub">Data-synchronized event playback with a schematic curvature surface. Timing, waveform phase, detector strain and matched-filter SNR are tied to the selected record; the geometry is not a numerical GR solution.</div></div>
    <div class="clock"><strong id="gh-record">t = 0.000 s</strong><div id="gh-gps">GPS —</div><div id="gh-sync"></div></div>
  </div>
  <div class="stagebar"><div class="stage" data-stage="Pre-event">PRE-EVENT</div><div class="stage" data-stage="Inspiral">INSPIRAL</div><div class="stage" data-stage="Merger">MERGER</div><div class="stage" data-stage="Ringdown">RINGDOWN</div></div>
  <div class="canvaswrap"><canvas id="gh-scene" width="1100" height="500"></canvas><div class="badge" id="gh-badge"></div></div>
  <div class="controls">
    <button type="button" id="gh-play">▶ Play full record</button><button type="button" id="gh-pause">Ⅱ Pause</button><button type="button" id="gh-restart">↺ Restart</button><button type="button" id="gh-jump">↦ Jump to merger</button><button type="button" id="gh-replay">↻ Replay merger</button>
    <label>Full-record speed <select id="gh-rate"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
    <label>Merger slow motion <select id="gh-slow"><option value="5">5× slower</option><option value="20" selected>20× slower</option><option value="50">50× slower</option></select></label>
  </div>
  <div class="timeline"><span>0 s</span><input id="gh-scrub" type="range" min="0" max="1" step="0.0001" value="0" aria-label="Record time scrubber"><span id="gh-end">32 s</span></div>
  <div class="plots"><div class="panel"><div class="panelhead">Processed detector strain + reference waveform</div><canvas id="gh-wave" width="720" height="180"></canvas></div><div class="panel"><div class="panelhead">Matched-filter SNR</div><canvas id="gh-snr" width="720" height="180"></canvas></div></div>
  <div class="readouts"><div class="readout"><span>Stage</span><b id="gh-stage-read">—</b></div><div class="readout"><span>GW frequency</span><b id="gh-freq">—</b></div><div class="readout"><span>Approx. separation</span><b id="gh-sep">—</b></div><div class="readout"><span>Peak SNR</span><b id="gh-snr-read">—</b></div></div>
  <div class="note">Synchronization may use the recovered candidate or the external catalog timestamp. Orbital phase follows approximately half the dominant GW phase; the inspiral separation readout uses a Newtonian estimate and is suppressed near merger.</div>
</div>
<script>
(()=>{{
  const D={data_json};
  const root=document.getElementById('gh-merger');
  if(root.dataset.ready==='1') return; root.dataset.ready='1';
  const scene=document.getElementById('gh-scene'), wave=document.getElementById('gh-wave'), snrC=document.getElementById('gh-snr');
  const ctx=scene.getContext('2d'), wctx=wave.getContext('2d'), sctx=snrC.getContext('2d');
  const scrub=document.getElementById('gh-scrub'); scrub.max=String(D.duration); document.getElementById('gh-end').textContent=D.duration.toFixed(1)+' s';
  document.getElementById('gh-title').textContent=D.caseLabel+' • '+D.detector;
  document.getElementById('gh-sync').textContent='collision sync: '+D.syncLabel;
  document.getElementById('gh-snr-read').textContent=Number.isFinite(D.peakSnr)?D.peakSnr.toFixed(2):'—';
  const noEvent=!D.expectedSignal || D.collisionTime==null;
  document.getElementById('gh-jump').disabled=noEvent; document.getElementById('gh-replay').disabled=noEvent;
  let t=0, playing=false, mode='full', last=performance.now();
  let eventEnd=D.duration;
  function lerp1(xs,ys,x,def=0){{if(!xs||!xs.length)return def;if(x<=xs[0])return ys[0];const n=xs.length;if(x>=xs[n-1])return ys[n-1];let lo=0,hi=n-1;while(hi-lo>1){{const m=(lo+hi)>>1;if(xs[m]<=x)lo=m;else hi=m}}const q=(x-xs[lo])/(xs[hi]-xs[lo]||1);return ys[lo]*(1-q)+ys[hi]*q;}}
  function stageAt(x){{if(noEvent)return 'No merger';const P=D.phases||{{}};for(const name of ['Inspiral','Merger','Ringdown']){{const p=P[name];if(p&&x>=p[0]&&x<=p[1])return name;}} if(P.Inspiral&&x<P.Inspiral[0])return 'Pre-event'; return 'Post-event';}}
  function rgba(hex,a){{const h=hex.replace('#','');const n=parseInt(h,16);return `rgba(${{(n>>16)&255}},${{(n>>8)&255}},${{n&255}},${{a}})`}}
  function dynAt(x){{if(noEvent)return {{phase:0,f:0,amp:0,sep:0}};const rel=x-D.collisionTime;const A=D.dynamics;return {{phase:lerp1(A.relT,A.phase,rel,0),f:lerp1(A.relT,A.fGw,rel,0),amp:lerp1(A.relT,A.amp,rel,0),sep:lerp1(A.relT,A.separationKm,rel,0)}};}}
  function drawFabric(cx,cy,bodies,stage,amp){{
    ctx.save();
    const top=258,bottom=472;
    function field(x,y){{
      let d=0;
      for(const q of bodies){{
        const wx=q.width||78, wy=q.height||86;
        const dx=(x-q.x)/wx, dy=(y-(q.planeY||top))/wy;
        d+=(q.depth||0)*Math.exp(-0.5*dx*dx-0.22*dy*dy);
      }}
      return d;
    }}
    // Horizontal grid lines: each mass produces its own local depression.
    ctx.lineWidth=1;
    for(let r=0;r<8;r++){{
      const y0=top+r*29;
      ctx.beginPath();
      for(let i=0;i<=54;i++){{
        const x=i*scene.width/54;
        const y=y0+field(x,y0);
        if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
      }}
      ctx.strokeStyle=`rgba(111,159,198,${{0.20-r*0.010}})`;
      ctx.stroke();
    }}
    // Perspective longitudinal grid lines.
    for(let c=0;c<20;c++){{
      ctx.beginPath();
      for(let i=0;i<=34;i++){{
        const y0=top+i*(bottom-top)/34;
        const perspective=(y0-top)/(bottom-top);
        const bx=(c+0.5)*scene.width/20;
        const x=scene.width/2+(bx-scene.width/2)*(0.66+0.34*perspective);
        const y=y0+field(x,y0);
        if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
      }}
      ctx.strokeStyle='rgba(111,159,198,.14)';
      ctx.stroke();
    }}
    // Subtle lip around each local well makes the two-well -> one-well transition readable.
    for(const q of bodies){{
      ctx.beginPath();
      ctx.ellipse(q.x,(q.planeY||top)+(q.depth||0)*0.58,(q.width||78)*0.62,13,0,0,Math.PI*2);
      ctx.strokeStyle='rgba(211,183,107,.16)';
      ctx.stroke();
    }}
    ctx.restore();
    if(stage==='Merger'||stage==='Ringdown'){{
      for(let k=0;k<4;k++){{
        const rr=48+((performance.now()/18+k*58)%225);
        ctx.beginPath();ctx.arc(cx,cy+58,rr,0,Math.PI*2);
        ctx.strokeStyle=`rgba(211,183,107,${{Math.max(0,.16-rr/1750)*(0.35+amp)}})`;ctx.stroke();
      }}
    }}
  }}
  function sphere(x,y,r,label,hot=false){{{{const g=ctx.createRadialGradient(x-r*.3,y-r*.4,2,x,y,r);g.addColorStop(0,hot?'#ffe9a8':'#9ab7cf');g.addColorStop(.22,hot?'#a66b2b':'#1d3b55');g.addColorStop(.72,'#05080d');g.addColorStop(1,'#000');ctx.fillStyle=g;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.strokeStyle='rgba(244,231,194,.28)';ctx.stroke();if(label){{{{ctx.fillStyle='#f4e7c2';ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillText(label,x,y-r-9)}}}}}}}}
  function drawScene(){{
    const W=scene.width,H=scene.height;ctx.clearRect(0,0,W,H);
    const bg=ctx.createLinearGradient(0,0,0,H);bg.addColorStop(0,'#071726');bg.addColorStop(1,'#02070d');ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
    for(let i=0;i<62;i++){{const x=(i*163.7)%W,y=(i*i*37.1)%205;ctx.fillStyle=`rgba(241,230,197,${{.11+(i%6)/40}})`;ctx.fillRect(x,y,1+(i%5===0),1+(i%5===0));}}
    const stage=stageAt(t), d=dynAt(t), cx=W/2, cy=214;
    let bodies=[];
    if(noEvent){{
      drawFabric(cx,cy,[],stage,0);
      ctx.fillStyle='#c7d2dc';ctx.font='700 23px system-ui';ctx.textAlign='center';ctx.fillText('No merger trigger',cx,190);
      ctx.font='14px system-ui';ctx.fillStyle='#8fa3b5';ctx.fillText('Off-source detector data continues in the synchronized traces below',cx,218);
    }}else if(stage==='Ringdown'||stage==='Post-event'){{
      const ringOsc=stage==='Ringdown'?Math.sin(performance.now()/58)*Math.min(6,9*d.amp):0;
      const depth=72+ringOsc;
      const q={{x:cx,planeY:258,depth:depth,width:96,height:92}};
      bodies=[q];drawFabric(cx,cy,bodies,stage,d.amp);
      const r=36+(stage==='Ringdown'?2.5*d.amp:0);
      const sy=q.planeY+q.depth*0.54-r*0.62;
      const wobble=stage==='Ringdown'?Math.sin(performance.now()/47)*Math.min(3.5,8*d.amp):0;
      sphere(cx+wobble,sy,r,D.masses.final?D.masses.final.toFixed(1)+' M☉ remnant':'remnant',stage==='Ringdown');
    }}else{{
      let sep=170;if(d.sep>0){{sep=Math.max(38,Math.min(205,32+0.48*d.sep));}}
      if(stage==='Merger'){{
        const P=D.phases.Merger;
        const q=Math.max(0,Math.min(1,(t-P[0])/(D.collisionTime-P[0]||.01)));
        sep*=1-q;
      }}
      const a=d.phase/2;const m1=D.masses.m1||1,m2=D.masses.m2||1,mt=m1+m2;
      const r1=sep*(m2/mt),r2=sep*(m1/mt);
      const x1=cx+r1*Math.cos(a),x2=cx-r2*Math.cos(a);
      const plane1=255+0.20*r1*Math.sin(a),plane2=255-0.20*r2*Math.sin(a);
      const rad1=24+7*m1/mt,rad2=24+7*m2/mt;
      const q1={{x:x1,planeY:plane1,depth:38+30*m1/mt,width:64+30*m1/mt,height:82}};
      const q2={{x:x2,planeY:plane2,depth:38+30*m2/mt,width:64+30*m2/mt,height:82}};

      // At the final contact, the two local wells continuously collapse into the remnant well.
      if(stage==='Merger' && sep<18){{
        const q={{x:cx,planeY:258,depth:70,width:94,height:90}};
        bodies=[q];drawFabric(cx,cy,bodies,stage,d.amp);
        const rr=34;
        sphere(cx,q.planeY+q.depth*0.54-rr*0.62,rr,'coalescence',true);
      }}else{{
        bodies=[q1,q2];drawFabric(cx,cy,bodies,stage,d.amp);
        const sy1=q1.planeY+q1.depth*0.54-rad1*0.62;
        const sy2=q2.planeY+q2.depth*0.54-rad2*0.62;
        sphere(x1,sy1,rad1,m1.toFixed(1)+' M☉',stage==='Merger');
        sphere(x2,sy2,rad2,m2.toFixed(1)+' M☉',stage==='Merger');
        ctx.strokeStyle='rgba(221,229,236,.12)';ctx.setLineDash([5,6]);ctx.beginPath();ctx.ellipse(cx,255,Math.max(20,sep/2),Math.max(7,sep*.10),0,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
      }}
    }}
    ctx.fillStyle='#f1e6c5';ctx.font='750 24px system-ui';ctx.textAlign='left';ctx.fillText(stage.toUpperCase(),24,42);
    ctx.fillStyle='#95a8ba';ctx.font='12px system-ui';
    ctx.fillText(stage==='Inspiral'?'Orbital rate rises • separation decreases • GW frequency increases':stage==='Merger'?'Two local curvature wells merge into one remnant well':stage==='Ringdown'?'Remnant distortion decays toward a settled state':'Detector-record timeline',24,64);
    document.getElementById('gh-stage-read').textContent=stage;
    document.getElementById('gh-freq').textContent=d.f>0?d.f.toFixed(1)+' Hz':'—';
    document.getElementById('gh-sep').textContent=(stage==='Inspiral'&&d.sep>0)?d.sep.toFixed(0)+' km':'—';
    document.querySelectorAll('#gh-merger .stage').forEach(e=>e.classList.toggle('on',e.dataset.stage===stage));
    document.getElementById('gh-badge').innerHTML=noEvent?'<b>Off-source control</b><br>No catalog merger in this interval':`<b>${{D.detector}} • ${{D.syncLabel}}</b><br>collision @ ${{D.collisionTime.toFixed(4)}} s`;
  }}
  function drawSeries(canvas,c,sets,cursor,threshold=null){{const W=canvas.width,H=canvas.height;c.clearRect(0,0,W,H);c.fillStyle='rgba(2,10,18,.72)';c.fillRect(0,0,W,H);c.strokeStyle='rgba(244,231,194,.08)';for(let i=1;i<4;i++){{c.beginPath();c.moveTo(0,i*H/4);c.lineTo(W,i*H/4);c.stroke();}}let ymin=Infinity,ymax=-Infinity;for(const s of sets)for(const v of s.y)if(Number.isFinite(v)){{ymin=Math.min(ymin,v);ymax=Math.max(ymax,v)}}if(!Number.isFinite(ymin)||ymax===ymin){{ymin=-1;ymax=1}}const X=v=>v/D.duration*W,Y=v=>H-12-(v-ymin)/(ymax-ymin)*(H-24);for(const s of sets){{c.strokeStyle=s.color;c.lineWidth=s.width||1;c.beginPath();s.t.forEach((xx,i)=>{{const px=X(xx),py=Y(s.y[i]);if(i===0)c.moveTo(px,py);else c.lineTo(px,py)}});c.stroke();}}if(threshold!=null&&threshold>=ymin&&threshold<=ymax){{c.strokeStyle='rgba(244,231,194,.45)';c.setLineDash([5,4]);c.beginPath();c.moveTo(0,Y(threshold));c.lineTo(W,Y(threshold));c.stroke();c.setLineDash([]);}}c.strokeStyle='#e8c66a';c.lineWidth=2;c.beginPath();c.moveTo(X(cursor),0);c.lineTo(X(cursor),H);c.stroke();}}
  function updateReadout(){{document.getElementById('gh-record').textContent='t = '+t.toFixed(3)+' s / '+D.duration.toFixed(1)+' s';const gps=D.gpsStart==null?null:D.gpsStart+t;document.getElementById('gh-gps').textContent=gps==null?'GPS unavailable':'GPS '+gps.toFixed(3);scrub.value=String(t);drawScene();drawSeries(wave,wctx,[{{t:D.realWave.t,y:D.realWave.y,color:'rgba(120,169,212,.65)'}},{{t:D.referenceWave.t,y:D.referenceWave.y,color:'#f4e7c2',width:1.5}}],t);drawSeries(snrC,sctx,[{{t:D.snr.t,y:D.snr.y,color:'#9dd9b5',width:1.4}}],t,D.threshold);}}
  function tick(now){{const dt=Math.min(.08,(now-last)/1000);last=now;if(playing){{if(mode==='full')t+=dt*Number(document.getElementById('gh-rate').value);else t+=dt/Number(document.getElementById('gh-slow').value);if(t>=eventEnd){{t=eventEnd;playing=false}}if(t>D.duration){{t=D.duration;playing=false}}updateReadout();}}requestAnimationFrame(tick)}}
  document.getElementById('gh-play').addEventListener('click',()=>{{mode='full';eventEnd=D.duration;playing=true;last=performance.now()}});document.getElementById('gh-pause').addEventListener('click',()=>playing=false);document.getElementById('gh-restart').addEventListener('click',()=>{{playing=false;t=0;updateReadout()}});
  document.getElementById('gh-jump').addEventListener('click',()=>{{if(noEvent)return;playing=false;t=Math.max(0,D.collisionTime-.12);updateReadout()}});
  document.getElementById('gh-replay').addEventListener('click',()=>{{if(noEvent)return;const P=D.phases;const start=Math.max(0,Math.min(P.Merger?P.Merger[0]-.10:D.collisionTime-.18,D.collisionTime-.18));eventEnd=Math.min(D.duration,Math.max(P.Ringdown?P.Ringdown[1]+.05:D.collisionTime+.10,D.collisionTime+.10));t=start;mode='event';playing=true;last=performance.now();updateReadout()}});
  scrub.addEventListener('input',e=>{{playing=false;t=Math.max(0,Math.min(D.duration,Number(e.target.value)));updateReadout()}});
  updateReadout();requestAnimationFrame(tick);
}})();
</script>
"""
