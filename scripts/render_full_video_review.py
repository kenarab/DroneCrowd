"""CPU-only local review videos and combined CSV from verified full campaigns."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile

import night_video_campaign as night

ROOT, TEMP = night.ROOT, night.TEMP
DEST = ROOT/"outputs/campana-nocturna-20261026"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def read_counts(source):
    path = night.base(source)/"counts.csv"
    proof = json.loads((night.base(source)/"counts-provenance.json").read_text())
    sha = night.bounded.sha256_file(night.base(source)/"plan.json")
    plan = night.load_plan(source, sha)
    if (proof.get("source") != source or proof.get("plan_sha256") != sha
            or proof.get("counts_sha256") != night.bounded.sha256_file(path)
            or len(proof.get("reports", [])) != len(plan["jobs"])
            or proof["reports"] != [{"path": j["output_dir"]+"/report.json",
                "sha256": night.bounded.sha256_file(ROOT/j["output_dir"]/"report.json")} for j in plan["jobs"]]
            or path.read_bytes() != night.counts_payload(night.prediction_rows(source, plan, verify_outputs=False))):
        raise ValueError("CSV or report provenance differs from verified campaign")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != night.SOURCES[source]["count"] or [int(r["source_ordinal"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Complete ordered CSV required")
    return rows


def label(row):
    # Display rounded estimates; the CSV retains full precision.
    return " | ".join(f"{name} {float(row[key]):.0f}" for key, name in (
        ("sr", "SR"), ("clip-ebc", "CLIP-EBC"), ("mpcount", "MPCount")))


def render(source):
    rows = read_counts(source)
    manifest = night.base(source)/"plan.json"
    plan = night.load_plan(source, night.bounded.sha256_file(manifest))
    if any(not night.completed(source, ROOT/j["output_dir"], j, ROOT/j["input_manifest"], plan) for j in plan["jobs"]):
        raise ValueError("Maps must verify before rendering")
    output = DEST/f"{source}-conteos.mp4"
    if output.exists():
        raise FileExistsError("Keep existing review video; do not overwrite")
    DEST.mkdir(exist_ok=True)
    if not FONT.is_file():
        raise FileNotFoundError("Pinned local review font missing")
    first = int(rows[0]["pts"])*Fraction(rows[0]["time_base"])
    with tempfile.TemporaryDirectory(dir=TEMP, prefix="dronecrowd-overlay-") as tmp:
        script = Path(tmp)/"counts.commands"
        lines = []
        for row in rows:
            time = int(row["pts"])*Fraction(row["time_base"])-first
            # Apply just before the corresponding frame to avoid decimal
            # rounding at a rational PTS boundary. No video frame is dropped.
            at = max(0.0, float(time)-0.000001)
            lines.append(f"{at:.9f} drawtext@counts reinit 'text={label(row)}';")
        script.write_text("\n".join(lines)+"\n")
        s = night.SOURCES[source]
        size = 40 if s["width"] > 1000 else 18
        filters = ["setpts=PTS-STARTPTS"]
        if s["crop"]:
            filters.append(f"crop={s['width']}:{s['height']}:0:0")
        filters += [f"sendcmd=f={script}", f"drawtext@counts=fontfile={FONT}:text='{label(rows[0])}':"
            f"fontsize={size}:x=20:y=20:fontcolor=white:box=1:boxcolor=black@0.75"]
        staging = Path(tmp)/"review.mp4"
        subprocess.run(["/usr/bin/ffmpeg", "-nostdin", "-v", "error", "-threads", "1",
            "-filter_threads", "1", "-i", str(night.source_path(source)), "-map", "0:v:0", "-an", "-sn",
            "-vf", ",".join(filters), "-fps_mode", "passthrough", "-c:v", "libx264",
            "-enc_time_base:v", rows[0]["time_base"],
            "-video_track_timescale", str(Fraction(rows[0]["time_base"]).denominator),
            "-threads", "1", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(staging)], check=True, timeout=580)
        metadata = json.loads(subprocess.check_output(["/usr/bin/ffprobe", "-v", "error",
            "-select_streams", "v:0", "-show_packets", "-show_entries", "stream=width,height,nb_frames,time_base:packet=pts",
            "-of", "json", str(staging)], timeout=15))
        probe = metadata["streams"][0]
        if (probe["width"], probe["height"], int(probe["nb_frames"])) != (s["width"], s["height"], s["count"]):
            raise ValueError("Review encode changed geometry or frame count")
        expected_pts = [int(r["pts"])*Fraction(r["time_base"])-first for r in rows]
        actual_pts = sorted(int(p["pts"])*Fraction(probe["time_base"]) for p in metadata["packets"])
        if actual_pts != expected_pts:
            raise ValueError("Review encode changed original frame timestamps")
        # Temporaries and output are on the same filesystem; keep publication atomic.
        staging.rename(output)
    night.legacy.write_new(DEST/f"{source}-review.json", {"source": source,
        "video_sha256": night.bounded.sha256_file(output), "frame_count": len(rows),
        "plan_sha256": night.bounded.sha256_file(manifest),
        "counts_sha256": night.bounded.sha256_file(night.base(source)/"counts.csv"),
        "display": "Rounded model estimates per original frame; not validated attendance"})


def combine():
    rows = [{**r, "source": source} for source in night.SOURCES for r in read_counts(source)]
    if len(rows) != 2592:
        raise ValueError("Expected2592 source frames")
    DEST.mkdir(exist_ok=True)
    with (DEST/"counts-all.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "source_ordinal", "pts", "time_base",
            "pts_seconds", "rgb_sha256", *night.MODELS])
        writer.writeheader()
        writer.writerows(rows)
    points = {source: [{"t": float(r["pts_seconds"]), **{m: float(r[m]) for m in night.MODELS}}
                       for r in rows if r["source"] == source] for source in night.SOURCES}
    # Offline exploratory plot: no requests, uploads or third-party libraries.
    html = '''<!doctype html><meta charset="utf-8"><title>DroneCrowd: curvas por cuadro</title>
<style>body{font:17px sans-serif;max-width:1200px;margin:30px auto;padding:12px;background:#fafafa}
canvas{background:white;width:100%;border:1px solid #aaa}label{margin-right:20px}#value{min-height:26px}</style>
<h1>DroneCrowd: estimaciones por cuadro</h1><p>Conteos en cada encuadre e instante.
No sumar cuadros o clips como asistentes distintos. Sin referencia manual exhaustiva, estas curvas no acreditan precisión.</p>
<label>Video <select id="source"><option>0269</option><option>0265</option><option>trim</option></select></label>
<span style="color:#1664b5">SR</span> · <span style="color:#a54518">CLIP-EBC</span> ·
<span style="color:#297742">MPCount</span><p id="value">Mové el puntero por la curva para consultar el cuadro.</p>
<canvas id="plot" width="1200" height="500"></canvas>
<p>Eje horizontal: segundos del video. Eje vertical: personas estimadas. El CSV conserva precisión y procedencia.</p>
<script>
const data=__DATA__, canvas=document.querySelector('#plot'),c=canvas.getContext('2d'),
sel=document.querySelector('#source'),keys=['sr','clip-ebc','mpcount'],colors=['#1664b5','#a54518','#297742'];
let rows;
function draw(){rows=data[sel.value];let lo=rows[0].t,hi=rows[rows.length-1].t,
y=Math.max(1,...rows.flatMap(r=>keys.map(k=>r[k])))*1.08;
c.clearRect(0,0,1200,500);c.font='16px sans-serif';c.strokeStyle='#ddd';
for(let i=0;i<=5;i++){let v=y*i/5,py=450-400*i/5;c.beginPath();c.moveTo(70,py);c.lineTo(1170,py);c.stroke();c.fillStyle='#222';c.fillText(v.toFixed(0),3,py+5);}
for(let i=0;i<=5;i++)c.fillText((lo+(hi-lo)*i/5).toFixed(2),65+1100*i/5,480);
keys.forEach((k,j)=>{c.strokeStyle=colors[j];c.lineWidth=2;c.beginPath();rows.forEach((r,i)=>{
let x=70+1100*(r.t-lo)/(hi-lo),py=450-400*r[k]/y;i?c.lineTo(x,py):c.moveTo(x,py)});c.stroke()});}
sel.onchange=draw;canvas.onmousemove=e=>{let x=(e.clientX-canvas.getBoundingClientRect().left)/canvas.getBoundingClientRect().width*1200,
target=rows[0].t+Math.max(0,Math.min(1,(x-70)/1100))*(rows[rows.length-1].t-rows[0].t),
idx=rows.reduce((best,r,i)=>Math.abs(r.t-target)<Math.abs(rows[best].t-target)?i:best,0),r=rows[idx];
document.querySelector('#value').textContent='Cuadro '+idx+' · '+r.t.toFixed(4)+' s · '+keys.map(k=>k+' '+r[k].toFixed(2)).join(' · ')};draw();
</script>'''.replace('__DATA__', json.dumps(points, allow_nan=False))
    with (DEST/"curvas.html").open("x") as handle:
        handle.write(html)


def main():
    night.bounded.set_runtime_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["render", "combine"])
    parser.add_argument("--source", choices=night.SOURCES)
    args = parser.parse_args()
    if args.action == "combine":
        combine()
    elif args.source:
        render(args.source)
    else:
        parser.error("--source required")


if __name__ == "__main__":
    main()
