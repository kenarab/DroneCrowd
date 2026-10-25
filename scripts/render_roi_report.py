"""Show frozen visual references beside density outputs of both checkpoints."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sample_video import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    cases=[("den 1×","rois-density-x1",1),("den 2×","rois-density-x2",2),
           ("loc: densidad 1×","rois-localization-x1",1),("loc: densidad 2×","rois-localization-x2",2)]
    runs=[]
    for label,directory,scale in cases:
        path=args.root/directory/"evaluation.json"
        report=json.loads(path.read_text())
        if report["status"]!="complete" or len(report["rows"])!=8:
            raise ValueError("Expected eight completed controls")
        runs.append((label,path.parent,report,scale))
    controls_path=Path(runs[0][2]["controls"])
    digest=sha256(controls_path)
    if any(r["controls_sha256"]!=digest for _,_,r,_ in runs):
        raise ValueError("Runs do not share frozen controls")
    controls=json.loads(controls_path.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    groups={"personas":[1,5,6],"negativos":[3,4,7],"densos-sin-recuento":[0,2]}
    for name,indices in groups.items():
        figure,axes=plt.subplots(len(indices),5,figsize=(22,4.8*len(indices)),squeeze=False,layout="constrained")
        figure.suptitle("Controles visuales provisionales — verde: confirmado; naranja: dudoso\nMapas de densidad; precisión no certificada")
        for axes_row,index in zip(axes,indices):
            entry=controls["entries"][index]
            box=entry["roi_context_xyxy"]
            x,y,right,bottom=box
            rgb=Image.open(controls_path.parent/entry["rgb_file"]).crop(box)
            axes_row[0].imshow(rgb,interpolation="nearest")
            axes_row[0].set_title(entry["name"]+"\nReferencia: "+str(entry["reference_interval"]))
            for field,color in (("points_confirmed_xy","lime"),("points_uncertain_xy","orange")):
                for px,py in entry[field]:
                    axes_row[0].scatter(px,py,s=25,facecolors="none",edgecolors=color,linewidths=.7)
            maps=[]
            for label,directory,report,scale in runs:
                result=next(r for r in report["rows"] if r["index"]==index)
                raw=np.load(directory/result["density_file"],allow_pickle=False)/100
                region=raw[y*scale//2:bottom*scale//2,x*scale//2:right*scale//2]
                display=np.asarray(Image.fromarray(region).resize(((right-x)//2,(bottom-y)//2),Image.Resampling.BOX))*scale**2
                maps.append((label,result,display))
                rows.append({"name":entry["name"],"source_kind":entry["source_kind"],"reference_type":entry["reference_type"],
                    "reference_interval":entry["reference_interval"],"checkpoint_scale":label,"count":result["count"]})
            maximum=max(1e-6,max(float(np.quantile(display,.995)) for _,_,display in maps))
            for axis,(label,result,display) in zip(axes_row[1:],maps):
                axis.imshow(display,cmap="magma",vmin=0,vmax=maximum)
                axis.set_title(f"{label}\nSalida: {result['count']:.2f}")
            for axis in axes_row:
                axis.axis("off")
        figure.savefig(args.output_dir/(name+".png"),dpi=120)
        plt.close(figure)
    with (args.output_dir/"counts.csv").open("x",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir/"provenance.json").write_text(json.dumps({"controls_sha256":digest,
        "script_sha256":sha256(__file__),"map_display":"BOX area preserving to native ROI stride2; common color scale within each row; negatives black",
        "annotation_warning":"Assistant visual candidates, not human validated GT; Gaussian mass crosses ROI boundaries"},indent=2)+"\n")
    print(json.dumps({"controls":8,"comparisons":len(rows),"output":str(args.output_dir)}))


if __name__=="__main__":
    main()
