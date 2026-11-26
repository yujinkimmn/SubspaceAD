"""Standalone HTML report builder. Standard library only; no model or image loading."""
import argparse
import csv
import html
import json
from pathlib import Path
from urllib.parse import quote

METRICS = ("recall","fpr","worst_equipment_fpr","underkill","dice","iou","auroc",
           "fp_regions_per_normal","tp","fp","tn","fn")
SIZES = ("1px","2-4px","5+px")
CSS = """
*{box-sizing:border-box}body{font:15px/1.5 system-ui,sans-serif;background:#f6f8fb;color:#172332;margin:0}
main{max-width:1500px;margin:auto;padding:28px}h1{margin:0;font-size:30px}h2{margin:0 0 12px}
h3{margin:20px 0 8px}a{color:#165eae}p{margin:8px 0}nav{display:flex;flex-wrap:wrap;gap:10px}
nav a{padding:6px 12px;border:1px solid #c9d5e2;border-radius:8px;background:white;text-decoration:none}
section{background:white;border:1px solid #d9e1e9;border-radius:12px;padding:20px;margin:18px 0}
.note{color:#526375;font-size:13px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 10px;border-bottom:1px solid #e0e7ef;text-align:right;white-space:nowrap}
th{background:#edf2f7}td:first-child,th:first-child{text-align:left}.selected{background:#eaf3fe}
.error{color:#a33b29;white-space:normal;text-align:left}.muted{color:#718096}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
.chart{border:1px solid #e0e7ef;border-radius:8px;padding:12px}.chart h4{margin:0 0 10px}
.barline{display:grid;grid-template-columns:90px minmax(40px,1fr) 64px;gap:8px;align-items:center;margin:5px 0;font-size:12px}
.track{height:12px;background:#e9eef4;border-radius:3px;overflow:hidden}.fill{height:100%;background:#3776b0}
.fpr .fill{background:#bb7040}.tiny .fill{background:#507d69}.num{font-variant-numeric:tabular-nums;text-align:right}
details{margin-top:14px}summary{cursor:pointer;font-weight:600}footer{font-size:12px;color:#536477}
@media print{body{background:white}main{padding:0}section{break-inside:avoid}nav{display:none}.scroll{overflow:visible}table{font-size:9px}td,th{white-space:normal;padding:3px}}
"""

def esc(v):
    return html.escape("" if v is None else str(v), quote=True)

def percent(v):
    return "—" if v is None else f"{v*100:.2f}%"

def numeric(v):
    return "—" if v is None else f"{v:.4f}" if isinstance(v,float) else str(v)

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def region(m,k,field="recall"):
    return m.get("regions",{}).get(k,{}).get(field)

def metric_cells(m):
    return [percent(m.get("recall")),percent(m.get("fpr")),percent(m.get("worst_equipment_fpr")),
            percent(m.get("underkill")),numeric(m.get("dice")),numeric(m.get("iou")),
            numeric(m.get("auroc")),numeric(m.get("fp_regions_per_normal")),
            *[percent(region(m,k)) for k in SIZES],
            *[numeric(m.get(k)) for k in ("tp","fp","tn","fn")]]

HEAD = ["Recall","FPR","Worst FPR","Underkill","Dice","IoU","AUROC","FP regions / normal",
        "1px recall","2–4px recall","5+px recall","TP","FP","TN","FN"]

def table(headers,rows):
    parts=["<div class='scroll'><table><thead><tr>"]
    parts += [f"<th>{esc(h)}</th>" for h in headers]
    parts.append("</tr></thead><tbody>")
    for cells,selected in rows:
        parts.append("<tr class='selected'>" if selected else "<tr>")
        parts += [f"<td>{esc(v)}</td>" for v in cells]
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)

def chart(title,results,key,kind=""):
    lines=[f"<div class='chart {kind}'><h4>{esc(title)}</h4>"]
    for r in results:
        m=r.get("test",{})
        v=key(m) if r.get("status")=="ok" else None
        label=r["experiment"].split("_")[0]
        tooltip=esc(r["experiment"])
        width=0 if v is None else min(100,max(0,float(v)*100))
        lines.append(f"<div class='barline' title='{tooltip}'><span>{esc(label)}</span>"
                     f"<div class='track'><div class='fill' style='width:{width:.4f}%'></div></div>"
                     f"<span class='num'>{percent(v)}</span></div>")
    lines.append("<p class='note'>0–100%; — = unavailable</p></div>")
    return "".join(lines)

def chart_group(results):
    return "<div class='charts'>"+"".join([
        chart("Test image FPR · lower is better",results,lambda m:m.get("fpr"),"fpr"),
        chart("Test image recall · higher is better",results,lambda m:m.get("recall")),
        chart("Test 1px region recall",results,lambda m:region(m,"1px"),"tiny"),
        chart("Test 2–4px region recall",results,lambda m:region(m,"2-4px"),"tiny"),
        chart("Test Dice",results,lambda m:m.get("dice"))
    ])+"</div>"

def tradeoff_chart(results,split,key,title):
    colors=("#2368ae","#c35d36","#27866c","#8861a9","#b08b22","#42616a")
    parts=[f"<div class='chart'><h4>{esc(title)}</h4>",
           "<svg viewBox='0 0 320 245' role='img' aria-label='FPR versus recall' style='width:100%;max-height:270px'>"]
    for i in range(5):
        x=42+60*i; y=202-45*i
        parts.append(f"<path d='M{x} 22V202 M42 {y}H282' stroke='#e0e7ef' fill='none'/>")
        parts.append(f"<text x='{x}' y='218' font-size='10' text-anchor='middle'>{i*25}</text>"
                     f"<text x='36' y='{y+3}' font-size='10' text-anchor='end'>{i*25}</text>")
    parts.append("<path d='M42 22V202H282' stroke='#536477' fill='none'/>"
                 "<text x='162' y='236' font-size='11' text-anchor='middle'>FPR (%)</text>"
                 "<text transform='translate(12 112) rotate(-90)' font-size='11' text-anchor='middle'>Recall (%)</text>")
    legend=[]
    for i,r in enumerate(results):
        curve=r.get("operating_points",{}).get("curve",[])
        xy=[(p[split].get("fpr"),p[split].get(key)) for p in curve]
        xy=[(x,y) for x,y in xy if x is not None and y is not None]
        if not xy:
            continue
        color=colors[i%len(colors)]
        coords=" ".join(f"{42+240*x:.2f},{202-180*y:.2f}" for x,y in xy)
        parts.append(f"<polyline points='{coords}' stroke='{color}' stroke-width='1.8' fill='none'/>")
        legend.append(f"<span style='color:{color}'>&#9632; {esc(r['experiment'].split('_')[0])}</span>")
    parts.append("</svg><p class='note'>"+" &nbsp; ".join(legend)+"</p></div>")
    return "".join(parts)

def tradeoff_section(results):
    if not any(r.get("operating_points") for r in results):
        return ""
    body=["<h3>Recall–FPR trade-off (validation-derived thresholds)</h3>",
          "<p class='note'>All thresholds are derived from validation scores only and then applied unchanged to test. "
          "Test curves are diagnostic, not threshold-selection tools. Missing region groups remain unavailable.</p>",
          "<div class='charts'>"]
    for split in ("val","test"):
        for key,label in (("recall","image recall"),("region_1px","1px region recall"),("region_2_4px","2–4px region recall")):
            body.append(tradeoff_chart(results,split,key,f"{split.upper()} {label}"))
    body.append("</div>")
    rows=[]; learning=[]
    for r in results:
        for p in r.get("operating_points",{}).get("points",[]):
            if not (p["kind"]=="primary" or p["target"] in (.05,.1,.9,.95)):
                continue
            for split in ("val","test"):
                m=p[split]
                rows.append(([r["experiment"],p["kind"],percent(p["target"]),split,numeric(p["threshold"]),
                              percent(m["fpr"]),percent(m["recall"]),percent(m["region_1px"]),percent(m["region_2_4px"])],False))
        b=r.get("backbone")
        if b:
            learning.append(([r["experiment"],b["epochs"],f"{b['weight_max_abs_change']:.6g}",b["mixstyle_calls"],
                              json.dumps(b["final_proxy"])],False))
    body.append("<details open><summary>Operating points</summary>"+table(["Experiment","Validation target","Target","Split","Threshold","FPR","Image recall","1px recall","2–4px recall"],rows)+"</details>")
    if learning:
        body.append("<details open><summary>Backbone training diagnostics (proxy, not PaDiM performance)</summary>"+
                    table(["Experiment","Epochs","Max weight change","MixStyle calls","Final validation proxy"],learning)+"</details>")
    return "".join(body)

def csv_row(proto,eq,r,split="",scope="",equipment="",m=None,best=None):
    m=m or {}
    row=dict(protocol=proto,heldout_equipment=eq,experiment=r.get("experiment"),
             status=r.get("status"),selected_by_validation=r.get("experiment")==best and best is not None,
             split=split,scope=scope,equipment=equipment,threshold=r.get("threshold"),
             seconds=r.get("seconds"),error=r.get("error"))
    row.update({k:m.get(k) for k in METRICS})
    for k in SIZES:
        row[k+"_recall"]=region(m,k)
        row[k+"_count"]=region(m,k,"count")
        row[k+"_hit"]=region(m,k,"hit")
    return row

CSV_FIELDS=["protocol","heldout_equipment","experiment","status","selected_by_validation",
            "split","scope","equipment","threshold","seconds","error",*METRICS,
            *[k+"_"+f for k in SIZES for f in ("recall","count","hit")]]

def build_report(root,index=None):
    root=Path(root)
    if index is None:
        index=load_json(root/"index.json")
    if not isinstance(index,list):
        raise ValueError("index.json must contain a list of protocols")
    rows=[]; sections=[]; overview=[]; holdout=[]; nav=[]
    for i,p in enumerate(index):
        name=p["protocol"]
        if name not in ("pooled",) and not (name.startswith("holdout_") and name[8:].isdigit()):
            raise ValueError(f"Unexpected protocol name: {name}")
        eq=p.get("equipment")
        state=p.get("status","unknown")
        directory=root/name
        result_path=directory/"results.json"
        label=name if eq is None else f"{name} · equipment {eq}"
        anchor=f"protocol-{i}"
        nav.append(f"<a href='#{anchor}'>{esc(label)}</a>")
        section=[f"<section id='{anchor}'><h2>{esc(label)}</h2><p>Status: {esc(state)}</p>"]
        if p.get("error"):
            section.append(f"<p class='error'>{esc(p['error'])}</p>")
        if state in ("skipped","pending") or not result_path.is_file():
            if state not in ("skipped","pending"):
                section.append("<p class='error'>results.json missing; metrics unavailable.</p>")
            overview.append(([label,state,"—","—","—","—","—","—"],False))
            rows.append(csv_row(name,eq,dict(status=state if state in ("skipped","pending") else "missing",
                                             error=p.get("error") or (None if state in ("skipped","pending") else "results.json missing"))))
            sections.append("".join(section)+"</section>")
            continue
        results=load_json(result_path)
        selected_path=directory/"selected.json"
        best=load_json(selected_path).get("selected_by_validation") if selected_path.is_file() else None
        chosen=next((r for r in results if r.get("experiment")==best and r.get("status")=="ok"),None)
        m=chosen["test"] if chosen else {}
        overview.append(([label,state,best or "—",percent(m.get("fpr")),percent(m.get("recall")),
                          numeric(m.get("dice")),percent(region(m,"1px")),percent(region(m,"2-4px"))],False))
        config_path=directory/"config.json"
        config=load_json(config_path) if config_path.is_file() else {}
        section.append(f"<p>Validation-selected model: <strong>{esc(best or 'None recorded')}</strong>. "
                       f"Validation targets: image recall {percent(config.get('recall'))}, "
                       f"small-region recall {percent(config.get('region_recall'))}.</p>")
        if (directory/"report.html").is_file():
            section.append(f"<p><a href='{quote(name)}/report.html'>Optional original detailed report</a></p>")
        section.append(chart_group(results))
        section.append(tradeoff_section(results))
        main_rows=[]; detail_rows=[]; region_rows=[]
        for r in results:
            exp=r["experiment"]; status=r["status"]; is_best=exp==best
            display=exp+(" ★" if is_best else "")
            if status!="ok":
                main_rows.append(([display,status,"—","—",*["—"]*len(HEAD),r.get("error","")],is_best))
                rows.append(csv_row(name,eq,r,best=best))
                continue
            for split in ("val","test"):
                m=r[split]
                main_rows.append(([display,status,split,numeric(r.get("threshold")),*metric_cells(m),""],is_best))
                region_rows.append(([display,split,*[f"{numeric(region(m,k,'hit'))}/{numeric(region(m,k,'count'))}" for k in SIZES]],is_best))
                rows.append(csv_row(name,eq,r,split,"overall","",m,best))
                for equipment,em in sorted(m.get("by_equipment",{}).items()):
                    detail_rows.append(([display,split,equipment,*metric_cells(em)],is_best))
                    rows.append(csv_row(name,eq,r,split,"equipment",equipment,em,best))
            if name.startswith("holdout_"):
                m=r["test"]
                holdout.append(([eq,display,percent(m.get("fpr")),percent(m.get("recall")),
                                 numeric(m.get("dice")),percent(region(m,"1px")),percent(region(m,"2-4px")),
                                 numeric(m.get("fp")),numeric(m.get("fn"))],is_best))
        section.append("<h3>All experiments · validation & test</h3>")
        section.append(table(["Experiment","Status","Split","Threshold",*HEAD,"Error"],main_rows))
        section.append("<details open><summary>Region hit / count (native pixel area)</summary>"+
                       table(["Experiment","Split",*SIZES],region_rows)+"</details>")
        section.append("<details open><summary>Per-equipment metrics</summary>"+
                       table(["Experiment","Split","Equipment",*HEAD],detail_rows)+"</details>")
        sections.append("".join(section)+"</section>")
    content=["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>CSAM consolidated report</title><style>",CSS,"</style></head><body><main>",
             "<h1>CSAM consolidated report</h1>",
             "<p class='note'>Self-contained metrics and charts. No external scripts, images or network requests. "
             "Optional detailed report links need the original folder structure.</p>",
             "<p>★ = selected by validation within each protocol. Test is report-only; do not select models using test. "
             "Pooled and holdout results are different evaluation conditions, not interchangeable.</p>",
             "<nav>","".join(nav),"</nav><section><h2>Protocol overview · validation-selected models</h2>",
             table(["Protocol / equipment","Status","Selected model","Test FPR","Test recall","Test Dice",
                    "Test 1px recall","Test 2–4px recall"],overview),"</section>"]
    if holdout:
        content += ["<section><h2>Holdout comparison · all successful experiments</h2>",
                    table(["Held-out equipment","Experiment","Test FPR","Test recall","Dice","1px recall",
                           "2–4px recall","FP","FN"],holdout),"</section>"]
    content += sections
    content += ["<footer>FPR = FP / (FP + TN); image recall = TP / (TP + FN). Region recall = fraction of "
                "ground-truth connected components touched by at least one predicted pixel. Size is pixel area, not diameter. "
                "Missing values are not zero. Worst FPR is unavailable at individual-equipment scope. "
                "Thresholds are calibrated separately per protocol and model; scores are not directly comparable. "
                "No source images or per-image paths are embedded.</footer></main></body></html>"]
    with (root/"summary.csv.tmp").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerows(rows)
    (root/"summary.csv.tmp").replace(root/"summary.csv")
    (root/"index.html.tmp").write_text("".join(content),encoding="utf-8")
    (root/"index.html.tmp").replace(root/"index.html")
    return root/"index.html"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path,help="Existing run folder containing index.json and protocol results.json")
    args=p.parse_args()
    print(build_report(args.root))

if __name__=="__main__":
    main()
