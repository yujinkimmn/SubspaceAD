"""Standalone HTML report builder. Standard library only; no model or image loading."""
import argparse
import csv
import html
import json
from pathlib import Path
from urllib.parse import quote

METRICS = ("recall","fpr","worst_equipment_fpr","underkill","dice","iou","auroc","image_auroc","pixel_auroc","image_f1","pixel_aupro",
           "fp_regions_per_normal","pixel_fpr","bg_fp","bg_pixels","tp","fp","tn","fn")
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
.guide-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:12px 0}
.guide-card{background:#f3f7fb;border:1px solid #d9e1e9;border-radius:8px;padding:12px}
.guide-card h3{margin:0 0 6px;font-size:16px}.guide-card p{margin:0}
.callout{border-left:4px solid #3776b0;background:#edf5fc;padding:10px 14px;margin:12px 0}
.guide table td:not(:first-child){white-space:normal;text-align:left}
.guide table th:not(:first-child){text-align:left}
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
            numeric(m.get("image_auroc",m.get("auroc"))),numeric(m.get("pixel_auroc")),
            numeric(m.get("image_f1")),numeric(m.get("pixel_aupro")),numeric(m.get("fp_regions_per_normal")),percent(m.get("pixel_fpr")),
            *[percent(region(m,k)) for k in SIZES],
            *[numeric(m.get(k)) for k in ("tp","fp","tn","fn")]]

HEAD = ["Recall","FPR","Worst FPR","Underkill","Dice","IoU","Image AUROC","Pixel AUROC","Image F1","Pixel AUPRO@0.30","FP regions / normal","Background pixel FPR",
        "1px recall","2–4px recall","5+px recall","TP","FP","TN","FN"]

PADIM_EXPERIMENTS = (
    ("p0_pretrained", "기준 모델", "공식 pretrained backbone을 그대로 사용하고 정상 feature의 Gaussian만 학습한다."),
    ("p1_classification", "Image classification", "정상/불량 image label로 backbone을 조정한다."),
    ("p2_classification_mixstyle", "P1 + MixStyle", "다른 설비의 feature 통계를 섞어 특정 설비에 대한 의존성을 줄이는지 확인한다."),
    ("p3_consistency", "Normal consistency", "정상 이미지의 밝기·대비 등이 변해도 원래 teacher feature에 가깝도록 학습한다."),
    ("p4_consistency_mixstyle", "P3 + MixStyle", "정상 consistency 학습에 cross-equipment MixStyle을 추가한다."),
    ("p5_pixel_mixstyle", "P2 + Pixel supervision", "실제 void mask까지 활용해 작은 불량의 위치 정보를 보존하도록 학습한다."),
    ("p5_pixel_no_mixstyle", "P1 + Pixel supervision (선택 실행)", "MixStyle 없이 pixel loss의 효과를 측정한다."),
    ("p0_direct_native", "P0 + native map 직접 사용 (선택 실행)", "Image score를 고정하고 pixel map 복원의 영향을 측정한다."),
)

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

def reading_guide():
    return (
        "<section id='read-guide' class='guide'><h2>처음 보는 분을 위한 결과 읽는 법</h2>"
        "<p>목표는 정상 배경을 불량으로 오인하는 <strong>overkill</strong>을 줄이면서 "
        "실제 void, 특히 1px 또는 2–4px 크기의 작은 불량을 놓치지 않는 것이다.</p>"
        "<div class='guide-grid'>"
        "<div class='guide-card'><h3>1. 평가 조건 확인</h3><p>Pooled는 설비를 함께 사용한다. "
        "Holdout은 특정 설비를 train/val에서 제외하고 해당 설비의 test만 평가한다. 서로 다른 조건이므로 섞어 비교하지 않는다.</p></div>"
        "<div class='guide-card'><h3>2. 선정 모델 확인</h3><p>★는 해당 protocol의 validation에서 선택된 모델이다. "
        "Test는 이미 정한 기준으로 독립 평가하며, test 점수가 가장 높은 모델을 뜻하지 않는다.</p></div>"
        "<div class='guide-card'><h3>3. 불량 검출과 과검출 함께 확인</h3><p>FPR은 낮을수록, image recall과 "
        "1px/2–4px region recall은 높을수록 좋다. 하나만 좋아서는 충분하지 않다.</p></div>"
        "<div class='guide-card'><h3>4. 설비별 편차 확인</h3><p>전체 FPR이 낮더라도 Worst FPR이 크면 일부 설비에서 "
        "overkill이 집중될 수 있다. 설비별 상세표와 holdout 성능까지 확인한다.</p></div></div>"
        "<div class='callout'><strong>예시 (설명용 가상 수치):</strong> 정상 100장 중 5장을 잘못 검출하면 FPR 5%, "
        "실제 불량 20장 중 19장을 검출하면 image recall 95%이다. "
        "그러나 실제 1px 불량 영역 10개 중 6개만 맞추었다면 1px region recall은 60%이다.</div>"
        "<p class='note'>새 PaDiM 실험에서는 validation의 image recall, 작은 영역 recall, "
        "배경 pixel FPR 조건을 모두 충족한 모델 중 설비별 FPR 최댓값, 전체 FPR, Dice 순으로 선택한다. "
        "ok = 완료, partial = 일부 실험 미완료, skipped = 실행 조건 미충족, failed = 오류, — = 측정 불가/미기록(0이 아님).</p></section>"
    )

def metric_guide():
    definitions = [
        ("Image Recall ↑", "실제로 불량인 이미지 중 검출한 비율 (TP / (TP + FN))."),
        ("Image FPR ↓ / Underkill ↓", "정상 이미지 중 과검출한 비율 (FP / (FP + TN)) / 실제 불량 이미지 중 놓친 비율 (1 − Recall)."),
        ("Worst Equipment FPR ↓", "설비별 image FPR 중 최대값. 평균에서 가려지는 취약 설비 확인용."),
        ("Image AUROC ↑", "정상/불량 이미지의 score 순위 분리 정도. 1에 가까울수록 좋지만 특정 threshold의 FPR 또는 recall을 보장하지 않는다."),
        ("Pixel AUROC ↑", "정상/불량 pixel의 anomaly score 순위 분리 정도. 작은 불량의 실제 위치를 놓쳐도 높게 나올 수 있다."),
        ("Image F1 ↑", "Precision(불량이라고 판정한 이미지 중 실제 불량 비율)과 recall의 조화평균. 1에 가까울수록 좋으며 최종 image 판정 기준으로 계산한다."),
        ("Pixel AUPRO@0.30 ↑", "Pixel FPR 0~0.30 범위에서 각 정답 연결 영역을 얼마나 덮는지 평가한다. 크기가 다른 영역에 동등한 weight를 준다."),
        ("1px / 2–4px / 5+px region recall ↑", "실제 불량 연결 영역(8-neighbor) 중 예측 mask와 한 pixel 이상 겹친 비율. 크기는 원본 pixel 면적이다."),
        ("Dice / IoU ↑", "예측 mask와 정답 mask가 얼마나 겹치는지 나타낸다. 1에 가까울수록 좋다."),
        ("FP regions / normal ↓", "정상 이미지 한 장당 잘못 검출된 연결 영역 개수."),
        ("TP / FP / TN / FN", "불량을 맞춤 / 정상인데 불량으로 판정 / 정상을 맞춤 / 불량인데 정상으로 판정한 이미지 수."),
    ]
    return ("<section id='metric-guide' class='guide'><h2>지표 사전: 어떤 숫자를 봐야 할까?</h2>"
            "<p>Image 지표는 이미지 전체의 정상/불량 판정을, pixel 및 region 지표는 불량의 위치를 평가한다. "
            "새 결과의 validation/test AUROC와 test AUPRO는 raw score로 계산한다. 기존 저장 결과에는 당시 산식이 적용되어 있다. "
            "Image와 pixel은 각각 독립적인 threshold를 사용한다. ↑ 높을수록 좋음, ↓ 낮을수록 좋음.</p>"
            +table(["지표","의미"],[(list(r),False) for r in definitions])+"</section>")

def padim_guide():
    return (
        "<section id='padim-guide' class='guide'><h2>P0–P5: 무엇이 다른가?</h2>"
        "<p>모든 실험은 같은 ResNet18 초기 가중치와 데이터 분할을 사용한다. "
        "최종 검사는 공통 PaDiM 방식이다: 이미지 → backbone feature 추출 → 정상 데이터의 위치별 통계와 비교 "
        "→ anomaly map → image/pixel 판정. 차이는 정상 통계를 만들기 <em>전에</em> backbone을 어떻게 학습하느냐이다.</p>"
        +table(["Experiment","학습 방법","실험 질문"],[(list(r),False) for r in PADIM_EXPERIMENTS])
        +"<p><strong>MixStyle</strong>은 학습 중 서로 다른 설비 feature의 채널별 평균·표준편차를 섞는다. "
        "P1/P2와 P3/P4를 비교하면 MixStyle의 추가 효과를, P2/P5를 비교하면 pixel mask 학습의 추가 효과를 확인할 수 있다. "
        "학습 완료 후에는 backbone을 고정하고 head와 MixStyle을 사용하지 않으며, 확인된 정상 train 이미지로 Gaussian을 다시 fit한다.</p>"
        "<p class='note'>P0–P5 비교는 학습 방법의 차이를 검증하는 실험이며 MixStyle이 실제로 overkill을 감소시킨다는 의미는 아니다.</p></section>"
    )

def padim_comparison(results):
    """Human-readable matched validation comparisons; never select from test."""
    by_name={r["experiment"]:r for r in results}
    if not any(name in by_name for name,_,_ in PADIM_EXPERIMENTS):
        return ""

    def delta(before,after,get):
        x,y=get(before),get(after)
        return "—" if x is None or y is None else f"{(y-x)*100:+.2f} pp"

    pairs=(
        ("p0_pretrained","p1_classification","Image label 학습 효과"),
        ("p1_classification","p2_classification_mixstyle","Classification에 MixStyle 추가"),
        ("p0_pretrained","p3_consistency","Normal consistency 효과"),
        ("p3_consistency","p4_consistency_mixstyle","Consistency에 MixStyle 추가"),
        ("p2_classification_mixstyle","p5_pixel_mixstyle","MixStyle 사용 시 pixel supervision 효과"),
        ("p1_classification","p5_pixel_no_mixstyle","MixStyle 없이 pixel supervision 효과"),
        ("p5_pixel_no_mixstyle","p5_pixel_mixstyle","Pixel supervision 사용 시 MixStyle 효과"),
        ("p0_pretrained","p0_direct_native","Pixel map 복원 효과"),
    )
    rows=[]
    labels = {"p5_pixel_no_mixstyle": "P5 (no MixStyle)", "p0_direct_native": "P0 (direct native)"}
    for baseline,experiment,question in pairs:
        if baseline not in by_name or experiment not in by_name:
            continue
        a,b=by_name[baseline],by_name[experiment]
        valid=a.get("status")=="ok" and b.get("status")=="ok"
        va,vb=a.get("val",{}),b.get("val",{})
        a_label = labels.get(baseline, baseline.split("_")[0].upper())
        b_label = labels.get(experiment, experiment.split("_")[0].upper())
        cells=[f"{a_label} → {b_label}",question]
        cells.extend([delta(va,vb,get) if valid else "—" for get in (
            lambda m:m.get("fpr"),lambda m:m.get("worst_equipment_fpr"),
            lambda m:m.get("recall"),lambda m:region(m,"1px"),lambda m:region(m,"2-4px"))])
        rows.append((cells,False))
    if not rows:
        return "<p class='note'>Matched P0–P5 comparisons require both experiments in a pair.</p>"
    return ("<h3>실험별 학습 효과 비교 (validation)</h3>"
            "<p>비교 실험 값 − 기준 실험 값. 단위 pp는 percentage point(퍼센트포인트)이다. "
            "예: FPR 10% → 5%이면 −5.00 pp. FPR은 음수, recall은 양수 방향이 좋다. "
            "각 모델은 자기 validation으로 보정한 threshold에서 비교하므로 최종 선정 조건과 실제 validation recall도 함께 확인한다. "
            "실험이 실패했거나 지표가 없으면 —로 표시한다.</p>"
            +table(["비교","검증하는 효과","Δ FPR ↓","Δ Worst FPR ↓","Δ Image Recall ↑","Δ 1px Recall ↑","Δ 2–4px Recall ↑"],rows))

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
        chart("Test Dice",results,lambda m:m.get("dice")),
        chart("Test image F1",results,lambda m:m.get("image_f1")),
        chart("Test pixel AUROC",results,lambda m:m.get("pixel_auroc")),
        chart("Test pixel AUPRO@0.30",results,lambda m:m.get("pixel_aupro"))
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
          "<p>Threshold를 낮추면 더 많은 불량을 찾지만 정상 이미지 과검출도 늘 수 있다. "
          "그래프의 가로축은 정상 image FPR, 세로축은 image 또는 크기별 region recall이다. "
          "같은 FPR에서 recall이 높을수록, 같은 recall에서 FPR이 낮을수록 좋으므로 왼쪽 위에 가까운 곡선을 찾는다.</p>",
          "<p class='note'>이 그래프와 operating point는 validation의 raw score에서 정한 공통 image/pixel threshold를 test에도 그대로 적용한 진단 결과다. "
          "최종 독립 image/pixel F1-adaptive threshold (정규화 후 0.5)와 다르며 test 곡선으로 threshold를 다시 정하지 않는다. "
          "해당 크기의 정답 영역이 없으면 지표가 표시되지 않는다.</p>",
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
    body.append("<details><summary>Operating points · 목표 recall/FPR에서 비교하기</summary>"
                "<p class='note'>image_recall은 validation 목표 image recall, 1px_recall/2-4px_recall은 "
                "해당 크기 영역의 목표 recall, fpr_ceiling은 validation에서 허용하는 최대 image FPR이다. "
                "예를 들어 validation image recall 95%에서 FPR이 낮은지와 test에서도 recall을 유지하는지 확인한다.</p>"
                +table(["Experiment","Validation target","Target","Split","Threshold","FPR","Image recall","1px recall","2–4px recall"],rows)+"</details>")
    if learning:
        body.append("<details><summary>Backbone training diagnostics (proxy, not PaDiM performance)</summary>"
                    "<p class='note'>Weight change는 backbone 파라미터가 실제 변경되었는지, MixStyle calls는 실제 혼합 호출 수를 확인하는 값이다. "
                    "Final validation proxy는 임시 head 또는 consistency 진단 지표이며 최종 PaDiM 성능이 아니다.</p>"+
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
    is_padim_suite=False
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
        is_padim_suite |= any(r.get("experiment") in {e[0] for e in PADIM_EXPERIMENTS} for r in results)
        selected_path=directory/"selected.json"
        best=load_json(selected_path).get("selected_by_validation") if selected_path.is_file() else None
        chosen=next((r for r in results if r.get("experiment")==best and r.get("status")=="ok"),None)
        m=chosen["test"] if chosen else {}
        overview.append(([label,state,best or "—",percent(m.get("fpr")),percent(m.get("recall")),
                          numeric(m.get("dice")),percent(region(m,"1px")),percent(region(m,"2-4px"))],False))
        config_path=directory/"config.json"
        config=load_json(config_path) if config_path.is_file() else {}
        section.append(f"<p>Validation-selected model: <strong>{esc(best or 'None recorded')}</strong>. "
                       f"Validation goals: image recall {percent(config.get('recall'))}, "
                       f"small-region recall {percent(config.get('region_recall'))}.</p>")
        section.append("<p class='note'>먼저 Test FPR, Recall 및 작은 영역 Recall 차트를 확인하고, "
                       "그다음 표에서 validation/test와 설비별 수치를 비교한다. "
                       "아래 test 결과는 독립 평가 결과이며 모델을 다시 선택하는 기준이 아니다.</p>")
        operating_results = [r for r in results if "validation_constraints" in r]
        if operating_results:
            section.append(f"<p>Validation background pixel FPR cap: {percent(config.get('pixel_fpr_cap'))}. "
                           + ("</p>" if best else "No candidate met all validation constraints.</p>"))
            checks = sorted({k for r in operating_results for k in r["validation_constraints"]})
            status_text = {True: "충족", False: "미달", None: "GT 없음"}
            section.append(table(["Experiment",*checks], [
                ([r["experiment"],*[status_text[r["validation_constraints"][k]]
                   if k in r["validation_constraints"] else "—" for k in checks]],r["experiment"]==best)
                for r in operating_results]))
        section.append(chart_group(results))
        section.append(padim_comparison(results))
        section.append(tradeoff_section(results))
        post=[r for r in results if r.get("post_processor")]
        if post:
            keys=("image_threshold","image_min","image_max","pixel_threshold","pixel_min","pixel_max")
            section.append("<details><summary>Validation-calibrated post-processing · threshold 세부값</summary>"
                           "<p class='note'>새 판정은 image/pixel raw score가 각 임계값 이상이면 불량으로 판정한다. "
                           "정규화는 표시용이고, test에서도 validation 임계값을 그대로 쓴다.</p>"+
                           table(["Experiment","Calibration","Decision rule",*keys], [
                               ([r["experiment"],r["post_processor"].get("method"),
                                 r["post_processor"].get("decision_rule","normalized_gt"),
                                 *[numeric(r["post_processor"].get(k)) for k in keys]],r["experiment"]==best)
                               for r in post])+"</details>")
        if "padim_input_shape" in config:
            section.append(f"<p>PaDiM input (H,W): {esc(config['padim_input_shape'])}; "
                           f"upsample factor: {esc(config.get('upsample_factor') or 'square letterbox')}; "
                           f"max layer depth: {esc(config.get('max_layer_depth'))}; "
                           f"n_features: {esc(config.get('padim_dim'))}; "
                           f"anomaly map restoration: {esc(config.get('padim_map_restore','not recorded'))}; "
                           f"image score source: {esc(config.get('padim_image_score_source','not recorded'))}.</p>")
        if (directory/"report.html").is_file():
            section.append(f"<p><a href='{quote(name)}/report.html'>Optional original detailed report</a></p>")
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
        section.append("<h3>Standard metrics (validation/test)</h3>"
                       "<p class='note'>기존 평가 지표와 비교하기 위한 표이다. AUROC/AUPRO는 score 기반 평가, Image F1은 최종 판정 기반 평가이다.</p>")
        standard=[]
        for r in results:
            if r["status"] != "ok": continue
            v,t=r["val"],r["test"]
            standard.append(([r["experiment"],numeric(v.get("image_auroc",v.get("auroc"))),numeric(v.get("pixel_auroc")),numeric(t.get("image_f1")),numeric(t.get("pixel_aupro")),numeric(t.get("pixel_auroc"))],r["experiment"]==best))
        section.append(table(["Experiment","Val image AUROC","Val pixel AUROC","Test image F1","Test pixel AUPRO@0.30","Test pixel AUROC"],standard))
        section.append("<h3>All experiments · validation & test</h3>")
        section.append(table(["Experiment","Status","Split","Threshold",*HEAD,"Error"],main_rows))
        section.append("<details><summary>Region hit / count (native pixel area) · 검출한 영역 수 / 전체 영역 수</summary>"+
                       table(["Experiment","Split",*SIZES],region_rows)+"</details>")
        section.append("<details><summary>Per-equipment metrics · 설비별 지표</summary>"+
                       table(["Experiment","Split","Equipment",*HEAD],detail_rows)+"</details>")
        sections.append("".join(section)+"</section>")
    content=["<!doctype html><html lang='ko'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>CSAM consolidated report</title><style>",CSS,"</style></head><body><main>",
             "<h1>CSAM consolidated report</h1><p>Void 검출 실험 통합 결과</p>",
             "<p class='note'>이 HTML 하나로 지표와 차트를 열람할 수 있다. 별도의 상세 보고서 링크는 원래 결과 폴더가 필요하다.</p>",
             "<nav><a href='#read-guide'>읽는 법</a><a href='#overview'>선정 모델 요약</a>"
             "<a href='#metric-guide'>지표 사전</a>",
             "<a href='#padim-guide'>P0–P5 실험 설명</a>" if is_padim_suite else "",
             "<a href='comparison.html'>P0 대비 정성·정량 비교</a>" if (root/"comparison.html").is_file() else "",
             "".join(nav),"</nav>",reading_guide(),"<section id='overview'><h2>Protocol overview · validation-selected models</h2>"
             "<p>각 평가 조건에서 validation으로 선택된 모델의 test 결과이다. 아래 실험별 표에서 다른 후보도 확인한다.</p>",
             table(["Protocol / equipment","Status","Selected model","Test FPR","Test recall","Test Dice",
                    "Test 1px recall","Test 2–4px recall"],overview),"</section>"]
    if holdout:
        content += ["<section><h2>Holdout comparison · all successful experiments</h2>",
                    table(["Held-out equipment","Experiment","Test FPR","Test recall","Dice","1px recall",
                           "2–4px recall","FP","FN"],holdout),"</section>"]
    if is_padim_suite:
        content.append(padim_guide())
    content.append(metric_guide())
    content += sections
    content += ["<footer>FPR = FP / (FP + TN); image recall = TP / (TP + FN). Region recall = fraction of "
                "ground-truth connected components touched by at least one predicted pixel. Size is pixel area, not diameter. "
                "Pixel AUPRO integrates region-balanced overlap up to pixel FPR 0.30, normalized by 0.30. "
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

