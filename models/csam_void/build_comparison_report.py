"""Portable, local P0-vs-P1..P5 report from compact per-image test evidence."""
import argparse
import base64
import json
import random
from statistics import quantiles
from pathlib import Path

from build_report import CSS, PADIM_EXPERIMENTS, esc, load_json, table

BASE = "p0_pretrained"
CANDIDATES = tuple(name for name, _, _ in PADIM_EXPERIMENTS if name != BASE)
BUCKETS = ("all", "1px", "2-4px", "5+px")
EXAMPLES_PER_CATEGORY = 2


def ratio(n, d):
    return n / d if d else None


def pct(value):
    return "N/A" if value is None else f"{value*100:.2f}%"


def pp(before, after):
    return "N/A" if before is None or after is None else f"{(after-before)*100:+.2f} pp"


def score(value):
    return "N/A" if value is None else f"{value:.4f}"


def tab(headers, rows):
    return table(headers, [(list(row), False) for row in rows])


def load_samples(path, expected_ids=None):
    data = load_json(path)
    if data.get("schema_version") != 1 or data.get("split") != "test":
        raise ValueError("Unsupported comparison evidence schema")
    samples = data["samples"]
    by_id = {r["id"]: r for r in samples}
    if not samples or len(by_id) != len(samples):
        raise ValueError("Empty evidence or duplicate sample IDs")
    if expected_ids is not None and set(by_id) != expected_ids:
        raise ValueError("Evidence sample IDs do not match the protocol test audit")
    for r in samples:
        if (r["label"] not in (0,1) or len(r["bg_hist"]) != 10 or
                sum(r["bg_hist"]) != r["bg_pixels"] or not 0<=r["bg_fp"]<=r["bg_pixels"] or
                any(not 0<=v["overlap"]<=v["area"] for v in r["regions"])):
            raise ValueError("Inconsistent comparison evidence counts")
    return by_id


def align_samples(base, candidate):
    """Never silently compare different test populations or different GT regions."""
    if set(base) != set(candidate):
        raise ValueError("P0 and candidate test IDs differ; paired comparison is invalid")
    pairs = []
    for sample_id, a in base.items():
        b = candidate[sample_id]
        for key in ("fingerprint", "label", "equipment", "lot", "wafer", "bg_pixels"):
            if a[key] != b[key]:
                raise ValueError(f"Sample {sample_id}: image/GT or metadata mismatch ({key})")
        geometry = lambda r: [(v["area"], v["bbox"]) for v in r["regions"]]
        if geometry(a) != geometry(b):
            raise ValueError(f"Sample {sample_id}: GT components differ")
        pairs.append((a, b))
    return pairs


def region_counts(pairs):
    counts = {k: dict(retained=0, lost=0, gained=0, missed=0,
                      base_detected_coverage_sum=0., candidate_detected_coverage_sum=0.) for k in BUCKETS}
    for a, b in pairs:
        for ra, rb in zip(a["regions"], b["regions"]):
            area = ra["area"]
            bucket = "1px" if area == 1 else "2-4px" if area <= 4 else "5+px"
            ah, bh = ra["overlap"] > 0, rb["overlap"] > 0
            key = "retained" if ah and bh else "lost" if ah else "gained" if bh else "missed"
            for k in ("all", bucket):
                counts[k][key] += 1
                if ah:
                    counts[k]["base_detected_coverage_sum"] += ra["overlap"]/area
                    counts[k]["candidate_detected_coverage_sum"] += rb["overlap"]/area
    for c in counts.values():
        c["total"] = sum(c[key] for key in ("retained","lost","gained","missed"))
        c["base_hits"] = c["retained"] + c["lost"]
        c["candidate_hits"] = c["retained"] + c["gained"]
        c["retention"] = ratio(c["retained"], c["base_hits"])
        c["base_detected_coverage"] = ratio(c["base_detected_coverage_sum"],c["base_hits"])
        c["candidate_detected_coverage"] = ratio(c["candidate_detected_coverage_sum"],c["base_hits"])
    return counts


def image_counts(pairs, label):
    selected = [(a, b) for a, b in pairs if a["label"] == label]
    both = sum(a["image_prediction"] and b["image_prediction"] for a, b in selected)
    lost = sum(a["image_prediction"] and not b["image_prediction"] for a, b in selected)
    gained = sum(not a["image_prediction"] and b["image_prediction"] for a, b in selected)
    return dict(total=len(selected), both=both, lost=lost, gained=gained, base=both+lost, candidate=both+gained)


def background(pairs, label=None):
    selected = [(a, b) for a, b in pairs if label is None or a["label"] == label]
    total = sum(a["bg_pixels"] for a, _ in selected)
    counts = []
    for idx in (0, 1):
        fp = sum(pair[idx]["bg_fp"] for pair in selected)
        mean = ratio(sum(pair[idx]["bg_score_sum"] for pair in selected), total)
        hist = [sum(pair[idx]["bg_hist"][i] for pair in selected) for i in range(10)]
        counts.append(dict(fp=fp, total=total, fpr=ratio(fp, total), mean=mean, hist=hist))
    return counts


def summarize_pairs(pairs):
    regions = region_counts(pairs)
    normal = image_counts(pairs, 0)
    void = image_counts(pairs, 1)
    return dict(regions=regions, normal=normal, void=void,
                bg_all=background(pairs), bg_normal=background(pairs, 0), bg_void=background(pairs, 1))


def paired_lot_intervals(pairs, *, repeats=1000):
    """Percentile intervals for candidate minus P0 on matched, resampled root lots."""
    metrics = ("Image FPR", "1px recall", "2–4px recall")
    lots = {}
    for base, candidate in pairs:
        counts = lots.setdefault(base["lot"], [[0, 0] for _ in metrics])
        if base["label"] == 0:
            counts[0][0] += int(candidate["image_prediction"]) - int(base["image_prediction"])
            counts[0][1] += 1
        for a, b in zip(base["regions"], candidate["regions"]):
            if a["area"] <= 4:
                index = 1 if a["area"] == 1 else 2
                counts[index][0] += int(b["overlap"] > 0) - int(a["overlap"] > 0)
                counts[index][1] += 1

    blocks = [lots[k] for k in sorted(lots)]
    rng = random.Random(42)
    deltas = [[] for _ in metrics]
    supporting = [sum(block[i][1] > 0 for block in blocks) for i in range(len(metrics))]
    if max(supporting, default=0) >= 2:
        for _ in range(repeats):
            draw = rng.choices(blocks, k=len(blocks))
            for i, values in enumerate(deltas):
                if supporting[i] >= 2:
                    n = sum(block[i][1] for block in draw)
                    if n:
                        values.append(sum(block[i][0] for block in draw) / n)

    rows = []
    for i, name in enumerate(metrics):
        n = sum(block[i][1] for block in blocks)
        estimate = ratio(sum(block[i][0] for block in blocks), n)
        interval = "N/A"
        if len(deltas[i]) >= 2:
            cuts = quantiles(deltas[i], n=40, method="inclusive")
            interval = f"{cuts[0]*100:+.2f} ~ {cuts[-1]*100:+.2f} pp"
        rows.append((name, n, supporting[i], pp(0., estimate), interval))
    return rows


def region_table(summary):
    rows = []
    for bucket, c in summary["regions"].items():
        rows.append((bucket, c["total"], c["base_hits"], c["candidate_hits"],
                     c["retained"], c["lost"], c["gained"], c["missed"], pct(c["retention"]),
                     pct(ratio(c["base_hits"],c["total"])), pct(ratio(c["candidate_hits"],c["total"])),
                     pct(c["base_detected_coverage"]),pct(c["candidate_detected_coverage"])))
    return tab(["GT 면적", "전체 영역", "P0 검출", "후보 검출", "유지", "놓침", "새로 검출", "둘 다 놓침",
                "P0 검출 유지율 ↑", "P0 region recall", "후보 region recall",
                "P0 기검출 영역 coverage", "후보 기검출 영역 coverage"], rows)


def background_table(summary):
    rows = []
    for name, key in (("전체 GT 배경", "bg_all"), ("정상 이미지 배경", "bg_normal"),
                      ("Void 이미지의 GT 외부", "bg_void")):
        a, b = summary[key]
        rows.append((name, a["total"], a["fp"], b["fp"], pct(a["fpr"]), pct(b["fpr"]),
                     pp(a["fpr"], b["fpr"]), score(a["mean"]), score(b["mean"])))
    return tab(["범위", "배경 pixel 수", "P0 FP pixel", "후보 FP pixel", "P0 pixel FPR ↓",
                "후보 pixel FPR ↓", "차이 ↓", "P0 평균 score", "후보 평균 score"], rows)


def equipment_table(pairs):
    groups = {}
    for a, b in pairs:
        groups.setdefault(a["equipment"], []).append((a, b))
    rows, fprs, bg_fprs, normal_bg_fprs = [], [], [], []
    for eq, group in sorted(groups.items()):
        s = summarize_pairs(group)
        n = s["normal"]
        a, b = ratio(n["base"],n["total"]), ratio(n["candidate"],n["total"])
        x, y = s["bg_all"]
        nx,ny = s["bg_normal"]
        r = s["regions"]["all"]
        rows.append((eq, n["total"], s["void"]["total"], pct(a), pct(b), pp(a,b),
                     n["lost"], n["gained"], pct(x["fpr"]), pct(y["fpr"]), pp(x["fpr"],y["fpr"]),
                     pct(nx["fpr"]), pct(ny["fpr"]), pp(nx["fpr"],ny["fpr"]),
                     score(x["mean"]), score(y["mean"]), f'{r["retained"]}/{r["base_hits"]}',
                     r["lost"], r["gained"]))
        if a is not None and b is not None:
            fprs.append((a,b))
        if x["fpr"] is not None and y["fpr"] is not None:
            bg_fprs.append((x["fpr"],y["fpr"]))
        if nx["fpr"] is not None and ny["fpr"] is not None:
            normal_bg_fprs.append((nx["fpr"],ny["fpr"]))
    worst_rows = []
    for label, values in (("설비별 image FPR",fprs),("설비별 background pixel FPR",bg_fprs),
                          ("설비별 정상 background pixel FPR",normal_bg_fprs)):
        for agg, fun in (("최댓값",max), ("설비 간 범위 (max-min)",lambda v:max(v)-min(v))):
            valid = len(values) >= (2 if "범위" in agg else 1)
            x,y = (fun([v[i] for v in values]) for i in (0,1)) if valid else (None,None)
            worst_rows.append((label,agg,pct(x),pct(y),pp(x,y)))
    return (tab(["설비", "정상 수", "Void 수", "P0 image FPR", "후보 image FPR", "차이 ↓",
                 "정상 복구", "신규 과검출", "P0 bg pixel FPR", "후보 bg pixel FPR", "차이 ↓",
                 "P0 정상 bg pixel FPR", "후보 정상 bg pixel FPR", "차이 ↓",
                 "P0 bg 평균 score", "후보 bg 평균 score", "유지 / P0 검출", "놓침", "추가 검출"],rows)
            +tab(["설비 편차 지표", "집계", "P0", "후보", "차이 ↓"],worst_rows))


def hist_chart(summary):
    a,b = summary["bg_all"]
    rows = []
    for i in range(10):
        lo,hi = i/10,(i+1)/10
        p,q = ratio(a["hist"][i],a["total"]), ratio(b["hist"][i],b["total"])
        bars = []
        for label,value,color in (("P0",p,"#64748b"),("후보",q,"#15803d")):
            bars.append(f"<div class='hist-line'><b>{label}</b><div class='track'>"
                        f"<div style='width:{100*(value or 0):.4f}%;background:{color}'></div></div>"
                        f"<span>{pct(value)}</span></div>")
        rows.append(f"<div class='hist-row'><span>{lo:.1f}–{hi:.1f}</span>{''.join(bars)}</div>")
    return ("<div class='hist'><h4>GT 배경 normalized anomaly score 분포</h4>"
            "<p class='note'>각 막대는 전체 GT 배경 pixel 중 해당 score 구간의 비율이다. "
            "0.5 경계의 정확한 FP 수는 위 표를 사용한다. 모델별 calibration이 다르므로 raw feature distance 비교는 아니다.</p>"
            +"".join(rows)+"</div>")


def matched_operating_points(base_result, candidate_result):
    def find(result, kind, target):
        return next((p for p in result.get("operating_points", {}).get("points", [])
                     if p["kind"] == kind and p["target"] == target), None)
    rows=[]
    for label, kind, target in (("Validation image recall 95%","image_recall",.95),
                                ("Validation 1px recall 90%","1px_recall",.9),
                                ("Validation 2–4px recall 90%","2-4px_recall",.9),
                                ("Validation image FPR ≤ 5%","fpr_ceiling",.05)):
        a,b=find(base_result,kind,target),find(candidate_result,kind,target)
        for split in ("val","test"):
            x,y=a.get(split,{}) if a else {}, b.get(split,{}) if b else {}
            rows.append((label,split,pct(x.get("fpr")),pct(y.get("fpr")),pp(x.get("fpr"),y.get("fpr")),
                         pct(x.get("recall")),pct(y.get("recall")),pct(x.get("region_1px")),pct(y.get("region_1px")),
                         pct(x.get("region_2_4px")),pct(y.get("region_2_4px"))))
    sources=[result.get("post_processor",{}).get("image_score_source") for result in (base_result,candidate_result)]
    source_note=("Image score는 모델 input 해상도 map의 max, region score는 원본 해상도 map에서 계산한다. "
                 if sources==["model_input_map_max","model_input_map_max"] else
                 "Image score source는 기존 실행의 저장된 calibration 정의를 따른다. ")
    return ("<h4>동일한 validation 목표에서 비교 (추가 진단)</h4>"
            "<p>각 모델의 validation score에서 정한 공통 raw image/pixel threshold를 test에도 고정 적용했다. "
            +source_note+"동일 목표에 도달하더라도 score tie 때문에 실제 validation recall은 다를 수 있다. "
            "위의 최종 판정과 별개이며 test threshold 최적화는 하지 않는다.</p>"
            +tab(["Validation 목표","Split","P0 FPR","후보 FPR","Δ FPR","P0 image recall","후보 image recall",
                  "P0 1px recall","후보 1px recall","P0 2–4px recall","후보 2–4px recall"],rows))


def case_groups(pairs):
    groups = {key: [] for key in ("retained", "lost", "gained", "normal_improved", "background_worse")}
    for a,b in pairs:
        ra, rb = a["regions"], b["regions"]
        lost = sum(x["overlap"]>0 and y["overlap"]==0 for x,y in zip(ra,rb))
        gained = sum(x["overlap"]==0 and y["overlap"]>0 for x,y in zip(ra,rb))
        retained = sum(x["overlap"]>0 and y["overlap"]>0 for x,y in zip(ra,rb))
        delta = b["bg_fp"] - a["bg_fp"]
        if retained and not lost and delta < 0:
            groups["retained"].append((a,b,-delta))
        if lost:
            groups["lost"].append((a,b,lost))
        if gained:
            groups["gained"].append((a,b,gained))
        if a["label"]==0 and (delta<0 or (a["image_prediction"] and not b["image_prediction"])):
            groups["normal_improved"].append((a,b, (int(a["image_prediction"] and not b["image_prediction"]), -delta)))
        if delta>0 or (a["label"]==0 and not a["image_prediction"] and b["image_prediction"]):
            groups["background_worse"].append((a,b,(int(a["label"]==0 and not a["image_prediction"] and b["image_prediction"]),delta)))
    return {k: sorted(v,key=lambda p:(p[2],p[0]["id"]),reverse=True) for k,v in groups.items()}


def panel_html(directory, experiment, sample_id):
    if not str(sample_id).isascii() or not str(sample_id).isdigit():
        return "<p>Panel ID invalid.</p>"
    path = directory/experiment/"test"/f"{sample_id}_panel.png"
    if not path.is_file():
        return "<p>Panel PNG 없음 (정량 결과는 유지).</p>"
    encoded=base64.b64encode(path.read_bytes()).decode("ascii")
    return (f"<figure><figcaption>{esc(experiment)}</figcaption>"
            f"<img loading='lazy' alt='{esc(experiment)} panel' src='data:image/png;base64,{encoded}'></figure>")


def case_section(directory, experiment, pairs, limit):
    titles = {"retained":"기존 void 유지 + 배경 FP 감소", "lost":"P0 검출 void를 놓친 사례",
              "gained":"P0가 놓친 void를 새로 검출", "normal_improved":"정상 과검출 완화 (image 또는 pixel)",
              "background_worse":"background FP 증가 / 신규 과검출"}
    parts = ["<h4>정성 평가: 동일 test 이미지 3-panel</h4>",
             "<p>사례는 동일한 이미지/GT에서 자동 분류한다. 각 분류의 전체 건수와 변화량이 큰 사례를 표시한다. "
             "범주는 중복될 수 있으며 성공 사례만 골라 보여주지 않는다. 원본 grid의 좌표는 0부터 시작한다.</p>"]
    for key, cases in case_groups(pairs).items():
        parts.append(f"<details{' open' if key in ('retained','lost') else ''}><summary>{esc(titles[key])} · {len(cases)}장 (최대 {limit}장 표시)</summary>")
        for a,b,_ in cases[:limit]:
            changes = []
            for i, (ra,rb) in enumerate(zip(a["regions"],b["regions"]),1):
                before,after = ra["overlap"]>0, rb["overlap"]>0
                if before != after:
                    y,x,y2,x2 = ra["bbox"]
                    changes.append(f"GT #{i}: {ra['area']}px, (row {y}:{y2}, col {x}:{x2}), {'유실' if before else '신규 검출'}")
            hits=lambda r: sum(v["overlap"]>0 for v in r["regions"])
            parts.append(f"<article class='case'><h5>Image {esc(a['id'])} · 설비 {esc(a['equipment'])} · "
                         f"Lot {esc(a['lot'])} · Wafer {esc(a['wafer'] or 'N/A')} · GT {'VOID' if a['label']==1 else 'NORMAL'}</h5>"
                         f"<p>검출 GT 영역 {hits(a)} → {hits(b)} / {len(a['regions'])}, "
                         f"배경 FP {a['bg_fp']} → {b['bg_fp']} / {a['bg_pixels']} pixels, "
                         f"Image 판정 {int(a['image_prediction'])} → {int(b['image_prediction'])}</p>")
            if changes:
                parts.append("<p class='note'>"+esc("; ".join(changes[:10]))+("</p>" if len(changes)<=10 else " …</p>"))
            parts.append(panel_html(directory, BASE, a["id"])+panel_html(directory,experiment,a["id"])+"</article>")
        parts.append("</details>")
    return "".join(parts)


def pair_section(directory, result, base_result, pairs, limit):
    name=result["experiment"]
    s=summarize_pairs(pairs)
    r=s["regions"]["all"]
    n,v=s["normal"],s["void"]
    x,y=s["bg_all"]
    nf0,nf1=ratio(n["base"],n["total"]),ratio(n["candidate"],n["total"])
    conclusion = (
        "P0의 검출 영역을 모두 유지" if r["base_hits"] and not r["lost"] else
        f"P0 검출 영역 {r['lost']}개 유실: recall trade-off 확인 필요" if r["base_hits"] else
        "P0가 검출한 영역 없음: 유지율 계산 불가")
    overview=(name, len(pairs), pct(r["retention"]), r["lost"], r["gained"],
              pct(nf0),pct(nf1),pp(nf0,nf1),pct(x["fpr"]),pct(y["fpr"]),pp(x["fpr"],y["fpr"]))
    body=[f"<section><h3>P0 vs {esc(name)}</h3>"
          f"<div class='callout'><strong>{esc(conclusion)}</strong><br>"
          f"배경 pixel FPR {pct(x['fpr'])} → {pct(y['fpr'])} ({pp(x['fpr'],y['fpr'])}), "
          f"정상 image FPR {pct(nf0)} → {pct(nf1)} ({pp(nf0,nf1)}). "
          "실제 개선 여부는 양쪽 지표와 설비별 결과를 함께 판단한다.</div>",
          "<h4>1. Void 검출 유지 여부</h4>",
          f"<p>동일 GT 연결 영역(8-connected)과 예측 mask가 한 pixel 이상 겹치면 검출로 정의한다. "
          f"P0 검출 유지율은 유지 / P0 검출 수이다. Coverage는 P0가 검출했던 같은 GT 영역에서 "
          f"정답 pixel 중 맞춘 비율을 영역별로 평균한 값(놓친 영역은 0)이다. "
          f"Image recall {pct(ratio(v['base'],v['total']))} → {pct(ratio(v['candidate'],v['total']))}. "
          f"Image 단위 기존 검출 유지: {v['both']}/{v['base']} "
          f"({pct(ratio(v['both'],v['base']))}), 기존 검출 놓침 {v['lost']}장, 신규 검출 {v['gained']}장.</p>",
          region_table(s),
          "<h4>2. Background overkill 및 설비 편차</h4>",
          f"<p>정상 이미지 {n['total']}장 중 P0 과검출 {n['base']}장, 후보 과검출 {n['candidate']}장. "
          f"P0 FP에서 정상 판정으로 복구 {n['lost']}장, 신규 과검출 {n['gained']}장. "
          "GT 외의 모든 pixel을 배경으로 정의하며 예측 mask가 켜진 비율을 계산한다.</p>",
          background_table(s),hist_chart(s),"<h4>설비별 비교</h4>",equipment_table(pairs),
          matched_operating_points(base_result,result),case_section(directory,name,pairs,limit),"</section>"]
    return overview,"".join(body)


def build_comparison_report(root, index=None, examples_per_category=EXAMPLES_PER_CATEGORY):
    root=Path(root)
    if not 0<=examples_per_category<=10:
        raise ValueError("examples_per_category must be 0–10")
    if index is None:
        index=load_json(root/"index.json")
    body=[]
    for protocol in index:
        name=protocol["protocol"]
        if name!="pooled" and not (name.startswith("holdout_") and name[8:].isdigit()):
            raise ValueError(f"Unexpected protocol name: {name}")
        directory=root/name
        body.append(f"<section class='protocol'><h2>{esc(name)} · {esc(protocol.get('equipment') or '전체 설비')}</h2>")
        if not (directory/"results.json").is_file():
            body.append(f"<p>비교할 실행 결과 없음: {esc(protocol.get('error') or protocol.get('status'))}</p></section>")
            continue
        results={r["experiment"]:r for r in load_json(directory/"results.json")}
        base=results.get(BASE, {})
        if base.get("status")!="ok":
            body.append("<p>P0가 완료되지 않았거나 선택되지 않아 비교 불가.</p></section>")
            continue
        try:
            audit=directory/"split_audit.json"
            test_ids={r["id"] for r in load_json(audit)["test"]} if audit.is_file() else None
            base_stats=load_samples(directory/BASE/"comparison_stats.json",test_ids)
        except (OSError, ValueError, KeyError) as e:
            body.append(f"<p>P0 비교 통계 없음/오류: {esc(e)}. "
                        "이전 실행의 3-panel만으로 정확한 영역 유지율을 복원할 수 없다.</p></section>")
            continue
        overview, details=[], []
        for exp in CANDIDATES:
            if exp not in results:
                continue
            result=results[exp]
            if result.get("status")!="ok":
                details.append(f"<p>{esc(exp)}: {esc(result.get('status'))} — {esc(result.get('error') or '비교 불가')}</p>")
                continue
            try:
                pairs=align_samples(base_stats,load_samples(directory/exp/"comparison_stats.json",test_ids))
                row,section=pair_section(directory,result,base,pairs,examples_per_category)
            except (OSError,ValueError,KeyError) as e:
                details.append(f"<p>{esc(exp)}: 정확한 paired 비교 불가 — {esc(e)}</p>")
                continue
            overview.append(row)
            details.append(section)
        if overview:
            body.append(tab(["후보 모델","동일 test 이미지 수","P0 region 유지율 ↑","놓친 GT 영역 ↓","추가 GT 영역 ↑",
                             "P0 image FPR","후보 image FPR","Δ FPR ↓","P0 bg pixel FPR","후보 bg pixel FPR","Δ bg FPR ↓"],overview))
        else:
            body.append("<p>완료된 비교 후보가 없다.</p>")
        body.extend(details)
        body.append("</section>")
    extra_css = """
.protocol{border-top:5px solid #3776b0}.protocol section{background:#fcfdff}
.case{border-top:1px solid #d9e1e9;padding:12px 0}.case h5{font-size:14px;margin:8px 0}
.case figure{margin:10px 0}.case img{display:block;max-width:100%;height:auto;image-rendering:pixelated;border:1px solid #e0e7ef}
.case figcaption{font-weight:600;margin-bottom:4px}.hist{max-width:820px;margin:18px 0}
.hist-row{display:grid;grid-template-columns:70px 1fr 1fr;gap:10px;margin:5px 0;align-items:center}
.hist-line{display:grid;grid-template-columns:40px 1fr 65px;gap:5px;align-items:center;font-size:12px}
.hist-line .track div{height:100%}.callout{border-left:4px solid #3776b0;background:#edf5fc;padding:12px;margin:12px 0}
"""
    html = ("<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>PaDiM P0 개선 효과 비교</title><style>"+CSS+extra_css+"</style></head><body><main>"
            "<h1>PaDiM P0 대비 개선 효과 비교</h1>"
            "<section><h2>이 보고서가 답하는 두 가지 질문</h2>"
            "<ol><li>P0에서 잡았던 동일한 실제 void를 후보 모델도 잡는가?</li>"
            "<li>설비별 정상/배경 anomaly score 및 과검출이 감소했는가?</li></ol>"
            "<p>모든 수치와 이미지 비교는 같은 protocol의 동일 test 이미지와 동일 GT mask를 기준으로 한다. "
            "최종 판정에는 각 모델의 validation 보정 threshold를 적용한다. 정성 사례는 자동 선정되며 전체 평가 건수와 함께 표시된다. "
            "Pooled와 holdout은 별도로 해석하고, test 결과로 모델이나 threshold를 다시 선택하지 않는다. "
            "유지율의 분모가 0이면 N/A이며 감소량(pp)은 후보 − P0이다.</p>"
            "<p class='note'>선택된 3-panel PNG를 HTML에 포함했으므로 별도 이미지 파일 없이 열람 가능하다. "
            "실제 이미지, 설비/lot/wafer 메타데이터가 포함된 사내 자료이므로 외부 업로드에 주의한다.</p></section>"
            +"".join(body)+"<footer>정량 평가는 확대 전 native pixel grid에서 계산되며 score는 모델별 validation normalization 기준이다.</footer>"
            "</main></body></html>")
    target=root/"comparison.html"
    tmp=root/"comparison.html.tmp"
    tmp.write_text(html,encoding="utf-8")
    tmp.replace(target)
    return target


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True,help="Completed P0–P5 run directory")
    parser.add_argument("--examples-per-category",type=int,default=EXAMPLES_PER_CATEGORY)
    args=parser.parse_args()
    print(build_comparison_report(args.root,examples_per_category=args.examples_per_category))
