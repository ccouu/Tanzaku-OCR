"""Conservative oriented-line grouping and collision-bounded crop margins.

Never erode detected text cores to manufacture non-overlap. Unresolved core
intersections are explicit review flags, not silently clipped glyphs.
"""
import math
import numpy as np
from PIL import Image

MINOR_OVERLAP_POLICY_VERSION = "minor_overlap_v1_strict_lt10pct"
MINOR_OVERLAP_MAX_AREA_FRAC = 0.10
MINOR_OVERLAP_MAX_PERP_FRAC = 0.10
MINOR_OVERLAP_MAX_ANGLE_DEG = 8.0


def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])
def signed_area(q):return .5*sum(cross(a,b) for a,b in zip(q,np.roll(q,-1,axis=0)))
def area(q):return abs(signed_area(np.asarray(q,dtype=float))) if len(q)>=3 else 0.


def intersection(a,b):
    """Exact convex-polygon clipping; unlike AABB IoU, respects rotated text."""
    out=np.asarray(a,dtype=float)
    clip=np.asarray(b,dtype=float)
    if signed_area(clip)<0:clip=clip[::-1]
    for c,d in zip(clip,np.roll(clip,-1,axis=0)):
        if not len(out):break
        inp=out;result=[];s=inp[-1]
        for e in inp:
            inside_e=cross(d-c,e-c)>=-1e-8;inside_s=cross(d-c,s-c)>=-1e-8
            if inside_e!=inside_s:
                den=cross(e-s,d-c)
                if abs(den)>1e-12:result.append(s+(e-s)*(cross(c-s,d-c)/den))
            if inside_e:result.append(e)
            s=e
        out=np.asarray(result,dtype=float).reshape(-1,2)
    return out


def overlap(a,b):
    inter=area(intersection(a,b));aa=area(a);bb=area(b)
    return dict(intersection=inter,iou=inter/max(1e-9,aa+bb-inter),iom=inter/max(1e-9,min(aa,bb)))


def classify_core_overlap(
    a,
    b,
    *,
    max_area_frac=MINOR_OVERLAP_MAX_AREA_FRAC,
    max_perp_frac=MINOR_OVERLAP_MAX_PERP_FRAC,
    max_angle_deg=MINOR_OVERLAP_MAX_ANGLE_DEG,
):
    """Classify an oriented text-core intersection pair under strict <10% area & perpendicular intrusion.

    Returns explicit area fractions, perpendicular intrusion fractions, reading/normal axes,
    denominators, severity ('clear' | 'minor_allowed' | 'major_review'), and allowance booleans.
    """
    poly = intersection(a["core"], b["core"])
    inter_area = area(poly)
    aa = area(a["core"])
    bb = area(b["core"])
    ha = float(a.get("height", 0.0))
    hb = float(b.get("height", 0.0))
    ua = np.asarray(a["u"], dtype=float)
    va = np.asarray(a["v"], dtype=float)
    ub = np.asarray(b["u"], dtype=float)
    vb = np.asarray(b["v"], dtype=float)

    base_ov = dict(
        policy_version=MINOR_OVERLAP_POLICY_VERSION,
        intersection=float(inter_area) if math.isfinite(inter_area) else float("nan"),
        iou=float(inter_area / max(1e-9, aa + bb - inter_area)) if (math.isfinite(inter_area) and aa > 0 and bb > 0) else 0.0,
        iom=float(inter_area / max(1e-9, min(aa, bb))) if (math.isfinite(inter_area) and aa > 0 and bb > 0) else 0.0,
        core_area_a=float(aa),
        core_area_b=float(bb),
        thickness_a=ha,
        thickness_b=hb,
        reading_axis_a=ua.tolist(),
        reading_axis_b=ub.tolist(),
        perp_axis_a=va.tolist(),
        perp_axis_b=vb.tolist(),
    )

    finite_geom = (
        math.isfinite(inter_area)
        and math.isfinite(aa) and aa > 1e-9
        and math.isfinite(bb) and bb > 1e-9
        and math.isfinite(ha) and ha > 1e-9
        and math.isfinite(hb) and hb > 1e-9
        and np.all(np.isfinite(ua)) and np.all(np.isfinite(va))
        and np.all(np.isfinite(ub)) and np.all(np.isfinite(vb))
        and np.linalg.norm(ua) > 1e-9 and np.linalg.norm(ub) > 1e-9
    )

    if not finite_geom:
        return dict(
            **base_ov,
            area_frac_a=float("nan"),
            area_frac_b=float("nan"),
            perp_extent_a=float("nan"),
            perp_extent_b=float("nan"),
            perp_frac_a=float("nan"),
            perp_frac_b=float("nan"),
            angle_deg=float("nan"),
            severity="major_review",
            minor_allowed=False,
            review_required=True,
            legacy_review_required=True,
            reason="degenerate_or_nonfinite_geometry",
        )

    if len(poly) < 3 or inter_area <= 1e-9:
        return dict(
            **base_ov,
            area_frac_a=0.0,
            area_frac_b=0.0,
            perp_extent_a=0.0,
            perp_extent_b=0.0,
            perp_frac_a=0.0,
            perp_frac_b=0.0,
            angle_deg=math.degrees(math.acos(min(1.0, abs(float(ua @ ub))))),
            severity="clear",
            minor_allowed=True,
            review_required=False,
            legacy_review_required=False,
            reason="no_core_intersection",
        )

    if not np.all(np.isfinite(poly)):
        return dict(
            **base_ov,
            area_frac_a=float("nan"),
            area_frac_b=float("nan"),
            perp_extent_a=float("nan"),
            perp_extent_b=float("nan"),
            perp_frac_a=float("nan"),
            perp_frac_b=float("nan"),
            angle_deg=float("nan"),
            severity="major_review",
            minor_allowed=False,
            review_required=True,
            legacy_review_required=True,
            reason="nonfinite_intersection_polygon",
        )

    area_frac_a = float(inter_area / aa)
    area_frac_b = float(inter_area / bb)
    perp_extent_a = float(np.ptp(poly @ va))
    perp_extent_b = float(np.ptp(poly @ vb))
    perp_frac_a = float(perp_extent_a / ha)
    perp_frac_b = float(perp_extent_b / hb)
    angle_deg = math.degrees(math.acos(min(1.0, abs(float(ua @ ub)))))

    is_minor = (
        math.isfinite(area_frac_a) and area_frac_a < max_area_frac
        and math.isfinite(area_frac_b) and area_frac_b < max_area_frac
        and math.isfinite(perp_frac_a) and perp_frac_a < max_perp_frac
        and math.isfinite(perp_frac_b) and perp_frac_b < max_perp_frac
        and math.isfinite(angle_deg) and angle_deg <= max_angle_deg
    )

    if is_minor:
        reason = "minor_overlap_allowed_strict_lt10pct"
    elif angle_deg > max_angle_deg:
        reason = "incompatible_reading_axes"
    elif area_frac_a >= max_area_frac or area_frac_b >= max_area_frac:
        reason = "core_area_fraction_ge_threshold"
    else:
        reason = "perpendicular_intrusion_ge_threshold"

    return dict(
        **base_ov,
        area_frac_a=area_frac_a,
        area_frac_b=area_frac_b,
        perp_extent_a=perp_extent_a,
        perp_extent_b=perp_extent_b,
        perp_frac_a=perp_frac_a,
        perp_frac_b=perp_frac_b,
        angle_deg=angle_deg,
        severity="minor_allowed" if is_minor else "major_review",
        minor_allowed=bool(is_minor),
        review_required=not bool(is_minor),
        legacy_review_required=True,
        reason=reason,
    )


def classify_boxes_core_overlaps(
    boxes,
    *,
    iom_trigger=0.02,
    max_area_frac=MINOR_OVERLAP_MAX_AREA_FRAC,
    max_perp_frac=MINOR_OVERLAP_MAX_PERP_FRAC,
    max_angle_deg=MINOR_OVERLAP_MAX_ANGLE_DEG,
):
    """Aggregate pairwise core-overlap classifications across a list of oriented boxes."""
    conflicts = []
    per_crop = [
        dict(
            overlap_severity="clear",
            minor_overlap_allowed=False,
            review_required=False,
            legacy_review_required=False,
            overlap_conflict_count=0,
            minor_conflict_count=0,
            major_conflict_count=0,
            overlap_source_ids=set(),
            minor_overlap_policy_version=MINOR_OVERLAP_POLICY_VERSION,
        )
        for _ in boxes
    ]

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            m = overlap(a["core"], b["core"])
            if m["iom"] <= iom_trigger:
                continue
            cls = classify_core_overlap(
                a, b, max_area_frac=max_area_frac, max_perp_frac=max_perp_frac, max_angle_deg=max_angle_deg
            )
            conflict = dict(a=i, b=j, source_a=list(a["source_ids"]), source_b=list(b["source_ids"]), **cls)
            conflicts.append(conflict)
            for idx, other in ((i, b), (j, a)):
                st = per_crop[idx]
                st["overlap_conflict_count"] += 1
                st["legacy_review_required"] = True
                st["overlap_source_ids"].update(other["source_ids"])
                if cls["minor_allowed"]:
                    st["minor_conflict_count"] += 1
                else:
                    st["major_conflict_count"] += 1

    for st in per_crop:
        st["overlap_source_ids"] = sorted(st["overlap_source_ids"])
        if st["overlap_conflict_count"] == 0:
            st["overlap_severity"] = "clear"
            st["minor_overlap_allowed"] = False
            st["review_required"] = False
        elif st["major_conflict_count"] > 0:
            st["overlap_severity"] = "major_review"
            st["minor_overlap_allowed"] = False
            st["review_required"] = True
        else:
            st["overlap_severity"] = "minor_allowed"
            st["minor_overlap_allowed"] = True
            st["review_required"] = False

    return conflicts, per_crop


def inside(point,polygon):
    q=np.asarray(polygon);sign=1 if signed_area(q)>0 else -1
    return all(sign*cross(b-a,np.asarray(point)-a)>=-1e-6 for a,b in zip(q,np.roll(q,-1,axis=0)))


def rectangle(points,u):
    u=np.asarray(u,dtype=float);u=u/np.linalg.norm(u);v=np.array([-u[1],u[0]])
    pts=np.asarray(points,dtype=float);x=pts@u;y=pts@v
    return np.array([u*x.min()+v*y.min(),u*x.max()+v*y.min(),u*x.max()+v*y.max(),u*x.min()+v*y.max()])


def frame(points,vertical=False):
    q=np.asarray(points,dtype=float).reshape(4,2);center=q.mean(0)
    q=q[np.argsort(np.arctan2(q[:,1]-center[1],q[:,0]-center[0]))]
    edges=np.roll(q,-1,axis=0)-q;lens=np.linalg.norm(edges,axis=1)
    a=(lens[0]+lens[2])/2;b=(lens[1]+lens[3])/2
    if max(a,b)/max(1e-9,min(a,b))<1.3:
        desired=np.array([0.,1.]) if vertical else np.array([1.,0.])
        index=max((0,1),key=lambda i:abs(float(edges[i]@desired))/max(1e-9,lens[i]))
    else:index=0 if a>=b else 1
    u=edges[index]/max(1e-9,lens[index]);axis=1 if vertical else 0
    if u[axis]<-1e-8 or (abs(u[axis])<1e-8 and u[1-axis]<0):u=-u
    return u,np.array([-u[1],u[0]])


def box(points,label,confidence,ids,parts=None):
    vertical=label=='dish_vertical';u,v=frame(points,vertical);core=rectangle(points,u)
    return dict(core=core,u=u,v=v,center=core.mean(0),height=float(np.ptp(core@v)),span=float(np.ptp(core@u)),
        label=label,confidence=float(confidence),source_ids=list(ids),parts=parts if parts is not None else [dict(core=core,u=u)])


def pair_metrics(a,b):
    dot=float(a['u']@b['u']);u=a['u']+b['u']*(1 if dot>=0 else -1);u=u/max(1e-9,np.linalg.norm(u));v=np.array([-u[1],u[0]])
    ay=a['core']@v;by=b['core']@v;ax=a['core']@u;bx=b['core']@u
    h1=float(np.ptp(ay));h2=float(np.ptp(by));h=max(h1,h2)
    return dict(angle=math.degrees(math.acos(min(1.,abs(dot)))),
        perpendicular_overlap=max(0.,min(ay.max(),by.max())-max(ay.min(),by.min()))/max(1e-9,min(h1,h2)),
        centerline_offset=abs(float(a['center']@v-b['center']@v))/max(1e-9,h),height_ratio=h/max(1e-9,min(h1,h2)),
        gap=max(0.,max(ax.min(),bx.min())-min(ax.max(),bx.max())),height=min(h1,h2),**overlap(a['core'],b['core']))


def deduplicate(boxes,decisions,threshold=.65):
    boxes=list(boxes)
    while True:
        changed=False
        for i,a in enumerate(boxes):
            for j in range(i+1,len(boxes)):
                b=boxes[j];m=pair_metrics(a,b)
                aligned=m['angle']<=8 and m['perpendicular_overlap']>=.7 and m['centerline_offset']<=.25
                duplicate=aligned and m['iou']>=threshold
                contained=aligned and a['label']==b['label'] and m['iom']>=.92 and m['height_ratio']<=1.5
                if not (duplicate or contained):continue
                larger,smaller=(a,b) if area(a['core'])>=area(b['core']) else (b,a)
                # A high-confidence word inside a credible full-line box must not
                # replace the full line and delete its remaining characters.
                if contained and area(larger['core'])>=1.2*area(smaller['core']) and larger['confidence']>=max(.25,.5*smaller['confidence']):
                    winner,loser=larger,smaller;reason='contained_fragment_complete_line_preserved'
                elif duplicate:
                    winner,loser=(a,b) if a['confidence']>=b['confidence'] else (b,a);reason='rotated_duplicate'
                else:continue
                decisions.append(dict(action='suppress_duplicate',reason=reason,kept=list(winner['source_ids']),suppressed=list(loser['source_ids']),metrics=m))
                # Collapse duplicate output count while retaining every observed
                # text extent;confidence/labelstillcome from the selectedwinner.
                core=rectangle(np.concatenate([winner['core'],loser['core']]),winner['u'])
                winner=box(core,winner['label'],winner['confidence'],sorted(set(winner['source_ids']+loser['source_ids'])))
                boxes=[x for k,x in enumerate(boxes) if k not in (i,j)]+[winner];changed=True;break
            if changed:break
        if not changed:return boxes


def merge_candidate(a,b,others):
    m=pair_metrics(a,b)
    if a['label']!=b['label'] or m['angle']>8 or m['perpendicular_overlap']<.7 or m['centerline_offset']>.25 or m['height_ratio']>1.5 or m['gap']>.75*m['height']:
        return None,m
    parts=a['parts']+b['parts'];u=a['u']+b['u']*(1 if a['u']@b['u']>=0 else -1);u/=np.linalg.norm(u);v=np.array([-u[1],u[0]])
    for p in parts:
        for q in parts:
            if abs(float(p['u']@q['u']))<math.cos(math.radians(8)):return None,m
    heights=[float(np.ptp(p['core']@v)) for p in parts];centers=[float(p['core'].mean(0)@v) for p in parts]
    if max(heights)/max(1e-9,min(heights))>1.5 or max(centers)-min(centers)>.25*max(heights):return None,m
    core=rectangle(np.concatenate([p['core'] for p in parts]),u)
    if np.ptp(core@v)>1.5*max(heights):return None,m
    # Prevent jumping across a third text region. Nearest compatible fragments
    # can still be combined first, then reconsidered as a whole line.
    for c in others:
        if c is a or c is b:continue
        if inside(c['center'],core) and not inside(c['center'],a['core']) and not inside(c['center'],b['core']):return None,m
    merged=box(core,a['label'],max(a['confidence'],b['confidence']),sorted(set(a['source_ids']+b['source_ids'])),parts)
    return merged,m


def group_lines(boxes,decisions,min_merge_iom=.2):
    boxes=list(boxes)
    while True:
        candidates=[]
        for i,a in enumerate(boxes):
            for j in range(i+1,len(boxes)):
                merged,m=merge_candidate(a,boxes[j],boxes)
                if merged is not None and m['iom']>=min_merge_iom:candidates.append((m['gap']/max(1e-9,m['height']),i,j,merged,m))
        if not candidates:return boxes
        _,i,j,merged,m=min(candidates,key=lambda x:(x[0],x[1],x[2]))
        decisions.append(dict(action='merge_line_fragments',sources=[boxes[i]['source_ids'],boxes[j]['source_ids']],result=merged['source_ids'],metrics=m))
        boxes=[b for k,b in enumerate(boxes) if k not in (i,j)]+[merged]


def padded(b,scale,width,height):
    q=b['core'];u=b['u'];v=b['v'];along=.12*b['height']*scale;across=.08*b['height']*scale
    result=np.array([q[0]-u*along-v*across,q[1]+u*along-v*across,q[2]+u*along+v*across,q[3]-u*along+v*across])
    result[:,0]=np.clip(result[:,0],0,width);result[:,1]=np.clip(result[:,1],0,height)
    return result


def process(detections,width,height,duplicate_iou=.65,min_merge_iom=.2):
    """Return JSON-ready crops, audit decisions and unresolved core intersections."""
    decisions=[];boxes=[box(d['points'],d['label'],d.get('confidence',1),[d.get('id',str(i))]) for i,d in enumerate(detections)]
    boxes=deduplicate(boxes,decisions,duplicate_iou);boxes=group_lines(boxes,decisions,min_merge_iom)
    boxes.sort(key=lambda b:(b['label']=='dish_vertical',-b['center'][0] if b['label']=='dish_vertical' else b['center'][1],b['center'][1] if b['label']=='dish_vertical' else b['center'][0]))
    scales=[1.]*len(boxes)
    for i,a in enumerate(boxes):
        for j in range(i+1,len(boxes)):
            b=boxes[j];m=overlap(a['core'],b['core'])
            current=overlap(padded(a,scales[i],width,height),padded(b,scales[j],width,height))['intersection']
            eps=max(1e-5,min(area(a['core']),area(b['core']))*1e-7)
            if current<=m['intersection']+eps:continue
            # Shrink ONLY added margins. Detected cores are never reduced.
            if m['intersection']>eps:scales[i]=scales[j]=0.;continue
            low=0.;high=1.
            for _ in range(16):
                mid=(low+high)/2
                inter=overlap(padded(a,scales[i]*mid,width,height),padded(b,scales[j]*mid,width,height))['intersection']
                if inter<=eps:low=mid
                else:high=mid
            scales[i]*=low;scales[j]*=low
    conflicts, per_crop = classify_boxes_core_overlaps(boxes)
    output=[]
    for i,b in enumerate(boxes):
        q=padded(b,scales[i],width,height)
        if scales[i]<.999:decisions.append(dict(action='reduce_padding',sources=b['source_ids'],remaining_fraction=scales[i],reason='Avoidnewneighboroverlapwithoutclippingtextcore'))
        st=per_crop[i]
        output.append(dict(id=f'line_{i:03d}',label=b['label'],confidence=b['confidence'],source_ids=b['source_ids'],
            core=b['core'].tolist(),crop_polygon=q.tolist(),orientation_flag=int(b['label']=='dish_vertical'),padding_scale=scales[i],
            overlap_severity=st['overlap_severity'],minor_overlap_allowed=st['minor_overlap_allowed'],
            review_required=st['review_required'],legacy_review_required=st['legacy_review_required'],
            overlap_conflict_count=st['overlap_conflict_count'],overlap_source_ids=st['overlap_source_ids'],
            minor_overlap_policy_version=st['minor_overlap_policy_version']))
    for c in conflicts:
        act = 'minor_core_overlap_allowed' if c['minor_allowed'] else 'unresolved_core_overlap'
        decisions.append(dict(action=act, **c))
    return dict(crops=output,decisions=decisions,conflicts=conflicts,raw_count=len(detections),processed_count=len(output),
        policy=dict(duplicate_rotated_iou=duplicate_iou,containment_iom=.92,angle_degrees=8,perpendicular_overlap=.7,
                    centerline_fraction=.25,height_ratio=1.5,max_gap_height=.75,head_tail_margin_height=.12,perpendicular_margin_height=.08,
                    minimum_merge_intersection_over_smaller_area=min_merge_iom,core_clipping=False,unresolved_overlap_is_flagged=True,
                    minor_overlap_policy_version=MINOR_OVERLAP_POLICY_VERSION,
                    minor_overlap_max_area_frac=MINOR_OVERLAP_MAX_AREA_FRAC,
                    minor_overlap_max_perp_frac=MINOR_OVERLAP_MAX_PERP_FRAC,
                    minor_overlap_max_angle_deg=MINOR_OVERLAP_MAX_ANGLE_DEG))


def rectify(image,quad):
    """Reading-axis RGB strip; vertical glyphs rotate CCW,matchingrecognizerflag1."""
    q=np.asarray(quad,dtype=float);w=max(2,round((np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2));h=max(2,round((np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2))
    dest=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],dtype=float);matrix=[];values=[]
    for (x,y),(u,v) in zip(dest,q):
        matrix.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]]);values.extend([u,v])
    transform=np.linalg.solve(np.asarray(matrix),np.asarray(values))
    return image.convert('RGB').transform((w,h),Image.Transform.PERSPECTIVE,tuple(transform),resample=Image.Resampling.BICUBIC,fillcolor=(240,240,240))


def recognition_input(strip):
    """Aspect-preservingH64,next128/256/384/512;extremelinesfitwithverticalletterbox."""
    scale=min(64/strip.height,512/strip.width);w=min(512,max(1,round(strip.width*scale)));h=min(64,max(1,round(strip.height*scale)))
    im=strip.convert('RGB').resize((w,h),Image.Resampling.BICUBIC);canvas=next(c for c in (128,256,384,512) if w<=c);top=(64-h)//2
    arr=np.pad(np.asarray(im),((top,64-h-top),(0,canvas-w),(0,0)),mode='edge')
    return Image.fromarray(arr),dict(valid_width=w,canvas=canvas,input_length=(w+3)//4,scaled_height=h,
        aspect_preserved=True,glyph_clipped=False,vertical_letterbox=h<64)
