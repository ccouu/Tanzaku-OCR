"""Collect public Tabelog interior or menu photos, one image per restaurant."""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import random
import time
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from PIL import Image, ImageDraw, ImageOps
import requests

from .common import ROOT, digest, dump_json, safe_path

OUTPUT_DIR = ROOT / 'data/raw/tabelog_interiors'
MENU_OUTPUT_DIR = ROOT / 'data/raw/tabelog_menus'
PHOTO_KINDS = {
    'interior': dict(directory=OUTPUT_DIR,subpath='dtlphotolst/3/smp2/',prefix='interior',
                     category='restaurant_interior',image_classes={'rstdtl-thumb-list__img','photolst__img'}),
    'menu': dict(directory=MENU_OUTPUT_DIR,subpath='dtlmenu/photo/',prefix='menu',
                 category='restaurant_menu',image_classes={'rstdtl-photo-list__img'}),
}
START_URL = 'https://tabelog.com/en/tokyo/rstLst/izakaya/'
USER_AGENT = 'JapaneseMenuDatasetCollector/1.0 (small user-directed research sample)'
PROFILE = re.compile(r'^/((?:en/)?[^/]+/A\d+/A\d+/(\d{8})/)')
VOID = {'img', 'br', 'hr', 'input', 'meta', 'link', 'source', 'area', 'base', 'col', 'embed', 'param', 'track', 'wbr'}


def profile_url(url):
    p = urlsplit(url)
    match = PROFILE.match(p.path)
    if p.scheme == 'https' and p.hostname == 'tabelog.com' and match:
        return 'https://tabelog.com/' + match[1], match[2]
    return None


def is_photo_url(url):
    p = urlsplit(url)
    return (p.scheme == 'https' and p.hostname == 'tblg.k-img.com'
            and p.path.startswith('/restaurant/images/')
            and p.path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))


def _parse_srcset_candidates(srcset_value):
    """Extract explicitly published URLs from an HTML srcset attribute, largest width/density first."""
    if not srcset_value:
        return []
    parsed = []
    for part in srcset_value.split(','):
        tokens = part.strip().split()
        if not tokens:
            continue
        url = tokens[0]
        score = 0.0
        if len(tokens) > 1:
            desc = tokens[1].lower()
            try:
                if desc.endswith('w'):
                    score = float(desc[:-1])
                elif desc.endswith('x'):
                    score = float(desc[:-1]) * 1000.0
            except ValueError:
                score = 0.0
        parsed.append((score, url))
    parsed.sort(key=lambda item: item[0], reverse=True)
    return [u for _, u in parsed]


class TabelogHTML(HTMLParser):
    def __init__(self, base_url, photo_kind='interior'):
        super().__init__(convert_charrefs=True)
        self.base_url, self.stack, self.links, self.photos = base_url, [], [], []
        self.next_url = None
        self.photo_kind = photo_kind
        self.image_classes = PHOTO_KINDS[photo_kind]['image_classes']

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a':
            attrs['href'] = urljoin(self.base_url, attrs.get('href', ''))
            attrs['_text'] = []
            self.links.append(attrs)
            if 'next' in attrs.get('rel', '').split() or 'c-pagination__arrow--next' in attrs.get('class', ''):
                if urlsplit(attrs['href']).hostname == 'tabelog.com':
                    self.next_url = attrs['href']
        if tag == 'img' and set(attrs.get('class', '').split()) & self.image_classes:
            anchor = next((a for t, a in reversed(self.stack) if t == 'a'), {})
            candidates = [
                anchor.get('href'),
                anchor.get('data-original'),
                attrs.get('data-original'),
                *_parse_srcset_candidates(attrs.get('data-srcset')),
                *_parse_srcset_candidates(attrs.get('srcset')),
                attrs.get('data-lazy-src'),
                attrs.get('data-src'),
                attrs.get('src'),
            ]
            for source in candidates:
                url = urljoin(self.base_url, source or '')
                if is_photo_url(url):
                    self.photos.append(dict(image_url=url, alt=attrs.get('alt', ''), photo_id=anchor.get('data-id')))
                    break
        if tag not in VOID:
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        anchor = next((a for t, a in reversed(self.stack) if t == 'a'), None)
        if anchor is not None:
            anchor['_text'].append(data)

    def restaurants(self):
        result, seen = [], set()
        for anchor in self.links:
            if not set(anchor.get('class', '').split()) & {'list-rst__rst-name-target', 'cpy-rst-name'}:
                continue
            profile = profile_url(anchor['href'])
            if profile and profile[1] not in seen:
                seen.add(profile[1])
                result.append(dict(restaurant_url=profile[0], restaurant_id=profile[1],
                    restaurant_name=' '.join(''.join(anchor['_text']).split())))
        return result


class AccessBlocked(RuntimeError):
    pass


class PublicClient:
    def __init__(self, delay=2):
        self.delay, self.last_request = delay, 0
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': USER_AGENT, 'Accept-Language': 'en,ja;q=0.8'})

    def get(self, url, referer=None):
        p = urlsplit(url)
        if p.scheme != 'https' or p.hostname not in {'tabelog.com', 'tblg.k-img.com'}:
            raise ValueError('Only public Tabelog page and photo hosts are allowed')
        time.sleep(max(0, self.delay - (time.monotonic()-self.last_request)))
        self.last_request = time.monotonic()
        response = self.session.get(url, headers={'Referer': referer} if referer else {},
                                    timeout=(15, 30), allow_redirects=False)
        html = 'text/html' in response.headers.get('Content-Type', '')
        if response.status_code in (401, 403, 429) or (html and 'cf-chl-' in response.text[:20000]):
            raise AccessBlocked(f'Access blocked or rate limited (HTTP {response.status_code})')
        response.raise_for_status()
        if response.is_redirect:
            raise RuntimeError('Unexpected redirect; inspect the public URL before continuing')
        return response

    def close(self):
        self.session.close()


def fetch_image_bytes(client, image_url, referer, known_hashes, min_side=256):
    """Download and validate image bytes in memory without writing to disk."""
    response = client.get(image_url, referer)
    blob = response.content
    if not response.headers.get('Content-Type', '').lower().startswith('image/'):
        raise ValueError('Response is not an image')
    if not 1024 <= len(blob) <= 20*1024*1024:
        raise ValueError('Image size is invalid')
    sha = hashlib.sha256(blob).hexdigest()
    if sha in known_hashes:
        raise ValueError('Duplicate photo content')
    with Image.open(io.BytesIO(blob)) as image:
        image.verify()
    with Image.open(io.BytesIO(blob)) as image:
        width, height, fmt = image.width, image.height, image.format
        if min(width, height) < min_side or fmt not in {'JPEG', 'PNG', 'WEBP'}:
            raise ValueError('Photo format or resolution is unsuitable')
    return blob, dict(sha256=sha, width=width, height=height, format=fmt, bytes=len(blob), image_url=image_url)


def write_validated_image(blob, dest, fmt):
    dest = Path(dest).with_suffix({'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}[fmt])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as stream:
        stream.write(blob)
    return dest


def download_image(client, image_url, dest, referer, known_hashes, min_side=256):
    """Save only decodable images; failed downloads never count as collected."""
    blob, info = fetch_image_bytes(client, image_url, referer, known_hashes, min_side=min_side)
    saved = write_validated_image(blob, dest, info['format'])
    return dict(image=saved.name, **info)


def make_contact_sheet(root, records):
    if not records:
        return
    cw, ch, per_page = 480, 345, 20
    pages = []
    for offset in range(0,len(records),per_page):
        subset = records[offset:offset+per_page]
        sheet = Image.new('RGB',(cw*2,((len(subset)+1)//2)*ch),'white')
        draw = ImageDraw.Draw(sheet)
        for i,row in enumerate(subset):
            x,y = i%2*cw,i//2*ch
            with Image.open(safe_path(root,row['image'])) as image:
                thumb = ImageOps.contain(ImageOps.exif_transpose(image).convert('RGB'),(cw-16,ch-45))
                sheet.paste(thumb,(x+(cw-thumb.width)//2,y+4))
            draw.text((x+8,y+ch-36),row['restaurant_id'],fill='black')
            draw.text((x+8,y+ch-20),row['restaurant_name'].encode('ascii','replace').decode()[:65],fill='black')
        filename = f'contact_sheet_{offset//per_page+1:03d}.jpg'
        sheet.save(root/filename,quality=90)
        pages.append(filename)
        if offset==0:
            sheet.save(root/'contact_sheet.jpg',quality=90)
    dump_json(root/'contact_sheets.json',{'page_size':per_page,'images':len(records),'pages':pages})


def run_crawler(output_dir=None, target=10, start_url=START_URL, delay=2,
                max_pages=None, max_restaurants=None, min_side=256, exclude_restaurants=(),
                photo_kind='interior', random_seed=None, exclude_hashes=(),
                quality_evaluator=None, max_photos_per_restaurant=None, status_path=None,
                extra_start_urls=(), exclusion_scope=None):
    if photo_kind not in PHOTO_KINDS:
        raise ValueError('Photo kind must be interior or menu')
    settings = PHOTO_KINDS[photo_kind]
    root = Path(output_dir if output_dir is not None else settings['directory']).resolve()
    if not root.is_relative_to(ROOT) or not 1 <= target <= 1000 or delay < 1:
        raise ValueError('Require workspace output, 1–1000 images and delay >=1 second')
    page_factor = 4 if photo_kind=='menu' else 2
    max_pages = max_pages if max_pages is not None else max(3, (target+19)//20*page_factor)
    max_restaurants = max_restaurants if max_restaurants is not None else max(50,target*3)
    if max_pages < 1 or max_restaurants < target:
        raise ValueError('Crawl limits must allow the requested target')
    if max_photos_per_restaurant is None:
        max_photos_per_restaurant = 8 if quality_evaluator is not None else 5
    if max_photos_per_restaurant < 1:
        raise ValueError('max_photos_per_restaurant must be >= 1')

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'manifest.json'
    state_path = root/'crawler_state.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    if previous.get('random_seed') is not None and random_seed!=previous['random_seed']:
        raise ValueError('Resume using the original random seed')
    rng=random.Random(random_seed) if random_seed is not None else None
    records = previous.get('images', [])
    if records and previous.get('photo_kind','interior')!=photo_kind:
        raise ValueError('Cannot mix menu and interior collections in one output directory')
    excluded = set(exclude_restaurants) | set(previous.get('excluded_restaurants', []))
    ids, hashes = set(), set(exclude_hashes)
    for row in records:
        if row['restaurant_id'] in excluded:
            raise ValueError('Quarantine an excluded restaurant before resuming the collection')
        if row['restaurant_id'] in ids or row['sha256'] in hashes or digest(safe_path(root, row['image'])) != row['sha256']:
            raise ValueError('Existing manifest contains duplicates or image checksum mismatches')
        ids.add(row['restaurant_id'])
        hashes.add(row['sha256'])

    prev_state = json.loads(state_path.read_text(encoding='utf-8')) if (quality_evaluator is not None and state_path.exists()) else {}
    inspected_prev = set(prev_state.get('inspected_restaurant_ids', []))
    visited_listings_prev = set(prev_state.get('visited_listings', []))
    skip_reasons = dict(prev_state.get('skip_reasons', {}))
    crop_totals = dict(prev_state.get('crop_totals', {'accepted': 0, 'closeup_32px': 0, 'borderline': 0, 'too_small': 0, 'invalid_geometry': 0}))
    successful_downloads = int(prev_state.get('successful_downloads', len(records)))
    attempted = int(prev_state.get('attempted_restaurants', len(inspected_prev)))
    pages_fetched = int(prev_state.get('pages_fetched', len(visited_listings_prev)))

    client = PublicClient(delay)
    errors = list(previous.get('errors', [])) if quality_evaluator is not None else []
    visited = set(ids) | excluded | inspected_prev
    listings = set(visited_listings_prev)
    blocked_error = None

    seed_urls = [start_url] + [u for u in extra_start_urls if u and u != start_url]
    if prev_state.get('listing_queue'):
        listing_queue = list(prev_state['listing_queue'])
    elif prev_state.get('next_listing_url'):
        listing_queue = [prev_state['next_listing_url']] + [u for u in seed_urls if u not in listings]
    else:
        listing_queue = list(seed_urls)
    current_listing_url = listing_queue[0] if listing_queue else None

    def _inc_skip(reason):
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    def _build_manifest_dict():
        note = ('Real menu-photo candidates; transcriptions and text polygons require review.' if photo_kind=='menu'
                else 'Real interior-photo candidates; not verified text-free negatives.')
        status = 'completed' if len(records)>=target else ('blocked' if blocked_error else 'partial')
        doc = dict(source='tabelog',photo_kind=photo_kind,requested_restaurants=target,
            downloaded_images=len(records), unique_restaurants=len(ids), one_image_per_restaurant=True,
            status=status, start_url=start_url,
            excluded_restaurants=sorted(excluded),
            random_seed=random_seed,sampling_policy='Seeded restaurant/photo shuffle within visited listing pages; not a uniform site-wide sample' if rng else 'Listing order',
            excluded_image_hashes=len(set(exclude_hashes)),
            user_agent=USER_AGENT, images=records, errors=errors,
            note=note+' No training labels assigned. Rights remain with respective owners; source availability is not a reuse license.')
        if quality_evaluator is not None:
            doc['quality_policy'] = quality_evaluator.policy_dict()
            doc['crop_totals'] = dict(crop_totals)
            doc['skip_reasons'] = dict(skip_reasons)
            doc['attempted_restaurants'] = attempted
            doc['successful_downloads'] = successful_downloads
            if exclusion_scope is not None:
                doc['exclusion_scope'] = exclusion_scope
            if blocked_error:
                doc['blocked_error'] = blocked_error
        return doc

    def checkpoint():
        doc = _build_manifest_dict()
        if quality_evaluator is not None:
            from .tabelog_quality import write_json_atomic
            write_json_atomic(manifest_path, doc)
            write_json_atomic(state_path, dict(
                updated_at_utc=datetime.now(timezone.utc).isoformat(),
                status=doc['status'],
                next_listing_url=current_listing_url,
                listing_queue=listing_queue,
                visited_listings=sorted(listings),
                inspected_restaurant_ids=sorted(visited - excluded),
                attempted_restaurants=attempted,
                successful_downloads=successful_downloads,
                accepted_restaurants=len(records),
                pages_fetched=pages_fetched,
                crop_totals=crop_totals,
                skip_reasons=skip_reasons,
                blocked_error=blocked_error,
            ))
            if status_path is not None:
                write_json_atomic(Path(status_path), dict(
                    updated_at_utc=datetime.now(timezone.utc).isoformat(),
                    status=doc['status'],
                    dataset_root=str(root),
                    requested_restaurants=target,
                    attempted_restaurants=attempted,
                    successful_downloads=successful_downloads,
                    accepted_restaurants=len(records),
                    pages_fetched=pages_fetched,
                    crop_totals=crop_totals,
                    skip_reasons=skip_reasons,
                    current_listing_url=current_listing_url,
                    blocked_error=blocked_error,
                    error_count=len(errors),
                ))
        else:
            dump_json(manifest_path, doc)

    try:
        if len(records) < target:
            robots = RobotFileParser()
            robots.parse(client.get('https://tabelog.com/robots.txt').text.splitlines())
            while listing_queue and pages_fetched < max_pages and len(records) < target and attempted < max_restaurants:
                listing_url = listing_queue.pop(0)
                if not listing_url or listing_url in listings:
                    continue
                current_listing_url = listing_url
                if not robots.can_fetch(USER_AGENT, listing_url):
                    raise AccessBlocked('Listing URL disallowed by robots.txt')
                listings.add(listing_url)
                pages_fetched += 1
                listing = TabelogHTML(listing_url)
                listing.feed(client.get(listing_url).text)
                if listing.next_url and listing.next_url not in listings:
                    listing_queue.insert(0, listing.next_url)
                current_listing_url = listing_queue[0] if listing_queue else None
                restaurants = listing.restaurants()
                if rng:rng.shuffle(restaurants)
                print(f'Listing {pages_fetched}: {len(restaurants)} restaurants ({listing_url})', flush=True)
                if not restaurants:
                    errors.append(dict(listing_url=listing_url, error='No restaurant links; inspect page layout'))
                    _inc_skip('empty_listing_page')
                if quality_evaluator is not None:
                    checkpoint()
                for restaurant in restaurants:
                    if len(records)>=target or attempted>=max_restaurants:
                        break
                    rid = restaurant['restaurant_id']
                    if rid in visited:
                        continue
                    visited.add(rid)
                    attempted+=1
                    gallery_url = urljoin(restaurant['restaurant_url'],settings['subpath'])
                    try:
                        if not robots.can_fetch(USER_AGENT, gallery_url):
                            raise AccessBlocked('Gallery URL disallowed by robots.txt')
                        gallery = TabelogHTML(gallery_url,photo_kind)
                        gallery.feed(client.get(gallery_url).text)
                        if not gallery.photos:
                            errors.append(dict(restaurant_id=rid,error=f'No {photo_kind} photographs'))
                            _inc_skip(f'no_{photo_kind}_photos')
                            if quality_evaluator is not None:
                                checkpoint()
                            continue
                        photos = gallery.photos
                        if rng:
                            rng.shuffle(photos)
                        elif photo_kind=='menu':
                            # Favor explicit menu captions over generic promotional photos.
                            def menu_score(photo):
                                caption = photo['alt'].casefold()
                                return sum(word in caption for word in ('menu','メニュー','お品書き','献立'))
                            photos = sorted(photos,key=menu_score,reverse=True)
                        accepted_this_venue = False
                        venue_reject_reasons = []
                        for photo_idx, photo in enumerate(photos[:max_photos_per_restaurant], start=1):
                            if quality_evaluator is None:
                                try:
                                    info = download_image(client,photo['image_url'],root/f'{settings["prefix"]}_{rid}.jpg',gallery_url,hashes,min_side)
                                except AccessBlocked:
                                    raise
                                except (requests.RequestException, ValueError, OSError) as error:
                                    errors.append(dict(restaurant_id=rid, error=type(error).__name__))
                                    continue
                                row = dict(**restaurant, **info, gallery_url=gallery_url,
                                    photo_id=photo['photo_id'], source_caption=photo['alt'],
                                    collected_at_utc=datetime.now(timezone.utc).isoformat(), source='real_web_photo',
                                    category=settings['category'],photo_kind=photo_kind,reviewed=False,
                                    license='not_established', split=None)
                                if photo_kind=='menu':
                                    row.update(text=None,annotations=[],transcription_verified=False,
                                               review_status='awaiting_menu_and_transcription_review')
                                else:
                                    row['negative_training_approved'] = False
                                dump_json(root/f'{settings["prefix"]}_{rid}.json',row)
                                records.append(row)
                                ids.add(rid)
                                hashes.add(info['sha256'])
                                successful_downloads += 1
                                checkpoint()
                                print(f'Saved {len(records)}/{target}: {rid} ({info["width"]}x{info["height"]})', flush=True)
                                accepted_this_venue = True
                                break
                            else:
                                from .tabelog_quality import append_jsonl, save_accepted_photo_crops, write_json_atomic
                                try:
                                    blob, info = fetch_image_bytes(client, photo['image_url'], gallery_url, hashes, min_side=min_side)
                                except AccessBlocked:
                                    raise
                                except (requests.RequestException, ValueError, OSError) as error:
                                    reason = str(error) if isinstance(error, ValueError) else type(error).__name__
                                    errors.append(dict(restaurant_id=rid, photo_index=photo_idx, image_url=photo['image_url'], error=reason))
                                    venue_reject_reasons.append(f'download:{reason}')
                                    continue
                                successful_downloads += 1
                                ev = quality_evaluator.evaluate_bytes(blob)
                                if not ev['photo_accepted']:
                                    venue_reject_reasons.append(ev['decision_reason'])
                                    append_jsonl(root/'rejected_candidates.jsonl', [dict(
                                        restaurant_id=rid, restaurant_url=restaurant['restaurant_url'],
                                        gallery_url=gallery_url, image_url=photo['image_url'],
                                        photo_index=photo_idx, sha256=info['sha256'],
                                        image_meta=ev['image_meta'],
                                        decision_reason=ev['decision_reason'],
                                        raw_detection_count=ev['raw_detection_count'],
                                        counts=ev['counts'],
                                        crop_summaries=[{k: v for k, v in c.items() if k not in ('core_polygon', 'crop_polygon')} for c in ev['crop_records']],
                                    )])
                                    continue

                                saved_path = write_validated_image(blob, root/f'{settings["prefix"]}_{rid}.jpg', info['format'])
                                info_saved = dict(image=saved_path.name, **info)
                                info_saved['raw_width'] = ev['image_meta']['raw_width']
                                info_saved['raw_height'] = ev['image_meta']['raw_height']
                                info_saved['exif_orientation'] = ev['image_meta']['exif_orientation']
                                info_saved['exif_transposed'] = ev['image_meta']['exif_transposed']
                                info_saved['width'] = ev['image_meta']['width']
                                info_saved['height'] = ev['image_meta']['height']

                                source_meta = dict(restaurant_url=restaurant['restaurant_url'], gallery_url=gallery_url, image_url=photo['image_url'])
                                crop_rows = save_accepted_photo_crops(root, rid, saved_path.name, info['sha256'], source_meta, ev)
                                append_jsonl(root/'crops_manifest.jsonl', crop_rows)
                                for ck in ('accepted', 'closeup_32px', 'borderline', 'too_small', 'invalid_geometry'):
                                    crop_totals[ck] = crop_totals.get(ck, 0) + int(ev['counts'].get(ck, 0))

                                row = dict(**restaurant, **info_saved, gallery_url=gallery_url,
                                    photo_id=photo['photo_id'], source_caption=photo['alt'],
                                    photo_candidate_index=photo_idx,
                                    collected_at_utc=datetime.now(timezone.utc).isoformat(), source='real_web_photo',
                                    category=settings['category'], photo_kind=photo_kind, reviewed=False,
                                    license='not_established', split=None,
                                    text=None, annotations=[], transcription_verified=False,
                                    review_status='awaiting_menu_and_transcription_review',
                                    quality=dict(
                                        decision_reason=ev['decision_reason'],
                                        raw_detection_count=ev['raw_detection_count'],
                                        processed_line_count=ev['processed_line_count'],
                                        counts=ev['counts'],
                                        saved_accepted_crops=[c['saved_crop_relpath'] for c in crop_rows if c['category']=='accepted' and c.get('saved_crop_relpath')],
                                        saved_borderline_crops=[c['saved_crop_relpath'] for c in crop_rows if c['category']=='borderline' and c.get('saved_crop_relpath')],
                                    ))
                                write_json_atomic(root/f'{settings["prefix"]}_{rid}.json', row)
                                records.append(row)
                                ids.add(rid)
                                hashes.add(info['sha256'])
                                checkpoint()
                                print(
                                    f'Accepted {len(records)}/{target}: {rid} ({info_saved["width"]}x{info_saved["height"]}, '
                                    f'acc={ev["counts"]["accepted"]}, cu32={ev["counts"]["closeup_32px"]}, bor={ev["counts"]["borderline"]}, '
                                    f'cand={photo_idx}/{min(len(photos), max_photos_per_restaurant)})',
                                    flush=True,
                                )
                                accepted_this_venue = True
                                break
                        if quality_evaluator is not None and not accepted_this_venue:
                            _inc_skip(venue_reject_reasons[-1] if venue_reject_reasons else 'quality_rejected')
                            checkpoint()
                    except AccessBlocked:
                        raise
                    except (requests.RequestException, ValueError) as error:
                        errors.append(dict(restaurant_id=rid, error=type(error).__name__))
                        _inc_skip(f'venue_error:{type(error).__name__}')
                        if quality_evaluator is not None:
                            checkpoint()
                if attempted>=max_restaurants:
                    break
    except AccessBlocked as error:
        blocked_error = str(error)
        errors.append(dict(error=blocked_error, access_blocked=True))
    except (requests.RequestException, RuntimeError) as error:
        errors.append(dict(error=str(error) if isinstance(error, RuntimeError) else type(error).__name__))
    finally:
        client.close()
        checkpoint()
        make_contact_sheet(root, records)
        if quality_evaluator is not None:
            from .tabelog_quality import build_review_site
            build_review_site(root, _build_manifest_dict())
    print(json.dumps(dict(downloaded=len(records), unique_restaurants=len(ids), attempted=attempted, blocked=bool(blocked_error), output=str(root))))
    return 0 if len(records)>=target else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photo-kind',choices=sorted(PHOTO_KINDS),default='interior')
    parser.add_argument('--output-dir',type=Path,help='Defaults to data/raw/tabelog_interiors or data/raw/tabelog_menus for the selected kind')
    parser.add_argument('--target', type=int, default=10)
    parser.add_argument('--start-url', default=START_URL)
    parser.add_argument('--extra-start-url', action='append', default=[], help='Fallback listing URLs if the first listing ends before target')
    parser.add_argument('--delay', type=float, default=2)
    parser.add_argument('--max-pages', type=int, help='Listing-page limit; default scales with target')
    parser.add_argument('--max-restaurants', type=int, help='Venue limit; default is at least three times target')
    parser.add_argument('--min-side', type=int, default=256)
    parser.add_argument('--exclude-restaurant', action='append', default=[], help='Skip a restaurant ID rejected during visual review; repeatable')
    parser.add_argument('--random-seed',type=int,help='Reproducibly shuffle restaurant and gallery photo candidates')
    parser.add_argument('--exclude-manifest',type=Path,action='append',default=[],help='Exclude restaurant IDs and hashes from an existing collection or Stage 2 manifest')
    parser.add_argument('--yolo-quality', action='store_true', help='Enable YOLO V5 native-resolution menu quality filtering and static review builder')
    parser.add_argument('--yolo-weights', type=Path, help='Override YOLO V5 OBB checkpoint path')
    parser.add_argument('--yolo-conf', type=float, default=0.35, help='YOLO OBB confidence threshold (default 0.35)')
    parser.add_argument('--max-photos-per-restaurant', type=int, help='Bounded candidate photos to inspect per restaurant')
    parser.add_argument('--status-path', type=Path, help='Optional atomic progress_status.json path')
    args = parser.parse_args()
    excluded=set(args.exclude_restaurant);hashes=set()
    for path in args.exclude_manifest:
        doc=json.loads(path.read_text(encoding='utf-8'))
        for row in doc.get('records',doc.get('images',[])):
            info=row.get('source_info') or row
            for key in ['restaurant_id','background_group']:
                if info.get(key):excluded.add(str(info[key]))
            for item in [row,info]:
                for key in ['sha256','background_sha256','source_image_sha256']:
                    if item.get(key):hashes.add(item[key])
    evaluator = None
    if args.yolo_quality:
        from .tabelog_quality import DEFAULT_YOLO_WEIGHTS, YoloMenuQualityEvaluator
        evaluator = YoloMenuQualityEvaluator(
            weights_path=args.yolo_weights or DEFAULT_YOLO_WEIGHTS,
            confidence=args.yolo_conf,
        )
    return run_crawler(args.output_dir,args.target,args.start_url,args.delay,args.max_pages,
                       args.max_restaurants,args.min_side,excluded,args.photo_kind,args.random_seed,hashes,
                       quality_evaluator=evaluator, max_photos_per_restaurant=args.max_photos_per_restaurant,
                       status_path=args.status_path, extra_start_urls=args.extra_start_url,
                       exclusion_scope={'exclude_manifests': [str(p) for p in args.exclude_manifest],
                                        'excluded_restaurant_ids': len(excluded), 'excluded_hashes': len(hashes)})
