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
            for source in [anchor.get('href'), attrs.get('data-original'), attrs.get('data-lazy-src'), attrs.get('data-src'), attrs.get('src')]:
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


def download_image(client, image_url, dest, referer, known_hashes, min_side=256):
    """Save only decodable images; failed downloads never count as collected."""
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
    dest = Path(dest).with_suffix({'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}[fmt])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('xb') as stream:
        stream.write(blob)
    return dict(image=dest.name, sha256=sha, width=width, height=height,
                format=fmt, bytes=len(blob), image_url=image_url)


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
                photo_kind='interior', random_seed=None, exclude_hashes=()):
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
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root/'manifest.json'
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
    client, errors, visited, listings = PublicClient(delay), [], set(ids) | excluded, set()
    attempted=0

    def checkpoint():
        note = ('Real menu-photo candidates; transcriptions and text polygons require review.' if photo_kind=='menu'
                else 'Real interior-photo candidates; not verified text-free negatives.')
        dump_json(manifest_path, dict(source='tabelog',photo_kind=photo_kind,requested_restaurants=target,
            downloaded_images=len(records), unique_restaurants=len(ids), one_image_per_restaurant=True,
            status='completed' if len(records)>=target else 'partial', start_url=start_url,
            excluded_restaurants=sorted(excluded),
            random_seed=random_seed,sampling_policy='Seeded restaurant/photo shuffle within visited listing pages; not a uniform site-wide sample' if rng else 'Listing order',
            excluded_image_hashes=len(set(exclude_hashes)),
            user_agent=USER_AGENT, images=records, errors=errors,
            note=note+' No training labels assigned. Rights remain with respective owners; source availability is not a reuse license.'))

    try:
        if len(records) < target:
            robots = RobotFileParser()
            robots.parse(client.get('https://tabelog.com/robots.txt').text.splitlines())
            listing_url = start_url
            for page_number in range(max_pages):
                if not listing_url or listing_url in listings or len(records)>=target:
                    break
                if not robots.can_fetch(USER_AGENT, listing_url):
                    raise AccessBlocked('Listing URL disallowed by robots.txt')
                listings.add(listing_url)
                listing = TabelogHTML(listing_url)
                listing.feed(client.get(listing_url).text)
                restaurants = listing.restaurants()
                if rng:rng.shuffle(restaurants)
                print(f'Listing {page_number+1}: {len(restaurants)} restaurants', flush=True)
                if not restaurants:
                    errors.append(dict(error='No restaurant links; inspect page layout'))
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
                        photos = gallery.photos
                        if rng:
                            rng.shuffle(photos)
                        elif photo_kind=='menu':
                            # Favor explicit menu captions over generic promotional photos.
                            def menu_score(photo):
                                caption = photo['alt'].casefold()
                                return sum(word in caption for word in ('menu','メニュー','お品書き','献立'))
                            photos = sorted(photos,key=menu_score,reverse=True)
                        for photo in photos[:5]:
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
                            checkpoint()
                            print(f'Saved {len(records)}/{target}: {rid} ({info["width"]}x{info["height"]})', flush=True)
                            break
                    except AccessBlocked:
                        raise
                    except (requests.RequestException, ValueError) as error:
                        errors.append(dict(restaurant_id=rid, error=type(error).__name__))
                listing_url = listing.next_url
                if attempted>=max_restaurants:
                    break
    except (AccessBlocked, requests.RequestException, RuntimeError) as error:
        errors.append(dict(error=str(error) if isinstance(error, RuntimeError) else type(error).__name__))
    finally:
        client.close()
        checkpoint()
        make_contact_sheet(root, records)
    print(json.dumps(dict(downloaded=len(records), unique_restaurants=len(ids), output=str(root))))
    return 0 if len(records)>=target else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photo-kind',choices=sorted(PHOTO_KINDS),default='interior')
    parser.add_argument('--output-dir',type=Path,help='Defaults to data/raw/tabelog_interiors or data/raw/tabelog_menus for the selected kind')
    parser.add_argument('--target', type=int, default=10)
    parser.add_argument('--start-url', default=START_URL)
    parser.add_argument('--delay', type=float, default=2)
    parser.add_argument('--max-pages', type=int, help='Listing-page limit; default scales with target')
    parser.add_argument('--max-restaurants', type=int, help='Venue limit; default is at least three times target')
    parser.add_argument('--min-side', type=int, default=256)
    parser.add_argument('--exclude-restaurant', action='append', default=[], help='Skip a restaurant ID rejected during visual review; repeatable')
    parser.add_argument('--random-seed',type=int,help='Reproducibly shuffle restaurant and gallery photo candidates')
    parser.add_argument('--exclude-manifest',type=Path,action='append',default=[],help='Exclude restaurant IDs and hashes from an existing collection or Stage 2 manifest')
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
    return run_crawler(args.output_dir,args.target,args.start_url,args.delay,args.max_pages,
                       args.max_restaurants,args.min_side,excluded,args.photo_kind,args.random_seed,hashes)
