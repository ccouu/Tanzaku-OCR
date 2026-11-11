import hashlib
import io
import json

from PIL import Image
import pytest

from src.data import tabelog_collection as crawler


def jpeg_bytes(color='navy'):
    stream = io.BytesIO()
    Image.new('RGB', (640, 480), color).save(stream, format='JPEG')
    return stream.getvalue()


class Response:
    def __init__(self, content, content_type):
        self.content = content
        self.headers = {'Content-Type': content_type}
        self.text = content.decode('utf8') if content_type == 'text/html' else ''


def test_parser_deduplicates_restaurants_and_uses_gallery_full_size_link():
    parser = crawler.TabelogHTML(crawler.START_URL)
    parser.feed('''<a class="list-rst__rst-name-target" href="/en/tokyo/A1301/A130103/13292459/"><b>Restaurant A</b></a>
    <a class="cpy-rst-name" href="/en/tokyo/A1301/A130103/13292459/?ref=x">Duplicate</a>
    <a rel="next" href="?PG=2">Next</a>
    <a href="https://tblg.k-img.com/restaurant/images/Rvw/1/640x640_rect_photo.jpg" data-id="12">
    <img class="rstdtl-thumb-list__img" src="https://tblg.k-img.com/restaurant/images/Rvw/1/150x150_square_photo.jpg" alt="Interior"></a>
    <img src="https://maps.googleapis.com/maps/api/staticmap?secret=not-used">''')
    assert len(parser.restaurants()) == 1
    assert parser.restaurants()[0]['restaurant_name'] == 'Restaurant A'
    assert parser.next_url == crawler.START_URL + '?PG=2'
    assert len(parser.photos) == 1
    assert '/640x640_rect_' in parser.photos[0]['image_url']
    assert parser.photos[0]['photo_id'] == '12'
    assert not crawler.profile_url('https://example.com/en/tokyo/A1301/A130103/13292459/')


def test_failed_or_duplicate_download_never_saves_file(tmp_path):
    class Client:
        def get(self, *_):
            return Response(b'<html>blocked</html>', 'text/html')
    path = tmp_path/'photo.jpg'
    with pytest.raises(ValueError, match='not an image'):
        crawler.download_image(Client(), 'https://tblg.k-img.com/photo.jpg', path, '', set())
    assert not path.exists()
    blob = jpeg_bytes()
    class ImageClient:
        def get(self, *_):
            return Response(blob, 'image/jpeg')
    with pytest.raises(ValueError, match='Duplicate'):
        crawler.download_image(ImageClient(), 'https://tblg.k-img.com/photo.jpg', path, '', {hashlib.sha256(blob).hexdigest()})
    assert not path.exists()


@pytest.mark.parametrize('excluded',[(),tuple(str(99000000+i) for i in range(100))])
def test_collection_counts_only_successes_and_one_per_restaurant(tmp_path, monkeypatch, excluded):
    photo_requests = []
    class Client:
        def __init__(self, delay):
            pass
        def close(self):
            pass
        def get(self, url, referer=None):
            if url.endswith('/robots.txt'):
                return Response(b'User-agent: *\nDisallow: /private/\n', 'text/html')
            if '/rstLst/' in url:
                html = ''.join(f'<a class="cpy-rst-name" href="https://tabelog.com/en/tokyo/A1301/A130103/{rid}/">Restaurant {rid}</a>' for rid in ('13000001','13000002'))
                return Response(html.encode(), 'text/html')
            if '/dtlphotolst/' in url:
                rid = url.split('/')[7]
                html = ''.join(f'<a href="https://tblg.k-img.com/restaurant/images/Rvw/1/{rid}_{i}.jpg"><img class="rstdtl-thumb-list__img" alt="Interior"></a>' for i in range(3))
                return Response(html.encode(), 'text/html')
            photo_requests.append(url)
            if url.endswith('_0.jpg'):
                return Response(b'error', 'text/html')
            return Response(jpeg_bytes('red' if '13000001' in url else 'blue'), 'image/jpeg')
    monkeypatch.setattr(crawler, 'PublicClient', Client)
    assert crawler.run_crawler(tmp_path, target=2, delay=1,exclude_restaurants=excluded) == 0
    manifest = json.loads((tmp_path/'manifest.json').read_text(encoding='utf8'))
    assert manifest['downloaded_images'] == manifest['unique_restaurants'] == 2
    assert len(list(tmp_path.glob('interior_*.jpg'))) == 2
    assert len(photo_requests) == 4  # one rejected and one successful per venue
    assert all(not url.endswith('_2.jpg') for url in photo_requests)
    assert all(r['negative_training_approved'] is False for r in manifest['images'])
    # Resume validates existing files and issues no further network requests.
    before = list(photo_requests)
    assert crawler.run_crawler(tmp_path, target=2, delay=1,exclude_restaurants=excluded) == 0
    assert photo_requests == before


def test_two_hundred_target_paginates_and_writes_bounded_contact_sheets(tmp_path, monkeypatch):
    listing_requests = []
    class Client:
        def __init__(self, delay):
            pass
        def close(self):
            pass
        def get(self, url, referer=None):
            if url.endswith('/robots.txt'):
                return Response(b'User-agent: *\nDisallow: /private/\n','text/html')
            if '/rstLst/' in url:
                listing_requests.append(url)
                page = len(listing_requests)
                html = ''.join(f'<a class="cpy-rst-name" href="https://tabelog.com/en/tokyo/A1301/A130103/{13000000+i}/">Restaurant {i}</a>' for i in range((page-1)*20,page*20))
                html += f'<a rel="next" href="https://tabelog.com/en/tokyo/rstLst/izakaya/{page+1}/">Next</a>'
                return Response(html.encode(),'text/html')
            if '/dtlphotolst/' in url:
                rid = url.split('/')[7]
                html = f'<a href="https://tblg.k-img.com/restaurant/images/Rvw/1/{rid}.jpg"><img class="rstdtl-thumb-list__img"></a>'
                return Response(html.encode(),'text/html')
            index = int(url.rsplit('/',1)[1].split('.')[0])-13000000
            return Response(jpeg_bytes((index,255-index,index*13%256)),'image/jpeg')
    monkeypatch.setattr(crawler,'PublicClient',Client)
    assert crawler.run_crawler(tmp_path,target=200,delay=1)==0
    manifest = json.loads((tmp_path/'manifest.json').read_text(encoding='utf8'))
    sheets = json.loads((tmp_path/'contact_sheets.json').read_text())
    assert manifest['downloaded_images']==manifest['unique_restaurants']==200
    assert len({r['sha256'] for r in manifest['images']})==200
    assert len(listing_requests)==10
    assert len(sheets['pages'])==10
    for page in sheets['pages']:
        with Image.open(tmp_path/page) as image:
            assert image.size==(960,3450)
    from scripts.data.verify_tabelog_collection import verify
    assert verify(tmp_path,target=200)['status']=='passed'
    corrupted = tmp_path/manifest['images'][0]['image']
    with corrupted.open('ab') as stream:
        stream.write(b'corrupt-image-content')
    with pytest.raises(ValueError,match='checksum'):
        verify(tmp_path,target=200)


@pytest.mark.parametrize('kind',['interior','menu'])
def test_rejected_photo_is_quarantined_and_excluded_on_resume(tmp_path,kind):
    from scripts.data.review_tabelog_collection import reject
    from src.common import dump_json
    rows = []
    for index,rid in enumerate(('13000001','13000002')):
        path = tmp_path/f'interior_{rid}.jpg'
        path.write_bytes(jpeg_bytes('red' if index==0 else 'blue'))
        row = dict(restaurant_id=rid,image=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        dump_json(path.with_suffix('.json'),row)
        rows.append(row)
    dump_json(tmp_path/'manifest.json',dict(images=rows,photo_kind=kind,downloaded_images=2,unique_restaurants=2,
        requested_restaurants=2,status='completed',excluded_restaurants=[]))
    reject(tmp_path,{'13000001':'Food close-up, not a room interior.'})
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['downloaded_images']==1 and manifest['status']=='partial'
    assert manifest['excluded_restaurants']==['13000001']
    assert not (tmp_path/'interior_13000001.jpg').exists()
    rejected = json.loads((tmp_path/'rejected/interior_13000001.json').read_text())
    assert rejected[kind+'_visual_review']=='failed'
    assert rejected['reviewed'] is True
    assert (tmp_path/'interior_13000002.jpg').is_file()


def test_menu_parser_uses_menu_photo_class_and_full_size_anchor():
    parser = crawler.TabelogHTML('https://tabelog.com/en/aichi/A2306/A230601/23075911/dtlmenu/photo/','menu')
    parser.feed('''<a class="rstdtl-photo-list__target" data-id="391034511" href="https://tblg.k-img.com/restaurant/images/Rvw/391034/640x640_rect_menu.jpg">
    <img class="rstdtl-photo-list__img" src="https://tblg.k-img.com/restaurant/images/Rvw/391034/320x320_rect_menu.jpg" alt="Food Menu"></a>
    <a href="https://tblg.k-img.com/restaurant/images/Rvw/1/640x640_rect_food.jpg"><img class="rstdtl-thumb-list__img" alt="Other photo"></a>''')
    assert len(parser.photos)==1
    assert '640x640_rect_menu' in parser.photos[0]['image_url']
    assert parser.photos[0]['photo_id']=='391034511'
    assert parser.photos[0]['alt']=='Food Menu'


def test_menu_collection_skips_empty_venues_and_preserves_no_fake_text(tmp_path,monkeypatch):
    gallery_urls = []
    class Client:
        def __init__(self,delay):
            pass
        def close(self):
            pass
        def get(self,url,referer=None):
            if url.endswith('/robots.txt'):
                return Response(b'User-agent: *\nDisallow: /private/\n','text/html')
            if '/rstLst/' in url:
                html = ''.join(f'<a class="cpy-rst-name" href="https://tabelog.com/en/tokyo/A1301/A130103/{rid}/">Restaurant {rid}</a>' for rid in ('13000001','13000002','13000003'))
                return Response(html.encode(),'text/html')
            if '/dtlmenu/photo/' in url:
                gallery_urls.append(url)
                rid = url.split('/')[7]
                if rid=='13000001':
                    return Response(b'<html>No menu photos yet.</html>','text/html')
                html = f'<a data-id="{rid}" href="https://tblg.k-img.com/restaurant/images/Rvw/1/{rid}.jpg"><img class="rstdtl-photo-list__img" alt="Menu"></a>'
                return Response(html.encode(),'text/html')
            return Response(jpeg_bytes('red' if '13000002' in url else 'blue'),'image/jpeg')
    monkeypatch.setattr(crawler,'PublicClient',Client)
    assert crawler.run_crawler(tmp_path,target=2,delay=1,photo_kind='menu')==0
    assert len(gallery_urls)==3 and all(u.endswith('/dtlmenu/photo/') for u in gallery_urls)
    assert len(list(tmp_path.glob('menu_*.jpg')))==2
    manifest = json.loads((tmp_path/'manifest.json').read_text(encoding='utf8'))
    assert manifest['photo_kind']=='menu'
    assert manifest['unique_restaurants']==2
    for row in manifest['images']:
        assert row['category']=='restaurant_menu'
        assert row['text'] is None and row['annotations']==[] and row['transcription_verified'] is False
    from scripts.data.verify_tabelog_collection import verify
    assert verify(tmp_path,target=2)['photo_kind']=='menu'
    with pytest.raises(ValueError,match='Cannot mix'):
        crawler.run_crawler(tmp_path,target=2,delay=1,photo_kind='interior')
