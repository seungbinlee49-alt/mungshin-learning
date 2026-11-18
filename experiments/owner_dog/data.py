"""Read the supplied dataset and derive every descriptor without host caches."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


IMAGE_PATTERN = re.compile(r'(?:^|/)사진(\d+)/(사람|개)([123])\.png$', re.IGNORECASE)
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
INVISIBLE_EYES = {(70, 'dog', 3), (147, 'dog', 3)}
PARTIAL_EYES = {(158, 'dog', 2), (180, 'dog', 2)}


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _normal_path(name):
    value = unicodedata.normalize('NFC', str(name).replace('\\', '/'))
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or re.match(r'^[A-Za-z]:', value):
        raise ValueError(f'Unsafe dataset member path: {name!r}')
    return value


def _zip_name(info):
    """Try CP949 recovery only for a legacy ZIP entry lacking UTF-8 metadata."""
    original = _normal_path(info.filename)
    if IMAGE_PATTERN.search(original):
        return original
    if not info.flag_bits & 0x800:
        try:
            recovered = _normal_path(info.filename.encode('cp437').decode('cp949'))
            if IMAGE_PATTERN.search(recovered):
                return recovered
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return original


class DatasetSource:
    """Index a ZIP or a directory; ZIP contents are never extracted."""

    def __init__(self, dataset, expected_pairs=200):
        self.path = Path(dataset).expanduser().resolve()
        self.expected_pairs = int(expected_pairs)
        if self.expected_pairs < 1:
            raise ValueError('expected_pairs must be positive')
        self.archive = None
        self.records = []
        self._locations = {}
        if self.path.is_file():
            if not zipfile.is_zipfile(self.path):
                raise ValueError('Dataset file must be a ZIP archive')
            self.archive = zipfile.ZipFile(self.path)
        elif not self.path.is_dir():
            raise FileNotFoundError(f'Dataset not found: {self.path}')
        try:
            self._index()
        except Exception:
            self.close()
            raise

    def _index(self):
        if self.archive is not None:
            entries = [(info, _zip_name(info)) for info in self.archive.infolist() if not info.is_dir()]
        else:
            entries = []
            for path in self.path.rglob('*'):
                if not path.is_file():
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(self.path):
                    raise ValueError(f'Dataset symlinks are not supported: {path.name}')
                entries.append((path, _normal_path(path.relative_to(self.path).as_posix())))
        extra_images = []
        for location, name in entries:
            if '__MACOSX' in PurePosixPath(name).parts or PurePosixPath(name).name.startswith('._'):
                continue
            match = IMAGE_PATTERN.search(name)
            if not match:
                if PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES:
                    extra_images.append(name)
                continue
            pair_id, label, photo_index = int(match[1]), match[2], int(match[3])
            role = 'person' if label == '사람' else 'dog'
            key = pair_id, role, photo_index
            if key in self._locations:
                raise ValueError(f'Duplicate dataset image: {name}')
            self._locations[key] = location
            self.records.append(dict(pair_id=pair_id, role=role, photo_index=photo_index,
                image_id=f'{role}_{photo_index:02d}', relative_image=name,
                zip_member=location.filename if self.archive is not None else None))
        expected = {(pair, role, view) for pair in range(1, self.expected_pairs + 1)
                    for role in ('person', 'dog') for view in (1, 2, 3)}
        actual = set(self._locations)
        missing, extra = sorted(expected-actual), sorted(actual-expected)
        if missing or extra or extra_images:
            raise ValueError('Dataset must contain exactly person1..3/dog1..3 for each numbered set. '
                f'Missing={missing[:6]}, unexpected={extra[:6]}, extra images={extra_images[:6]}')
        self.records.sort(key=lambda row: (row['pair_id'], row['role'], row['photo_index']))

    def read(self, key):
        location = self._locations[tuple(key)]
        return self.archive.read(location) if self.archive is not None else location.read_bytes()

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_eye_points(path, expected_pairs=200):
    """Validate the original screen-left/right annotation CSV without inference."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    fields = ('screen_left_eye_x', 'screen_left_eye_y', 'screen_right_eye_x', 'screen_right_eye_y')
    allowed_qc = {'ok', 'one_eye_not_visible', 'eye_occluded_by_fur', 'partial_occlusion_review'}
    for row in rows:
        try:
            pair, view = int(row['pair_id']), int(row['view_no'])
            role = {'human': 'person', 'dog': 'dog'}[row['species']]
            prefix = '사람' if role == 'person' else '개'
            points = np.array([float(row[field]) for field in fields], dtype=float).reshape(2, 2)
            if row['set_folder'] != f'사진{pair}' or row['image_file'] != f'{prefix}{view}.png':
                raise ValueError('Annotation folder/filename disagrees with its IDs')
            if not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= 256):
                raise ValueError('Eye coordinates must be within the original 256x256 image')
            if points[0, 0] >= points[1, 0] or row['qc_note'] not in allowed_qc:
                raise ValueError('Invalid left/right eye order or QC label')
            key = pair, role, view
            if key in result:
                raise ValueError('Duplicate eye annotation')
            result[key] = dict(left=points[0].tolist(), right=points[1].tolist(), qc_note=row['qc_note'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid eye CSV row {len(result)+2}: {exc}') from exc
    expected = {(pair, role, view) for pair in range(1, expected_pairs+1)
                for role in ('person', 'dog') for view in (1, 2, 3)}
    if set(result) != expected:
        raise ValueError(f'Eye CSV does not match all {expected_pairs*6} dataset images')
    flagged = {key for key, row in result.items() if row['qc_note'] in {'one_eye_not_visible', 'eye_occluded_by_fur'}}
    intended = {key for key in INVISIBLE_EYES if key[0] <= expected_pairs}
    if flagged != intended:
        raise ValueError('Invisible-eye flags differ from the shared experiment protocol')
    partial = {key for key, row in result.items() if row['qc_note'] == 'partial_occlusion_review'}
    if partial != {key for key in PARTIAL_EYES if key[0] <= expected_pairs}:
        raise ValueError('Partial-eye flags differ from the shared experiment protocol')
    return result


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def masked_mean(values, valid):
    if values.shape[:2] != valid.shape or np.any(valid.sum(axis=1) == 0):
        raise ValueError('Every set/species needs at least one usable image')
    if not np.isfinite(values[valid]).all():
        raise ValueError('Usable descriptors must be finite')
    means = np.where(valid[..., None], values, 0.).sum(axis=1) / valid.sum(axis=1)[:, None]
    return normalize_rows(means)


def _photo_features(per_image, orders, count, path):
    if count == 3:
        return normalize_rows(per_image.mean(axis=1))
    selected = np.roll(orders, -path, axis=1)[:, :count]
    return normalize_rows(per_image[np.arange(len(per_image))[:, None], selected].mean(axis=1))


def prepare_features(dataset, output, eye_csv, photo_selection, include_static=True, expected_pairs=200):
    """Build raw and set descriptors in a new cache, returning paths and arrays.

    The caller may create the output directory first. Existing feature/manifest
    files are refused. No source image or archive is modified or extracted.
    """
    from .features import extract_variants
    if eye_csv is not None:
        from .eye_features import align_eye_crop, crop_lbp
    output = Path(output)
    caches = [output/'features/per_image.npz', output/'features/set_features.npz',
              output/'features/eye_per_image.npz', output/'image_manifest.json', output/'input_audit.json']
    if any(path.exists() for path in caches):
        raise FileExistsError('Feature output already exists; choose a new output directory')
    points = read_eye_points(eye_csv, expected_pairs) if eye_csv is not None else None
    orders = {role: np.array(photo_selection['orders'][role], dtype=int) for role in ('person', 'dog')}
    if any(array.shape != (expected_pairs, 3) or not np.all(np.sort(array, axis=1) == [0, 1, 2])
           for array in orders.values()):
        raise ValueError('Photo selection orders must be three-image permutations per set/species')
    output.mkdir(parents=True, exist_ok=True)
    (output/'features').mkdir(exist_ok=True)
    dataset = Path(dataset).expanduser().resolve()
    archive_hash = _digest(dataset) if dataset.is_file() else None
    csv_hash = _digest(eye_csv) if eye_csv is not None else None
    per_image = {}
    eye = {role: np.full((expected_pairs, 3, 160), np.nan) for role in ('person', 'dog')}
    valid = {role: np.ones((expected_pairs, 3), dtype=bool) for role in ('person', 'dog')}
    crops = {role: np.empty((expected_pairs, 3, 24, 32, 3), dtype=np.uint8) for role in ('person', 'dog')}
    manifest = []
    with DatasetSource(dataset, expected_pairs) as source:
        for index, row in enumerate(source.records, 1):
            key = row['pair_id'], row['role'], row['photo_index']
            raw = source.read(key)
            try:
                with Image.open(io.BytesIO(raw)) as opened:
                    opened.load()
                    original_mode = opened.mode
                    image = opened.convert('RGBA')
            except (OSError, ValueError) as exc:
                raise ValueError(f'Cannot decode {row["relative_image"]}') from exc
            if points is not None and image.size != (256, 256):
                raise ValueError('Shared eye coordinates require the original 256x256 images')
            record = dict(row, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                          width=image.width, height=image.height, mode=original_mode, converted_mode='RGBA')
            manifest.append(record)
            for name, vector in extract_variants(image).items():
                cache_key = f'{name}__{key[1]}'
                if cache_key not in per_image:
                    per_image[cache_key] = np.full((expected_pairs, 3, len(vector)), np.nan)
                per_image[cache_key][key[0]-1, key[2]-1] = vector
            if points is not None:
                point = points[key]
                crop = align_eye_crop(image, point['left'], point['right'])
                crops[key[1]][key[0]-1, key[2]-1] = np.asarray(crop)
                usable = key not in INVISIBLE_EYES
                valid[key[1]][key[0]-1, key[2]-1] = usable
                if usable:
                    eye[key[1]][key[0]-1, key[2]-1] = crop_lbp(crop)
            if index % 300 == 0:
                print(f'Features {index}/{expected_pairs*6}', flush=True)
    if not all(np.isfinite(value).all() for value in per_image.values()):
        raise RuntimeError('One or more per-image descriptors are missing or nonfinite')
    set_features = {}
    for case in photo_selection['cases']:
        if case.get('feature_group', 'enhanced') != 'enhanced':
            continue
        for role in ('person', 'dog'):
            set_features[f'{case["id"]}__{role}'] = _photo_features(
                per_image[f'enhanced__{role}'], orders[role], int(case['photos']), int(case['path']))
    if points is not None:
        for role in ('person', 'dog'):
            eyes = masked_mean(eye[role], valid[role])
            whole = masked_mean(per_image[f'enhanced__{role}'], valid[role])
            set_features[f'eye_only_n3__{role}'] = eyes
            set_features[f'matched_global_n3__{role}'] = whole
            set_features[f'global_eye_n3__{role}'] = normalize_rows(
                np.concatenate([normalize_rows(whole), normalize_rows(eyes)], axis=1))
        raw_eyes = {f'eye__{role}': eye[role] for role in ('person', 'dog')}
        raw_eyes.update({f'valid__{role}': valid[role] for role in ('person', 'dog')})
        raw_eyes.update({f'crop__{role}': crops[role] for role in ('person', 'dog')})
        np.savez_compressed(output/'features/eye_per_image.npz', **raw_eyes)
    if include_static:
        for role in ('person', 'dog'):
            for name in ('hog_lbp', 'hog_only', 'lbp_only', 'enhanced', 'gabor_lbp'):
                set_features[f'static_{name}__{role}'] = normalize_rows(per_image[f'{name}__{role}'].mean(axis=1))
    if not all(np.isfinite(value).all() for value in set_features.values()):
        raise RuntimeError('Set feature preparation failed')
    np.savez_compressed(output/'features/per_image.npz', **per_image)
    np.savez_compressed(output/'features/set_features.npz', **set_features)
    if archive_hash is not None and _digest(dataset) != archive_hash:
        raise RuntimeError('Source archive changed during feature extraction')
    if csv_hash is not None and _digest(eye_csv) != csv_hash:
        raise RuntimeError('Eye coordinate CSV changed during feature extraction')
    audit = dict(status='PASS',sets=expected_pairs,images=len(manifest),
        person_images=expected_pairs*3,dog_images=expected_pairs*3,
        dataset_kind='zip' if dataset.is_file() else 'directory',archive_sha256=archive_hash,
        eye_csv_sha256=csv_hash,eye_features_enabled=points is not None,
        usable_person_eye_images=int(valid['person'].sum()) if points is not None else None,
        usable_dog_eye_images=int(valid['dog'].sum()) if points is not None else None,
        excluded_eye_images=[list(key) for key in sorted(INVISIBLE_EYES) if key[0] <= expected_pairs] if points is not None else [],
        partial_eye_images_retained=[list(key) for key in sorted(PARTIAL_EYES) if key[0] <= expected_pairs] if points is not None else [],
        source_images_modified=False,zip_extracted=False)
    _json_new(output/'image_manifest.json',dict(images=manifest,pairs=expected_pairs))
    _json_new(output/'input_audit.json',audit)
    return dict(set_features=set_features,manifest=manifest,audit=audit,
                feature_path=output/'features/set_features.npz',per_image_path=output/'features/per_image.npz')
