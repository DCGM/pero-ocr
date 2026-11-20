"""The line cropper (pero_ocr.core.crop_engine.EngineLineCropper): it cuts a text line out of the page along its
baseline and straightens it into a crop of fixed height, the input of every OCR engine.

Geometry is tested with linear intensity fields, which bilinear interpolation reproduces exactly: a field that grows
along the normal of the baseline must come out as crop rows equal to the sampled heights, a field that grows along the
baseline as linearly increasing crop columns. Curved baselines are tested with a thin bright stroke drawn along the
baseline, which a correct crop turns into a horizontal line at the baseline row.

Known limits of the cropper are pinned by strict xfail tests that describe the wanted behaviour.

The [LINE_CROPPER] stage around the cropper is tested in test/stages/test_cropper_stage.py."""
import cv2
import numpy as np
import pytest

from pero_ocr.core.crop_engine import EngineLineCropper

PAGE_SHAPE = (600, 800)
PRODUCTION = dict(line_height=48, poly=2, scale=1.15)  # LINE_HEIGHT, INTERP, LINE_SCALE of the production engines
ASCENDER, DESCENDER = 20, 8


def straight_baseline(angle_deg, length=300, centre=(400, 300)):
    """Integer endpoints (the cropper truncates baseline points to integers), and the exact angle and length."""
    a = np.deg2rad(angle_deg)
    p0 = np.round([centre[0] - length / 2 * np.cos(a), centre[1] - length / 2 * np.sin(a)])
    p1 = np.round(p0 + length * np.array([np.cos(a), np.sin(a)]))
    return np.stack([p0, p1]), np.arctan2(p1[1] - p0[1], p1[0] - p0[0]), np.hypot(*(p1 - p0))


def linear_fields_page(p0, angle, length):
    """Channel 0 grows by 1 per pixel along the normal of the baseline (128 on it, larger below the text),
    channel 1 grows from 28 to 228 along the baseline."""
    yy, xx = np.mgrid[0:PAGE_SHAPE[0], 0:PAGE_SHAPE[1]].astype(np.float64)
    normal = np.array([-np.sin(angle), np.cos(angle)])
    along = np.array([np.cos(angle), np.sin(angle)])
    normal_field = 128 + (xx - p0[0]) * normal[0] + (yy - p0[1]) * normal[1]
    along_field = 28 + 200 / length * ((xx - p0[0]) * along[0] + (yy - p0[1]) * along[1])
    page = np.stack([normal_field, along_field, np.zeros_like(xx)], axis=2)
    return np.clip(page, 0, 255).round().astype(np.uint8)


def stroke_page(points, thickness=3):
    page = np.zeros(PAGE_SHAPE + (3,), dtype=np.uint8)
    cv2.polylines(page, [np.round(points).astype(np.int32)], False, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    return page


def stroke_deviation(crop, line_height=48):
    """Per crop column: how far (in crop rows) the stroke's centre is from the baseline row; NaN where it is missing."""
    gray = crop.astype(np.float64).mean(axis=2)
    mass = gray.sum(axis=0)
    centre = (gray * np.arange(gray.shape[0])[:, None]).sum(axis=0) / np.maximum(mass, 1e-9)
    baseline_row = (line_height - 1) * ASCENDER / (ASCENDER + DESCENDER)
    return np.where(mass > 255, np.abs(centre - baseline_row), np.nan)


def curve(kind, n=60):
    t = np.linspace(0, 1, n)
    x = 100 + 600 * t
    if kind == 'gentle arc':  # 60 px sag over 600 px
        return np.stack([x, 300 + 240 * (t - .5) ** 2], axis=1)
    if kind == 'gentle wave':
        return np.stack([x, 300 + 12 * np.sin(2 * np.pi * t)], axis=1)
    if kind == 'S-curve':
        return np.stack([x, 300 + 40 * np.sin(2 * np.pi * t)], axis=1)
    if kind == 'strong arc':  # 200 px sag over 600 px
        return np.stack([x, 200 + 800 * (t - .5) ** 2], axis=1)
    theta = {'semicircle': np.linspace(np.pi, 2 * np.pi, n), 'almost a circle': np.linspace(0, 1.8 * np.pi, 2 * n)}[kind]
    return np.stack([400 + 220 * np.cos(theta), 330 + 220 * np.sin(theta)], axis=1)


def assert_follows_the_curve(points, poly, sample_every=6):
    page = stroke_page(points)
    crop = EngineLineCropper(line_height=48, poly=poly, scale=1.0).crop(page, points[::sample_every],
                                                                       [ASCENDER, DESCENDER])
    deviation = stroke_deviation(crop)
    assert np.mean(~np.isnan(deviation)) > 0.98, 'the stroke leaves the crop'
    # in crop rows; 2.5 rows is about 1.5 px on the page for these heights
    assert np.nanpercentile(deviation, 95) <= 2.5, 'the stroke is not straightened onto the baseline row'


# --- shape and type -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('line_height', [32, 40, 48])
def test_crop_has_the_requested_height(line_height):
    page = np.zeros(PAGE_SHAPE + (3,), dtype=np.uint8)
    crop = EngineLineCropper(line_height=line_height, poly=2, scale=1.25).crop(
        page, np.array([[50.0, 135.0], [550.0, 135.0]]), [30.0, 8.0])
    assert crop.shape[0] == line_height and crop.shape[2] == 3 and crop.dtype == np.uint8


@pytest.mark.parametrize('angle', [0, 30, 90, 180])
def test_crop_width_follows_the_line_length_and_scale(angle):
    """The crop keeps the aspect ratio: width = baseline length * line_height / (heights * LINE_SCALE)."""
    baseline, _, length = straight_baseline(angle)
    crop = EngineLineCropper(**PRODUCTION).crop(np.zeros(PAGE_SHAPE + (3,), np.uint8), baseline, [ASCENDER, DESCENDER])
    expected = (length - 1) * PRODUCTION['line_height'] / ((ASCENDER + DESCENDER) * PRODUCTION['scale'])
    assert abs(crop.shape[1] - expected) <= 2


# --- geometry of straight lines -----------------------------------------------------------------------------------

@pytest.mark.parametrize('angle', [0, 5, 30, 90, 135, 180, 270, -20])
def test_straight_line_at_any_angle_is_cut_out_along_its_normal(angle):
    """Crop rows sample the page at the heights from -ascender to +descender (times LINE_SCALE) along the normal."""
    baseline, a, length = straight_baseline(angle)
    crop = EngineLineCropper(**PRODUCTION).crop(linear_fields_page(baseline[0], a, length), baseline,
                                                [ASCENDER, DESCENDER]).astype(np.float64)
    heights = np.linspace(-ASCENDER * PRODUCTION['scale'], DESCENDER * PRODUCTION['scale'], PRODUCTION['line_height'])
    np.testing.assert_allclose(crop[:, 2:-2, 0], np.broadcast_to((128 + heights)[:, None], crop[:, 2:-2, 0].shape),
                               atol=1.5)


@pytest.mark.parametrize('angle', [0, 5, 30, 90, 135, 180, 270, -20])
def test_straight_line_at_any_angle_is_read_from_its_start_to_its_end(angle):
    """Crop columns run along the baseline from its first point to its last, evenly spaced."""
    baseline, a, length = straight_baseline(angle)
    crop = EngineLineCropper(**PRODUCTION).crop(linear_fields_page(baseline[0], a, length), baseline,
                                                [ASCENDER, DESCENDER]).astype(np.float64)
    width = crop.shape[1]
    expected = 28 + 200 / length * np.arange(width) * (length - 1) / (width - 1)
    np.testing.assert_allclose(crop[:, 2:-2, 1], np.broadcast_to(expected[2:-2], crop[:, 2:-2, 1].shape), atol=2.0)


def test_baseline_row_is_set_by_the_ascender_descender_ratio():
    """The baseline lands at row (line_height - 1) * ascender / (ascender + descender)."""
    baseline = np.array([[100.0, 300.0], [700.0, 300.0]])
    for ascender, descender in [(20, 8), (10, 10), (8, 20)]:
        page = stroke_page(baseline, thickness=1)
        crop = EngineLineCropper(line_height=48, poly=2, scale=1.0).crop(page, baseline, [ascender, descender])
        row = crop[:, 10:-10].astype(np.float64).mean(axis=(1, 2)).argmax()
        assert abs(row - 47 * ascender / (ascender + descender)) <= 1


# --- curved lines -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('poly', [0, 2], ids=['INTERP 0 (cubic)', 'INTERP 2 (quadratic)'])
def test_gentle_arc_is_straightened(poly):
    assert_follows_the_curve(curve('gentle arc'), poly)


@pytest.mark.parametrize('kind', ['gentle wave', 'S-curve'])
def test_wavy_line_is_straightened_with_cubic_interpolation(kind):
    assert_follows_the_curve(curve(kind), poly=0)


@pytest.mark.xfail(reason='known limit: a quadratic fit (INTERP 2, the production setting) cannot follow a '
                          'baseline that bends both ways, even a gentle wave', strict=True)
@pytest.mark.parametrize('kind', ['gentle wave', 'S-curve'])
def test_wavy_line_is_straightened_with_the_production_setting(kind):
    assert_follows_the_curve(curve(kind), poly=2)


@pytest.mark.xfail(reason='known limit: baselines are fitted as y(x) in a frame rotated by the end-to-end angle, so a '
                          'strongly curved line is sampled along wrong normals', strict=True)
@pytest.mark.parametrize('poly', [0, 2], ids=['INTERP 0 (cubic)', 'INTERP 2 (quadratic)'])
def test_strong_arc_is_straightened(poly):
    assert_follows_the_curve(curve('strong arc'), poly)


@pytest.mark.xfail(reason='known limit: a baseline that turns back (x not monotonic in the rotated frame) cannot be '
                          'represented as y(x); with INTERP 0 the crop fails and falls back to a blank crop',
                   strict=True)
@pytest.mark.parametrize('kind', ['semicircle', 'almost a circle'])
@pytest.mark.parametrize('poly', [0, 2], ids=['INTERP 0 (cubic)', 'INTERP 2 (quadratic)'])
def test_circular_line_is_straightened(kind, poly):
    assert_follows_the_curve(curve(kind, n=80), poly, sample_every=4)


# --- page border, mappings, failures ------------------------------------------------------------------------------

def test_line_partly_outside_the_page_is_padded_with_black():
    page = np.full(PAGE_SHAPE + (3,), 200, dtype=np.uint8)
    baseline = np.array([[-100.0, 300.0], [300.0, 300.0]])
    crop = EngineLineCropper(line_height=48, poly=2, scale=1.0).crop(page, baseline, [ASCENDER, DESCENDER])
    outside = int(100 * 48 / (ASCENDER + DESCENDER))  # the first 100 px of the baseline lie left of the page
    assert crop.shape[0] == 48
    assert crop[:, :outside - 2].max() == 0
    assert crop[:, outside + 2:].min() == 200


def test_forward_mapping_gives_the_source_position_of_every_crop_pixel():
    baseline, a, length = straight_baseline(30)
    page = linear_fields_page(baseline[0], a, length)
    cropper = EngineLineCropper(**PRODUCTION)
    crop, coords = cropper.crop(page, baseline, [ASCENDER, DESCENDER], return_forward_mapping=True)
    assert coords.shape == crop.shape[:2] + (2,) and coords.dtype == np.float32
    # the first crop column is the normal through the first baseline point, from -ascender to +descender
    heights = np.linspace(-ASCENDER * PRODUCTION['scale'], DESCENDER * PRODUCTION['scale'], PRODUCTION['line_height'])
    normal = np.array([-np.sin(a), np.cos(a)])
    np.testing.assert_allclose(coords[:, 0], baseline[0] + heights[:, None] * normal, atol=1.0)
    # the cropper remaps a shifted sub-image, which rounds the sub-pixel positions slightly differently
    remapped = cv2.remap(page, coords[..., 0], coords[..., 1], cv2.INTER_LINEAR)
    assert np.abs(remapped.astype(int) - crop).max() <= 1


def test_reverse_mapping_blends_the_crop_back_into_the_page():
    rng = np.random.default_rng(0)
    page = cv2.GaussianBlur(rng.integers(0, 256, PAGE_SHAPE + (3,), dtype=np.uint8), (0, 0), 3)
    baseline, _, _ = straight_baseline(10)
    cropper = EngineLineCropper(line_height=48, poly=2, scale=1.0)
    crop, mapping, offset = cropper.crop(page, baseline, [ASCENDER, DESCENDER], return_mapping=True)
    blended = cropper.blend_in(np.zeros_like(page), crop, mapping, offset)
    inside = cropper.get_blend_mask(mapping)[..., 0] > 0.99
    region = (slice(offset[0], offset[0] + mapping.shape[0]), slice(offset[1], offset[1] + mapping.shape[1]))
    assert inside.sum() > 0.5 * (ASCENDER + DESCENDER) * 300
    difference = np.abs(blended[region].astype(int) - page[region])[inside]
    assert np.median(difference) <= 2 and np.percentile(difference, 99) <= 12


def test_degenerate_baseline_gives_a_blank_crop(capsys):
    """A one-point baseline cannot be cropped; the cropper prints an error and returns a blank 32 px wide crop."""
    page = np.full(PAGE_SHAPE + (3,), 200, dtype=np.uint8)
    crop = EngineLineCropper(line_height=48, poly=2, scale=1.0).crop(page, np.array([[300.0, 300.0]]), [20, 8])
    assert crop.shape == (48, 32, 3) and crop.max() == 0
    assert 'line crop failed' in capsys.readouterr().out
