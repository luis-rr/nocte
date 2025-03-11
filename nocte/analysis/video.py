"""
Process video from the animals to extract when lights went on/off
"""

import colorsys
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nocte import stacks, timeslice
from nocte import traces as tr
from nocte.io import neuralynx

MICROS_TO_MS = .001


class VideoWriter:
    """
    A context manager for writing video frames to a file using opencv2

    Use like:
            with vid.VideoWriter(
                output_video_path,
                frame_width,
                frame_height,
                fps) as video_writer:

                video_writer.write(img)

    """

    def __init__(self, output_video_path, frame_width, frame_height, fps):
        self.output_video_path = output_video_path
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.out = None

    def __enter__(self):
        # noinspection PyUnresolvedReferences
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.frame_width, self.frame_height))
        return self.out

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.out is not None:
            self.out.release()


def seek_fast(cap, start_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

def seek_safe(cap, start_idx):
    for _ in tqdm(range(start_idx), desc='Seeking frames'):
        ret, _ = cap.read()
        if not ret:
            break

class VideoReader:
    """
    A context manager for reading video frames with step control.

    Usage:
        with VideoReader(...) as video_reader:
            for idx, frame in video_reader:
                pass
    """

    def __init__(self, input_video_path, start, stop, step=1, safe_seek=False):
        input_video_path = str(input_video_path)

        self.cap = cv2.VideoCapture(input_video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file {input_video_path}")

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start < 0 or stop >= total_frames or start > stop:
            raise ValueError(f"Invalid frame indices: {start}, {stop}")

        if step <= 0:
            raise ValueError("Step must be a positive integer")

        self.start = start
        self.stop = stop
        self.step = step
        self.safe_seek = safe_seek

    def __enter__(self):
        if not self.safe_seek:
            seek_fast(self.cap, self.start)
        else:
            seek_safe(self.cap, self.start)
        return self

    def __iter__(self):
        for idx in tqdm(range(self.start, self.stop, self.step), desc='Processing frames'):
            ret, frame = self.cap.read()
            if not ret:
                break
            yield idx, frame

            # Skip extra frames if step > 1
            for _ in range(self.step - 1):
                ret, _ = self.cap.read()
                if not ret:
                    break

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()  # Ensures proper cleanup exactly once


def load_movie(avi_path, start_ms=None, stop_ms=None, time_coord=True, step=1) -> stacks.Stack:
    cap = cv2.VideoCapture(str(avi_path))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # skip the last. dummy, frame
    frame_count = frame_count - 1

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_idx, stop_idx = 0, frame_count

    if start_ms is not None:
        start_idx = int(np.round(start_ms * timeslice.MS_TO_S * fps))
        assert 0 <= start_idx < frame_count, \
            f'Start time must fall within {timeslice.Win(0, frame_count / fps * timeslice.S_TO_MS)}'

    if stop_ms is not None:
        stop_idx = int(np.round(stop_ms * timeslice.MS_TO_S * fps))
        assert 0 <= stop_idx <= frame_count, \
            f'Stop time must fall within {timeslice.Win(0, frame_count / fps * timeslice.S_TO_MS)}'
        stop_idx = int(np.round(stop_ms * timeslice.MS_TO_S * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    idcs = np.arange(start_idx, stop_idx, step)

    data = np.empty((len(idcs), frame_height, frame_width, 3), np.dtype('uint8'))

    iteration = idcs
    if len(iteration) > 100:
        iteration = tqdm(iteration, 'frames')

    for i, _ in enumerate(iteration):

        ret, frame_data = cap.read()

        if not ret:
            break

        data[i] = frame_data

    coords = {}

    if time_coord:
        coords['time'] = idcs / fps * timeslice.S_TO_MS

    else:
        coords['frame'] = idcs

    coords['height'] = pd.RangeIndex(stop=frame_height)
    coords['width'] = pd.RangeIndex(stop=frame_width)
    coords['rgb'] = [0, 1, 2]

    dat = stacks.Stack.from_array(data, coords)

    return dat


def get_movie_frame_count(avi_path) -> int:

    cap = cv2.VideoCapture(str(avi_path))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return frame_count


def get_frame_idx_by_fps(avi_path, time_ms) -> int:
    """
    Get a frame time by naively assuming fixed fps.
    Note that this may need adjustment if we want the time relative
    to the ephys recording. For that, take a look at load_movie_frames_ms
    """
    cap = cv2.VideoCapture(str(avi_path))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # skip the last. dummy, frame
    frame_count = frame_count - 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_idx, stop_idx = 0, frame_count

    time_idx = int(np.round(time_ms * timeslice.MS_TO_S * fps))
    assert 0 <= start_idx < frame_count, \
        f'Time must fall within {timeslice.Win(0, frame_count / fps * timeslice.S_TO_MS)}'

    return time_idx


def load_movie_frame_idx(avi_path, idx) -> stacks.Stack:

    cap = cv2.VideoCapture(str(avi_path))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    ret, frame_data = cap.read()

    coords = {
        'height': pd.RangeIndex(stop=frame_height),
        'width': pd.RangeIndex(stop=frame_width),
        'rgb': [0, 1, 2],
    }

    dat = stacks.Stack.from_array(frame_data, coords)

    return dat


def load_movie_frames_ms(exp_info, show_times):
    video_path = exp_info.get_path_video()

    frame_times = get_cam_frame_timestamps_rec(exp_info)

    frames = {}

    for i, t in enumerate(tqdm(show_times, desc='load frames')):
        frames[t] = load_movie_frame_idx(
            video_path,
            np.searchsorted(frame_times, t)
        )

    return frames


def get_video_length_ms(avi_path):
    cap = cv2.VideoCapture(str(avi_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # skip the last. dummy, frame
    frame_count = frame_count - 1
    fps = cap.get(cv2.CAP_PROP_FPS)

    return (frame_count / fps) * 1000


def get_fps(avi_path):
    cap = cv2.VideoCapture(str(avi_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps


def extract_lights_on(frames: stacks.Stack) -> pd.Series:
    """
    Estimate when lights where on/off on a movie
    """
    one_pix = frames.mean(['height', 'width'])

    # Technically not the luminance, but a combination of hue and saturation
    # since the camera switches to black and white when it's infra-red (lights off)
    # and uses color (mainly green) otherwise.
    # hsv = np.array([colorsys.rgb_to_hsv(*frame) for frame in pbar(one_pix.values)])
    # signal = np.mean(hsv[:, [0, 1]], axis=1)

    hls = np.array([colorsys.rgb_to_hls(*frame) for frame in tqdm(one_pix.values)])
    signal = hls[:, 1]

    return pd.Series(signal, index=frames.coords['time'])


def series_exists(res_path, res_key):
    res_path = Path(res_path)

    if not res_path.exists():
        return False

    with h5py.File(res_path, mode='r') as f:
        return res_key in f


def extract_exp_luminance(vid_path, res_path, key):
    vid_path = Path(vid_path)
    res_path = Path(res_path)

    assert vid_path.exists()

    frames = load_movie(vid_path)
    if 0 in frames.shape:
        logging.error(f'No data in video. File under processing? {vid_path}')
        return

    lum = extract_lights_on(frames)

    res_path.parent.mkdir(parents=True, exist_ok=True)
    lum.to_hdf(str(res_path), key=key)


def fix_exp_luminance(exp_info, lum_raw, smooth_ms=None):
    lum = lum_raw.copy()

    if smooth_ms is not None:
        sampling_period = np.median(np.diff(lum_raw.index))
        lum = lum_raw.rolling(int(smooth_ms / sampling_period), center=True, min_periods=1).median()

    if exp_info['probe'] == 'neuronexus':
        lum = adjust_frame_times_multiple_tries(exp_info, lum)

    lum = (lum - lum.min()) / (lum.max() - lum.min())

    return lum


def extract_light_wins(lum: pd.Series, high_q=.99, low_q=.1, merge_length=1_000):
    th = .5 * (lum.quantile(high_q) + lum.quantile(low_q))

    light_wins = timeslice.Windows.build_from_contiguous_values(lum > th, include_right=False)
    light_wins['cat'] = light_wins['cat'].map({True: 'on', False: 'off'})

    light_wins = light_wins.merge_sandwiched(max_length=merge_length)

    # Make sure the last and first windows are "on"
    short = light_wins.lengths() < merge_length

    first = light_wins.index[0]
    if short[first]:
        light_wins.wins.loc[first, 'cat'] = 'on'

    last = light_wins.index[-1]
    if short[last]:
        light_wins.wins.loc[last, 'cat'] = 'on'

    light_wins = light_wins.merge_tight(same_cat=True)

    return light_wins


def get_first_timestamp(loader) -> float:
    timestamps = []

    for idx, lo in loader.loaders.items():
        with open(lo.header['full_path'], 'rb') as fid:
            recs = neuralynx.NeuralynxBaseLoader.read_records(fid, neuralynx.NCSLoader.RECORD, 0, 1)

            timestamps.append(recs['TimeStamp'][0])

    timestamps = np.asarray(timestamps)

    assert np.all(timestamps[0] == timestamps), f'Different starting timestamps for different sub-loaders'

    return np.min(timestamps) * MICROS_TO_MS


def get_cam_frame_timestamps(events_path, cam=b'cam0') -> np.ndarray:
    nev = neuralynx.NEVLoader.load_nev(events_path)

    timestamps = nev.records['TimeStamp']
    mask = nev.records['EventString'] == cam

    cam_frames = timestamps[mask]
    assert np.all(cam_frames[:-1] < cam_frames[1:]), 'Frames timestamps should be sorted'

    cam_frames = cam_frames * MICROS_TO_MS

    return cam_frames


def adjust_frame_times(exp_info, lum: pd.Series, cam=b'cam0'):
    new_time = get_cam_frame_timestamps_rec(exp_info, cam)

    if len(new_time) == len(lum) - 1:
        # in some versions of the data, we stored info about a dummy last frame that doesn't really exist
        logging.warning(f'Skipping last entry (missing 1 timestamp)')
        lum = lum.iloc[:-1]

    assert len(new_time) == len(lum), \
        f'Unable to adjust {exp_info.name}. ' \
        f'Got {len(lum):,d} frames but {len(new_time):,d} timestamps' \
        f' (diff: {len(lum) - len(new_time)})'

    offsets = new_time - lum.index
    logging.info(f'Adjusting {exp_info.name} for offset: {np.min(offsets)}- {np.max(offsets)}')

    return pd.Series(lum.values, index=new_time)


def get_cam_frame_timestamps_rec(exp_info, cam=b'cam0'):
    """
    Get frame timestamps in recording time
    """
    loader = exp_info.get_loader(accept_non_interp=True)

    start = get_first_timestamp(loader)

    cam_timestamps_path = exp_info.get_path() / 'Events.nev'

    cam_timestamps = get_cam_frame_timestamps(cam_timestamps_path, cam=cam)

    new_time = cam_timestamps - start

    return new_time


def adjust_frame_times_multiple_tries(exp_info, lum: pd.Series):
    to_try = [b'cam0', b'cam4', b'angle_basler']

    for i, cam in enumerate(to_try):
        try:
            return adjust_frame_times(exp_info, lum, cam=cam)

        except AssertionError:
            # print(cam, 'failed', flush=True)
            if i == len(to_try) - 1:
                raise
            else:
                logging.info(f'Try different camera label')


########################################################################################################################
# deep lab cut eye-tracking


def load_deeplabcut(path, sampling_rate=50.):
    df = pd.read_csv(
        str(path),
        index_col=0,
        header=[0, 1, 2],
    )

    assert df.columns.get_level_values('scorer').nunique() == 1

    df = df.droplevel('scorer', axis=1)

    right = df.loc[:, df.columns.get_level_values('bodyparts').str.startswith('R')]
    right = right.rename(
        columns=dict(zip(right.columns.get_level_values('bodyparts'),
                         right.columns.get_level_values('bodyparts').str.slice(1, None)))
    )

    left = df.loc[:, df.columns.get_level_values('bodyparts').str.startswith('L')]
    left = left.rename(
        columns=dict(zip(left.columns.get_level_values('bodyparts'),
                         left.columns.get_level_values('bodyparts').str.slice(1, None)))
    )

    df = pd.concat({'left': left, 'right': right}, axis=1, names=['side']).sort_index(axis=1)

    sampling_period_ms = 1000. / sampling_rate
    df.index = df.index * sampling_period_ms

    return df


def load_deeplabcut_multi(exp_paths):
    return {
        exp_name: load_deeplabcut(tracking_path)
        for exp_name, tracking_path in tqdm(exp_paths.items(), desc='load', total=len(exp_paths))
    }


def adjust_deeplabcut_time_multi(reg, exp_tracking):
    result = {}

    for exp_name, tracking in tqdm(exp_tracking.items(), desc='adjust'):

        if exp_name not in reg.experiment_names:
            logging.warning(f'{exp_name} missing from reg. Skipping adjustment.')

        else:
            new_times = get_cam_frame_timestamps_rec(reg.get_entry(exp_name))

            if len(new_times) == len(tracking):

                new_tracking = tracking.copy()

                new_tracking.index = new_times

                result[exp_name] = new_tracking

                print(f'{exp_name} time adjusted by {np.max(new_times - tracking.index):.2f} ms')

            else:
                logging.error(
                    f'Unable to adjust {exp_name}. '
                    f'Got {len(tracking):,d} frames but {len(new_times):,d} timestamps'
                    f' (diff: {len(tracking) - len(new_times)})'
                )

    return result


def extract_eye_open_multi(exp_dlc, max_pix=100):
    def euclidean_distance(xy0, xy1):
        diff = xy0 - xy1
        return np.sqrt(np.square(diff).sum(axis=1))

    exp_eyes = {
        exp_name: pd.DataFrame({
            'left': euclidean_distance(
                dlc['left', 'TopEL'][['x', 'y']],
                dlc['left', 'BottomEL'][['x', 'y']],
            ),

            'right': euclidean_distance(
                dlc['right', 'TopEL'][['x', 'y']],
                dlc['right', 'BottomEL'][['x', 'y']],
            ),
        })

        for exp_name, dlc in exp_dlc.items()
    }

    exp_eyes = pd.concat(exp_eyes, axis=1)

    exp_eyes.sort_index(inplace=True)

    exp_eyes = tr.Traces.from_multiindex_df(exp_eyes.rename_axis(columns=['exp_name', 'side']))

    # downsample to match upsampled beta
    exp_eyes = exp_eyes.resample(100, start=0)

    # remove bad tracking artifacts
    # Note we want to do this AFTER resampling, since that involves an interpolation
    # but we actually want the nans to indicate failure to track
    exp_eyes.traces[exp_eyes.traces > max_pix] = np.nan

    exp_eyes = exp_eyes.normalize_by_quantiles()

    return exp_eyes
