"""Ma et al. (2003)-inspired CASIA-V1 iris recognition with rotation compensation."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import ImageEnhancement
from FeatureExtraction import FeatureExtraction
from IrisMatching import IrisMatching
from PerformanceEvaluation import PerformanceEvaluation


# NEW: Seven rotation angles
ANGLES = [-9, -6, -3, 0, 3, 6, 9]


# NEW: Generate features from rotated training images
def generate_rotated_features(paths, frequency):

    features = []
    labels = []
    angles = []

    # Only use images from session 1
    train_paths = [
        path for path in paths
        if int(path.stem.split('_')[1]) == 1
    ]

    for i, path in enumerate(train_paths):

        eye, session, number = map(int, path.stem.split('_'))

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Cannot read image: {path}")

        # Localization only needs to be done once per image
        pupil, iris = IrisLocalization(img)

        # Generate seven rotated normalized images
        for angle in ANGLES:

            norm = IrisNormalization(
                img,
                pupil,
                iris,
                angle=angle
            )

            enhanced = ImageEnhancement(norm)

            v = FeatureExtraction(enhanced, frequency)

            features.append(v)
            labels.append(eye)
            angles.append(angle)

        if (i + 1) % 50 == 0:
            print(
                'Rotated preprocessing',
                i + 1,
                '/',
                len(train_paths),
                flush=True
            )

    return (
        np.stack(features),
        np.array(labels),
        np.array(angles)
    )


# NEW: Adapter that enables rotation matching during evaluation
class RotationMatcher:

    def __init__(self, model):
        self.model = model
        self.labels = model.labels

    def distances(self, X, metric='cosine'):
        return self.model.distances(
            X,
            metric=metric,
            rotation=True
        )

    def predict(self, X, metric='cosine'):
        return self.model.predict(
            X,
            metric=metric,
            rotation=True
        )

    def __getattr__(self, name):
        return getattr(self.model, name)


def main():

    p = argparse.ArgumentParser()

    p.add_argument('--data', default='data')
    p.add_argument('--out', default='results')
    p.add_argument('--frequency', type=float, default=0.15)
    p.add_argument('--recompute', action='store_true')

    # NEW: Option to disable rotation compensation
    p.add_argument('--no-rotation', action='store_true')

    args = p.parse_args()

    paths = sorted(Path(args.data).rglob('*.bmp'))

    if len(paths) != 756:
        raise ValueError(
            f'Expected 756 BMPs, got {len(paths)}'
        )

    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)

    # Existing feature cache
    cache = out / 'features.npz'

    if cache.exists() and not args.recompute:

        d = np.load(cache)

        X = d['X']
        y = d['y']
        sessions = d['sessions']

    else:

        features = []
        labels = []
        sessions = []
        errors = []

        for i, path in enumerate(paths):

            eye, session, number = map(
                int,
                path.stem.split('_')
            )

            img = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE
            )

            try:

                pupil, iris = IrisLocalization(img)

                norm = IrisNormalization(
                    img,
                    pupil,
                    iris
                )

                enhanced = ImageEnhancement(norm)

                v = FeatureExtraction(
                    enhanced,
                    args.frequency
                )

            except Exception as e:

                errors.append((path.name, str(e)))

                raise RuntimeError(
                    f'Preprocessing failed for {path}: {e}'
                ) from e

            features.append(v)
            labels.append(eye)
            sessions.append(session)

            if (i + 1) % 100 == 0:
                print(
                    'Preprocessed',
                    i + 1,
                    '/',
                    len(paths),
                    flush=True
                )

        X = np.stack(features)
        y = np.array(labels)
        sessions = np.array(sessions)

        np.savez_compressed(
            cache,
            X=X,
            y=y,
            sessions=sessions
        )

    # Original training/testing split
    train = sessions == 1
    test = sessions == 2

    print(
        f'Train: {sum(train)} images / '
        f'{len(np.unique(y[train]))} classes; '
        f'Test: {sum(test)} images'
    )

    # Original PCA + Fisher LDA training
    model = IrisMatching(
        n_components=100,
        shrinkage=0.1
    ).fit(X[train], y[train])

    # NEW: Rotation compensation
    if not args.no_rotation:

        print('\nGenerating rotated training features...')

        rotation_cache = out / (
            f'rotated_features_f{args.frequency:.4f}.npz'
        )

        if rotation_cache.exists() and not args.recompute:

            d = np.load(rotation_cache)

            X_rotated = d['X_rotated']
            y_rotated = d['y_rotated']
            rotation_angles = d['rotation_angles']

        else:

            X_rotated, y_rotated, rotation_angles = (
                generate_rotated_features(
                    paths,
                    args.frequency
                )
            )

            np.savez_compressed(
                rotation_cache,
                X_rotated=X_rotated,
                y_rotated=y_rotated,
                rotation_angles=rotation_angles
            )

        print(
            'Rotated training samples:',
            len(X_rotated)
        )

        # Calculate seven class centers per identity
        model.fit_rotation(
            X_rotated,
            y_rotated,
            rotation_angles
        )

        # Enable rotation-compensated matching
        evaluation_model = RotationMatcher(model)

    else:

        print('\nRotation compensation disabled.')

        evaluation_model = model

    # Original performance evaluation
    table3, table4 = PerformanceEvaluation(
        evaluation_model,
        X[test],
        y[test],
        out
    )

    print(
        '\nTable 3 / Identification CRR\n',
        table3.to_string(index=False)
    )

    print(
        '\nTable 4 / Verification\n',
        table4.to_string(index=False)
    )


if __name__ == '__main__':
    main()