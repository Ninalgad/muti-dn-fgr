"""
The following is a the inference code for running the baseline algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-preliminary-development-phase | gzip -c > example-algorithm-preliminary-development-phase.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK

from model import FetalAbdomenSegmentation

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

DEBUG = False


def run():
    # Read the input
    stacked_fetal_ultrasound_path = get_image_file_path(
        location=INPUT_PATH / "images/stacked-fetal-ultrasound")

    # Process the inputs: any way you'd like
    _show_torch_cuda_info()

    # Instantiate the algorithm
    algorithm = FetalAbdomenSegmentation()

    # Forward pass
    fetal_abdomen_probability_map, relative_frame_distances = algorithm.predict(
        stacked_fetal_ultrasound_path, debug=DEBUG)  # (372, 281, num_frames), (num_frames,)

    # Postprocess the output
    fetal_abdomen_postprocessed = algorithm.postprocess(
        fetal_abdomen_probability_map)  # (num_frames, 562, 744)

    # Select the fetal abdomen mask and the corresponding frame number
    fetal_abdomen_frame_number = get_frame_number(relative_frame_distances)
    fetal_abdomen_segmentation = fetal_abdomen_postprocessed[fetal_abdomen_frame_number]

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/fetal-abdomen-segmentation",
        array=fetal_abdomen_segmentation,
        frame_number=fetal_abdomen_frame_number,
    )
    write_json_file(
        location=OUTPUT_PATH / "fetal-abdomen-frame-number.json",
        content=fetal_abdomen_frame_number
    )

    # Print the output
    print("output folder contents:")
    print_directory_contents(OUTPUT_PATH)

    # Print shape and type of the output
    print("\nprinting output shape and type:")
    print(f"shape: {fetal_abdomen_segmentation.shape}")
    print(f"type: {type(fetal_abdomen_segmentation)}")
    print(f"dtype: {fetal_abdomen_segmentation.dtype}")
    print(f"unique values: {np.unique(fetal_abdomen_segmentation)}")
    print(f"frame number: {fetal_abdomen_frame_number}")
    print(type(fetal_abdomen_frame_number))

    return 0


def get_frame_number(rel_dist):
    n = len(rel_dist)

    # aggregate predictions from each frame
    rfn = np.argmax(compute_distance_distribution(n*rel_dist), n=n)

    # overall minimum prediction
    mfn = np.argmin(np.abs(rel_dist))

    # test show averaging yields best results
    return int(np.rint((rfn + mfn) / 2))


def compute_distance_distribution(relative_distances, n=840):
    dist = np.zeros(n)
    for i, x in enumerate(relative_distances):
        if x != -1:
            j = int(i + x)
            j = np.clip(j, 0, n - 1)
            dist[j] = 1 + dist[j]

    return dist


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + \
                  glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)


# Get image file path from input folder


def get_image_file_path(*, location):
    input_files = glob(str(location / "*.tiff")) + \
                  glob(str(location / "*.mha"))
    return input_files[0]


def write_array_as_image_file(*, location, array, frame_number=None):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"
    # Assert that the array is 2D
    assert array.ndim == 2, f"Expected a 2D array, got {array.ndim}D."

    # Convert the 2D mask to a 3D mask (this is solely for visualization purposes)
    array = convert_2d_mask_to_3d(
        mask_2d=array,
        frame_number=frame_number,
        number_of_frames=840,
    )

    image = SimpleITK.GetImageFromArray(array)
    # Set the spacing to 0.28mm in all directions
    image.SetSpacing([0.28, 0.28, 0.28])
    SimpleITK.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def convert_2d_mask_to_3d(*, mask_2d, frame_number, number_of_frames):
    # Convert a 2D mask to a 3D mask
    mask_3d = np.zeros((number_of_frames, *mask_2d.shape), dtype=np.uint8)
    # If frame_number == -1, return a 3D mask with all zeros
    if frame_number == -1:
        return mask_3d
    # If frame_number is within the valid range, set the corresponding frame to the 2D mask
    if frame_number is not None and 0 <= frame_number < number_of_frames:
        mask_3d[frame_number, :, :] = mask_2d
        return mask_3d
    # If frame_number is None or out of bounds, raise a ValueError
    else:
        raise ValueError(
            f"frame_number must be between -1 and {number_of_frames - 1}, got {frame_number}."
        )


def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print_directory_contents(child_path)
        else:
            print(child_path)


def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(
        f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(
            f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(
            f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
