# Dataset and text annotation format

[Back to the project README](../README.md)

## Image naming convention

The ship loader expects JPEG filenames containing an identity and camera identifier:

```text
<ship_id>_c<camera_id>_...jpg
```

Camera IDs must currently be between 1 and 8.

## Text annotation format

Each JSONL line should contain an `image_path` and an `annotation` object. The current mapping performs an exact string match between `image_path` and the image path produced by the dataset loader, so absolute/relative path conventions must be consistent.

```json
{
  "image_path": "/absolute/path/to/ShipReID2400/bounding_box_train/0000_c002_01_T_20220105_11_04_07_484171.jpg",
  "annotation": {
    "hull_color": "rusty blue",
    "fenders": {
      "density": "dense",
      "arrangement": "single row"
    },
    "cabin_details": {
      "structure_description": "a single-level blue cabin with a white railing",
      "windows_doors": "multiple rectangular windows"
    },
    "deck_info": {
      "deck_surface": "open cargo deck",
      "discriminative_objects": [
        {
          "object": "air-conditioning unit",
          "description": "a white outdoor unit mounted on the cabin side",
          "location": "cabin side"
        }
      ]
    }
  }
}
```

`utils/ship_text.py` can convert this structure into a global caption, visual-cue description and attribute tags. In the current snapshot, `datasets/bases.py` does not yet call this serializer and instead supplies a fixed caption. This must be corrected before reproducing the generated-caption experiment.
