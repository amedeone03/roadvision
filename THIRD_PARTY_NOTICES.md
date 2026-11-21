# Third-party notices

RoadVision calls the components below. RoadVision does not own them and does not grant a license to them.

## Ultralytics and YOLO11n

Detection uses the pretrained YOLO11n model through the Ultralytics Python package. Ultralytics licenses that package, and states that its YOLO11 models are licensed, under the GNU Affero General Public License, version 3.

- Ultralytics software: https://github.com/ultralytics/ultralytics
- Ultralytics license: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- Ultralytics licensing guidance: https://www.ultralytics.com/license
- YOLO11 documentation: https://docs.ultralytics.com/models/yolo11/

## ByteTrack

The tracker invoked by RoadVision is the ByteTrack implementation inside Ultralytics, `ultralytics/trackers/byte_tracker.py`. That file is part of the Ultralytics package and is covered by Ultralytics' AGPL-3.0 license. RoadVision supplies a tracker configuration. It does not include the tracker source.

The original ByteTrack research repository is a separate project. It is licensed under the MIT License, copyright Yifu Zhang, 2021. That MIT license applies to the original repository. It does not replace the AGPL-3.0 license on the Ultralytics implementation used here.

- Ultralytics tracker: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/byte_tracker.py
- Original ByteTrack repository: https://github.com/FoundationVision/ByteTrack
- Original ByteTrack license: https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE
- ByteTrack paper: Yifu Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022. https://arxiv.org/abs/2110.06864

## Model weights

`yolo11n.pt` is not stored in this repository. The setup command downloads it from Ultralytics into the working directory. Those weights stay subject to Ultralytics' upstream terms. Downloading the file does not transfer ownership to RoadVision and does not change the weights' license.
