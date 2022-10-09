original README see [README](https://github.com/NVlabs/planercnn/blob/master/README.md)

several modifications have been done for pytorch1.10.1 with cuda11.3

### Dependencies
In order to get rid of pre-compilation for `roialign` and `nms`, and to make it suitable for higher pytorch version, I did several changes as below:
#### RoIAlign
install RoIAlign for pytorch1.0 from [here](https://github.com/longcw/RoIAlign.pytorch)

in file `models/model.py`, change
```py
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
```
to
```py
from roi_align import CropAndResize
```
and replace every `CropAndResizeFunction` with `CropAndResize` without changing its usage
#### nms
in file `models/model.py`, change
```py
from nms.nms_wrapper import nms
```
to
```py
from torchvision.ops import nms
```
Notice that there are several differences between the usage of these two `nms` functions which I have changed in `model.py`

### Requirements
add h5py to `requirements.txt`
update requirements with current version I'm using

### BUGS
fix some small bugs in `data_prep/parse.py` and `evaluate.py`

### P.S.
it works for me in evaluation