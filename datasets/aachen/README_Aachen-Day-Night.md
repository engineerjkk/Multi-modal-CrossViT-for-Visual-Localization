# The Aachen Day-Night Dataset
This is the public release of the Aachen Day-Night dataset that is used in this paper to 
benchmark visual localization and place recognition algorithms under changing conditions:
```
T. Sattler, W. Maddern, C. Toft, A. Torii, L. Hammarstrand, E. Stenborg, D. Safari, M. Okutomi, M. Pollefeys, J. Sivic, F. Kahl, T. Pajdla. 
Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions. 
Conference on Computer Vision and Pattern Recognition (CVPR) 2018 
```
The dataset is based on the Aachen dataset described in this paper:
```
T. Sattler, T. Weyand, B. Leibe, L. Kobbelt.
Image Retrieval for Image-Based Localization Revisited. 
British Machine Vision Conference (BMCV) 2012
```
The Aachen Day-Night dataset uses the database images provided by the original Aachen 
dataset, together with additionally recorded sequences, to build a reference 3D model of the 
old inner city of Aachen, Germany. The reference 3D model was reconstructed using 
COLMAP [1,2]. After reconstruction, the additional images were removed  from the 3D model. 
The resulting 3D model consequently defines a set of 6DOF reference poses for the database 
images.
Both the database images and additional sequences were taken under daytime conditions.
The dataset provides 824 query images taken during day and 98 query images taken during 
night. All query images were taken using mobile phones. For the nighttime queries, software 
HDR was used to record high-quality, well-illuminated images. 
The  reference 6DOF poses for the query images will not be released. Rather, we provide an 
[evaluation service](http://visuallocalization.net/) (see below).


## License
The images of the Aachen Day-Night dataset are licensed under a 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) and are intended for non-commercial academic use only. 
All other data also provided as part of the Aachen Day-Night dataset, including the 3D model 
and the camera calibrations, is derived from these images. Consequently, all other data is 
also licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) and intended for non-commercial academic use only.

We are not planning to make the dataset available for commercial use.

### Using the Aachen Day-Night Dataset
By using the Aachen Day-Night dataset, you agree to the license terms set out above.
If you are using the Aachen Day-Night dataset, please cite **both** of the following two papers:
```
@inproceedings{Sattler2018CVPR,
  author={Sattler, Torsten and Maddern, Will and Toft, Carl and Torii, Akihiko and Hammarstrand, Lars and Stenborg, Erik and Safari, Daniel and Okutomi, Masatoshi and Pollefeys, Marc and Sivic, Josef and Kahl, Fredrik and Pajdla, Tomas},
  title={{Benchmarking 6DOF Outdoor Visual Localization in Changing Conditions}},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018},
}

@inproceedings{Sattler2012BMVC,
  author={Sattler, Torsten and Weyand, Tobias and Leibe, Bastian and Kobbelt, Leif},
  title={{Image Retrieval for Image-Based Localization Revisited}},
  booktitle = {British Machine Vision Conference (BMCV)},
  year = {2012},
}
```

### Privacy
We take privacy seriously. For this reason, we used software to automatically blur faces and 
license plates in the images before the release. We verified the results manually and added 
black bars for missed detections. If you have any concerns regarding the images and other 
data provided with this dataset, or find faces or license plates we have missed, please 
[contact us](mailto:torsat@chalmers.se).


## Provided Files
The following files are provides with this release of the Aachen Day-Night dataset in various 
directories:
* `3D-models/`: Contains the 3D model created from the reference images in various formats.
* `images/`: Contains the images of the Aachen Day-Night dataset.
* `query/`: Contains the intrinsic calibrations of the query images used in the dataset.
 
In the following, we will describe the different files provided with the dataset in more detail.

### 3D Model
This directory contains the reference 3D model build for the dataset. 
We provide the 3D reconstruction in various formats:
* `aachen_cvpr2018_db.nvm`: 3D model in the [NVM file format](http://ccwu.me/vsfm/doc.html#nvm) 
used by VisualSfM [5,6]. 
* `aachen_cvpr2018_db.out` and `aachen_cvpr2018_db.list.txt`: 3D model in the 
[file format](http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6) 
used by Bundler [7]. The `.list.txt` file specifies the images in the reconstruction. The order 
is the same as in the reconstruction.
* Binary `.info` file: These are binary file contains the poses of all database images, all 
3D points in the reference 3D model, and their corresponding descriptors. A C++ code 
snippet for loading the data is available [here](https://s3.amazonaws.com/LocationRecognition/Datasets/load_info_file_snippet.cc).

Please familiarize yourself with the different file formats. Please also pay special attention to 
the different camera coordinate systems and conventions of the different formats detailed below. 

#### Coordinate Systems and Conventions (Important!)
We provide the 3D models in different file formats for convenience. However, providing 
different formats also comes with a catch: *Different formats use different coordinate systems 
and conventions*.

##### Camera Coordinate Systems
The NVM model uses the camera coordinate system typically used in *computer vision*. In this 
camera coordinate system, the camera is looking down the `z`-axis, with the `x`-axis pointing 
to the right and the `y`-axis pointing downwards. The coordinate `(0, 0)` corresponds to the 
top-left corner of an image. 

The Bundler model and the `.info` file (which are based on the Bundler models) both use the 
same camera coordinate system, but one that differs from the *computer vision camera 
coordinate system*. More specifically, they use the camera coordinate system typically used 
in *computer graphics*. In this camera coordinate system, the camera is looking down the 
`-z`-axis, with the `x`-axis pointing to the right and the `y`-axis pointing upwards. The 
coordinate `(0, 0)` corresponds to the lower-left corner of an image. 
Camera poses in the Bundler and `.info` formats can be converted to poses in the NVM 
format via the following pseudo-code
```
// Here, R_* is the rotation from the world into the camera coordinate system
// and c_* is the position of the camera in the world-coordinate system.
// We denote the two coordinate systems as "nvm" (NVM and COLMAP) and 
// "out" (Bundler and .info files).
c_nvm = c_out;
// Mirrors the y- and z-coordinates
c_nvm[1] *= -1.0;
c_nvm[2] *= -1.0;
R_nvm = R_out;
// Changes the sign of some entries.
R_nvm(0, 1) *= -1.0;
R_nvm(0, 2) *= -1.0;
R_nvm(1, 0) *= -1.0;
R_nvm(2, 0) *= -1.0;
```
For the **evaluation** of poses estimated on the Aachen Day-Night dataset, you will need to 
provide **pose estimates in the coordinate system used by NVM**.

##### Conventions
The different types of models store poses in different formats.
* The NVM model stores the rotation (as a quaternion) from the world coordinate system 
to the camera coordinate system as well as the camera position in world coordinates. Thus, 
NVM stores a pose as `R, c` and the camera translation can be computed as 
`t = -(R * c)`.
* The Bundler model and `.info` file store the rotation (as a 3x3 matrix) from the world 
coordinate system to the camera coordinate system as well as the camera translation. Thus, 
they store a pose as `[R|t]`.

We strongly recommend that you familiarize yourself with the file format of the models that 
you plan to use.

##### Intrinsic Camera Parameters
The text file `database_intrinsics.txt` contains the intrinsic camera parameters of the 
reference images. Each line has the format
`image_name SIMPLE_RADIAL w h f cx cy r`
and corresponds to a reference image.
Here, `image_name` is the name of the image. The name includes the subfolder name. The 
path of the image is relative to the `images_upright/` directory. The remainder of the line 
describes the intrinsic calibration of the camera. We use the `SIMPLE_RADIAL` model used 
in COLMAP (see [here](https://colmap.github.io/cameras.html) for details). 
The intrinsic calibration is defined by the width `w` and height `h` of the image, its focal length 
`f`, the position of the principal point (`cx` and `cy`), and a radial distortion parameter `r`. 

Please note that the intrinsic parameters stored in the `.nvm`, `.out`, and `.info`  files might 
differ. You should ignore the intrinsics in these files and only use the ones provided in 
`database_intrinsics.txt`.

#### COLMAP Database
In addition to the 3D model, we provide a [COLMAP](https://colmap.github.io/) database, 
`aachen.db`, storing the upright RootSIFT [3,4] features used in the CVPR 2018 paper for the 
reference and query images. COLMAP provides functionality to export the features from the 
databases. However, COLMAP's database format has changed since `aachen.db` was created 
(now representing feature geometry by 6 instead of 4 values). We thus provide a Python 
script that can be used to export the features in the binary format used by VisualSfM. 

Notice that the database stores features for more images than are provided at the moment. 
These images correspond to the additional daytime images used to build the reference 3D 
model, which were later removed from the model to create the model released in this dataset. 
We will release these additional images later.

### Images
This directory contains a zip file with the images. The zip file extracts into a subdirectory 
`images_upright/`, which should be placed in the same folder as the `aachen.db` file. 
All images in the subfolders of `images_upright/query/` consistute the query images of the 
Aachen Day-Night dataset.

### Queries
We provide two text files with information about the intrinsics of the query images, one for the 
daytime queries and one for the nighttime queries. Both text files have the following file format: 
`image_name SIMPLE_RADIAL w h f cx cy r`
Here, `image_name` is the name of the query image. The name includes the subfolder names. 
The path of the image is relative to the `images_upright/` directory. The remainder of the 
line describes the intrinsic calibration of the camera. We use the `SIMPLE_RADIAL` model used 
in COLMAP (see [here](https://colmap.github.io/cameras.html) for details). 
The intrinsic calibration is defined by the width `w` and height `h` of the image, its focal length 
`f`, the position of the principal point (`cx` and `cy`), and a radial distortion parameter `r`. 


## Evaluating Results on the Aachen Day-Night Dataset
We have set up an [evaluation service](http://visuallocalization.net/) for the benchmarks proposed in the CVPR 
2018 paper. You are able to upload the poses estimated by your method to this service, 
which in turn will provide results to you. 

Please submit your results as a text file using the following file format. For each query 
image for which your method has estimated a pose, use a single line. This line should store the
result as `name.jpg qw qx qy qz tx ty tz`.  
Here,  `name` corresponds to the filename of the image, without any directory names. 
`qw qx qy qz` represents the **rotation** from world to camera coordinates as a 
**unit quaternion**. `tx ty tz` is the camera **translation** (**not the camera position**). 
An example, obtained using the *DenseVLAD* baseline from the CVPR 2018 paper, for such a 
line is 
```
IMG_20140520_182846.jpg 0.004300400000000 -0.009711270000000 0.087104600000000 0.996143000000000 724.380999999999972 23.511800000000008 180.934000000000026
```
Note that the type of condition (day or night) **is not specified** as there is a unique mapping 
from image names to conditions.

Please adhere to the following naming convention for the files that you submit:
```
Aachen_eval_[yourmethodname].txt
```
Here, `yourmethodname` is some name or identifier chosen by yourself. This name or identifier 
should be as unique as possible to avoid confusion with other methods. Once the evaluation 
service is ready, it will be used to display the results of your method.

**IMPORTANT:** Our evaluation tools expect that the coordinate system in which the camera 
pose is expressed is the **NVM coordinate system**. If you are using the Bundler or `.info` 
coordinate system to estimate poses, you will need to **convert poses to the NVM 
coordinate system** before submission. 
A good **sanity check** to ensure that you are submitting poses in the correct format is to 
query with a reference image and then check whether the pose matches the reference pose 
defined in the NVM model. 

## Acknowledgements
The database images used to build the 3D model for the Aachen Day-Night dataset were 
recorded by Martin Habbecke, Ming Li, Robert Menzel, Torsten Sattler, Dominik Sibbing, 
Patrick Sudowe, and Tobias Weyand.

[Mapillary](https://www.mapillary.com/) kindly blurred the images of the Aachen Day-Night 
dataset to preserve privacy.

## References:
1. J. L. Schönberger, J.-M. Frahm. Structure-from-Motion Revisited. Conference on
Computer Vision and Pattern Recognition (CVPR) 2016.
2. J. L. Schönberger, E. Zheng, M. Pollefeys, J.-M. Frahm. Pixelwise View Selection for
Unstructured Multi-View Stereo. European Conference on Computer Vision (ECCV) 2016.
3. R. Arandjelović, A. Zisserman. Three things everyone should know to improve object 
retrieval. Conference on Computer Vision and Pattern Recognition (CVPR) 2012.
4. D. Lowe. Distinctive Image Features from Scale-Invariant Keypoints. International Journal 
of Computer Vision (IJCV) 2004.
5. C. Wu. Towards Linear-time Incremental Structure From Motion. 3DV 2013
6. C. Wu. VisualSFM: A Visual Structure from Motion System. http://ccwu.me/vsfm/, 2011
7. N. Snavely, S. M. Seitz, R. Szeliski. Photo Tourism: Exploring image collections in 3D. 
ACM Transactions on Graphics (Proceedings of SIGGRAPH 2006) 2006.
