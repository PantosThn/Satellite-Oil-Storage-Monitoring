# Thesis

This work aims to derive oil tank volumes from high-resolution satellite data in order
to satisfy the growing demand for accurate measurements of oil tank volumes around the
world. Using Otsu thresholding, shadow area thresholding, and morphological closing,
the shadow of the oil tank is retrieved from a remote sensing HSV image. The shadow of
the oil tank is in the shape of a crescent. The height and the radius of the tank are used
in the calculation of the total volume of the tank. In order to assess the viability of the
suggested method, an optical remote sensing image with a high resolution was obtained
from the Motoroil refinery located in Agioi Theodoroi, Greece.


![motor_oil_refinery_sat_image](https://user-images.githubusercontent.com/26250049/187880651-d5e1e523-cea2-4b96-96e3-41756db715d4.jpg)
*Motor Oil Refinery from Space*

Oil powers the entire world. It controls the global economy; every commodity price is
dependent on oil because commodities must be transported, and the majority of modes of
transportation rely on oil; this is why every country attempts to keep their oil production
secret for a variety of reasons. Oil is stocked in storage tanks, and data on oil production
and consumption are opaque. Various countries that are the largest oil producers con-
stantly attempt to fix oil prices to meet their needs; at the moment, the price of a barrel
of oil is at an all-time high, and as a result, various countries are constantly in conflict
and in a state of war. Oil does not merely fuel various nations; it also creates numerous
tensions between nations, which is why they never want to share all of the information
about this resource. That is why various companies such as Planet and Orbital Insight
have begun collecting satellite images and estimating the location of oil storage tanks.
Floating head tanks aid in estimating the volume of oil tanks; when oil comes into direct
contact with air, it fumes, preventing a head from sitting on top of the storage tanks.
We can calculate the volume of the tank by calculating the relative inner and exterior
shadows.

This problem can be divided into three stages.
- Object Detection : Detect whether the image is floating-head tank.
- Shadow Extraction: Extract the image of the floating-head tank.
- Volume Calculation: Calculate the volume using the extracted shadow.

Volume is estimated in some research papers by calculating the volume of the cylinder,
by calculating the height of the storage tanks using the shadow, and by calculating the
radius of the storage tanks using the Hough Transform, but in some later papers, volume
is calculated using the interior and exterior shadow lengths.

The term "object recognition" refers to a collection of closely related computer vision
tasks that involve recognizing objects in digital photographs. Classification of images
entails predicting the class of an individual object contained within an image. The term
"object localization" refers to the process of locating and drawing a bounding box around
one or more objects within an image. Object detection is a combination of these two tasks
that enables the localization and classification of one or more objects within an image.
When a user or practitioner says "object recognition," they frequently mean "object
detection." The term object recognition will be used broadly to refer to both image classi-
fication (the task of determining which object classes are present in an image) and object
detection (the task of localizing all objects present in an image).

As such, we can group these three computer vision tasks into the following categories:
- Classification of Images: Ascertain the type or class of an object contained within
an image.A single-object image, such as a photograph, as an input.As a result, a
label for the class is generated (e.g. one or more integers that are mapped to class
labels).
- Object Localization: Using a bounding box, determine the presence and location of
objects in an image.As an input, use an image containing one or more objects, such
as a photograph. As output, one or more bounding boxes (e.g. defined by a point,
width, and height).
- Object Detection: Using a bounding box and the types or classes of the detected
objects, determine the presence of objects in an image.As an input, use an image
containing one or more objects, such as a photograph.Output: One or more bound-
ing boxes (e.g. defined by a point, width, and height), each labeled with a unique
class.Another extension of this decomposition of computer vision tasks is object
segmentation, also called "object instance segmentation" or "semantic segmenta-
tion," in which instances of recognized objects are indicated by highlighting their
specific pixels rather than by using a coarse bounding box.

As can be seen from this breakdown, object recognition refers to a collection of difficult
computer vision tasks.Our objective is to determine the volume of occupied space in
floating head tanks. We could create an object detection model for a single class, but to
avoid confusion with other types of tanks (i.e. Tank/Fixed head hank and Tank Cluster),
as well as to make the model more robust, we created a three-class object detection model.
YoloV3 with transfer learning is used for object detection because it is simple to train on
non-specialized machines. Additionally, Data Augmentation is used to boost the metric
score.

On the test set, the object detection algorithm known as yolov3 was able to get an AP
score of 0.84, while on the train set it earned a score of 0.942. In addition, the amount
of crude oil barrels that were projected to be contained in the floating head tanks that
were discovered was compared to the actual number of tanks. According to the findings,
the method that was suggested is effective at calculating the capacity of oil storage tanks
to a high degree of precision and with an accuracy that is appropriate for use in actual
applications.

# The application 

The application is extremely simple to comprehend and employ. After defining the
diameter and height of the tank, a request is sent to the API that is running in the back-
ground. Following the detection of floating head tanks, the shadow extract technique is
applied to every one of the detected floating head tanks. After the user inputs the diam-
eter and height, the volume is computed by multiplying those values by the percentage
returned by the back-end API as a json file. Then, on streamlit, basic calculations such
as the total number of barrels are performed and sent to the user so they can view the
results.

![results_4](https://user-images.githubusercontent.com/26250049/187878276-c782cac0-d626-4de1-be8d-080fb43b126a.png)


# Shadow extraction algorithm

Identified tanks will be run through a computer vision algorithm to extract the crescent shadows and estimate tank volumes. 
In short, the algorithm creates an enhanced version of the tank image using channels from RGB and LAB color space. Shadows are extracted from the enhanced image using thresholding. The thresholded image is cleaned up using morphological operations. Tank shadows are extracted from the clean image and used for volume estimation.

## How the shadow extraction algorithm works

We first create two versions of the image - one in HSV color space and one in LAB color space

![shadows_1](https://user-images.githubusercontent.com/26250049/187880635-54f7ce19-fa79-4a53-9fce-c083429906fc.PNG)

## Enhance shadows

Many methods have been proposed for ratioing these channels to enhance shadows. The NSVDI algorithm proposes (S−V)/(S+V)
The paper Estimating the Volume of Oil Tanks Based on High-Resolution Remote Sensing which explicitly deals with oil tanks suggests (H+1)/(V+1).
I found the (H+1)/(V+1)was thrown off by strong artifacts in the H channel, likely due to the source images being RGB jpeg images saved from Google Earth rather than real high resolution satellite photography. The (S−V)/(S+V) method worked well on some images but failed on others.
Experimentally I found the ratio −(l1+l3)/V+1 worked well, where l1 and l3 are the first and third channels of the LAB color space image.

![shadows_2](https://user-images.githubusercontent.com/26250049/187880605-f03bb5de-31fe-4512-a1b2-0312a4bc860a.PNG)

## Thresholding

In digital image processing, thresholding is one of the simplest methods of segmenting images. Basic thresholding methods replace each pixel in an image with a black pixel if the image intensity 'I' is less than some fixed constant 'T', or a white pixel if the image intensity is greater than that constant.


![shadows_3](https://user-images.githubusercontent.com/26250049/187880600-18a9a822-aa28-4625-92a4-c079666ccd27.PNG)

## Morphological Operations

- Hessian Filter - cleans up noise and line artifacts from white pipes which appear in many images
- Clear Border - clears contours from surrounding tanks
- Morphological Closing - helps separate shapes
- Area Closing - fills small holes
- Morphological Labeling - labels features

![shadows_4](https://user-images.githubusercontent.com/26250049/187880594-420b223c-765a-4136-ae8f-7b31fd41695d.PNG)

# Volume Estimation

We then filter the regions present by certain heuristics.

The bounding box of the feature should intersect the bounding box of the tank. The feature should have an area of more than 25 pixels. The pixel coverage of the labeled image should be approximately the same as in the threshold image.

The first two clear up small artifacts. The third deals with the fact that the Hessian filter sometimes creates regions in spaces that are otherwise empty.

![shadows_8](https://user-images.githubusercontent.com/26250049/187880582-942cf859-fbac-4b04-a34c-11af94cd9dcb.PNG)

## Calculating Volume and Number of Barrels

Volume is estimated as 1 minus the ratio of the smaller area to the larger area.
The larger area corresponds to the exterior shadow of the tank, while the smaller area
corresponds to the interior shadow.

Several methods exist for measuring and quantifying volumes of petroleum [15] (oil,
condensate, NGL, and gas). Quantities can be expressed in terms of mass (weight),
volume, and energy density. Various units, such as cubic metres and barrels, both of
which indicate volume, may be used to express the various numbers.
In the petroleum sector, older American (British) units are commonly utilized.When
doing conversions, such as from volume to energy content, there is no exact conversion
factor, and you must know/make assumptions about the substance’s attributes. As-
sumptions regarding energy content per cubic meter of gas and weight per volume unit of
natural gas liquids are examples.Volumes of oil and gas are often expressed in standard
cubic metres (Sm3), and the temperature and pressure at which they apply must also
be specified for an accurate representation of volumes. The standard conditions are 15
degrees Celsius and normal air pressure (1013.25 hPa).

Before amounts of different petroleum products (oil, gas, NGL, and condensate) can
be added, they must be converted to a standardized quantity and unit. Using standard
cubic metres of oil equivalents is the most prevalent technique (abbreviated as Sm3 o.e.).
When doing conversions, the Norwegian Petroleum Directorate applies a volumetric
conversion of NGL to liquid and an energy-based (but not accurate) conversion factor
for gas based on average shelf features. The characteristics of oil, gas, and NGL vary
over time and from field to field. In resource reports and other comparisons involving oil
equivalents, however, a consistent and uniform conversion factor is applied to all fields
and finds.

# Conclusion

In the method suggested in this work, remote sensing photos were converted into HSV
images, and a ratio image was created to highlight the shadows. The ratio image was
thresholded using the highest inter-class variance of the levels, followed by thresholding
according to area and processing using morphological operators to produce the oil tank’s
shadow.
In the future, Hough transform can be tested to detect the radius of the tank’s top
thus the tank’s volume can be determined with sufficient precision, we will do additional
research on the nature of shadows in various color spaces and combine the texture
characteristics of shadows in order to extract tank shadows from a larger variety of photos.
The Hough transform and machine-learning-related techniques should be further coupled
in the future to increase the accuracy of the tank identification rate and tank volume
estimation.

