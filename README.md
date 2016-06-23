# Content Based Image Retrival

# Things you need.
* [OpenCV](http://opencv.org/)
* [numpy](http://www.numpy.org/)
* [pywt](http://www.pybytes.com/pywavelets/)
* [Scipy](https://www.scipy.org/)

# How it works
Basic Details :
(Based on research by Yen Do et al.)[http://dl.acm.org/citation.cfm?id=2448648&dl=ACM&coll=DL&CFID=804778667&CFTOKEN=26088710]

Optimization Details :
(Our Approach)[http://devashishpurandare.me/assets/pdf/hcbir.pdf]


# Organization

- initial.py - Without database, unoptimized.
- feature.py - Feature extraction, similarity measurement,
- optimized.py - With database, optimized.
- prelib.py - Preprocessing and standardization.
