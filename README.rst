Mcrops
=======================
mcrops is small library for processing images of agricultural fields. 
The homepage of mcrops with user documentation is located on:

https://mcrops.readthedocs.io

Getting started
==========

You can use `pip` to install mcrops::

    pip install mcrops

Alternatively, you can download the latest source code using git::

    git clone git://github.com/raikel/mcrops.git

and run the setup command from the source directory::

    python setup.py install

from the source directory.

Example usages
============

Here is some example usages of mcrops.

Vegetation analysis
---

.. code-block:: python

    from mcrops import veget, utils
    import cv2 as cv
    
    # Path to crop field image
    image_path = 'path/to/image'
    # Image resolution in pixels/meter (it has not to be an exact value)
    res = 20

    # Load a crop field image
    image = cv.imread(image_path)

    h, w = image.shape[:2]
    image_draw = image.copy()
    
    # Segment vegetation
    veg_mask = veget.segment_vegetation(image)

    # Detect the crop field ROI area
    roi_poly = veget.detect_roi(veg_mask, row_sep=0.7, resolution=res)
    # Draw the contours of the ROI area
    cv.drawContours(
        image=image_draw,
        contours=[roi_poly],
        contourIdx=-1,
        color=(0, 0, 255),
        thickness=8,
        lineType=cv.LINE_AA
    )
    # Build a mask image from the ROI polyline
    roi_mask = utils.poly_mask(roi_poly, veg_mask.shape[:2])
    veg_mask[roi_mask == 0] = 0

    # Create a vegetation density map from the vegetation mask
    density_map = veget.mask_density(
        veg_mask,
        roi_mask,
        resolution=res,
        cell_size=(8, 8)
    )

    # Convert the vegetation density map to a color image
    image_map = utils.array_image(density_map, colormap=cv.COLORMAP_JET)

    cv.imshow('Crop field image', image_draw)
    cv.imshow('Vegetation mask', veg_mask)
    cv.imshow('Vegetation density map', image_map)

Workflow to contribute
======================

To contribute to mcrops, first create an account on `github
<http://github.com/>`_. Once this is done, fork the `mcrops repository
<http://github.com/raikel/mcrops>`_ to have your own repository,
clone it using 'git clone' on the computers where you want to work. Make
your changes in your clone, push them to your github account, test them
on several computers, and when you are happy with them, send a pull
request to the main repository.