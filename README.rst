Mcrops
=======================
mcrops is small library for processing images of agricultural fields. 
The homepage of mcrops with user documentation is located on:

https://mcrops.readthedocs.io

![Sample output from mcrops](images/sample_output.jpg?raw=true "Sample output")

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

Here is some example usages of `mcrops`.

.. code-block:: python

    import math    
    import cv2 as cv    
    from mcrops import veget, rows, utils
    
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
    roi_poly = veget.detect_roi(
        veg_mask, row_sep=row_sep, resolution=res
    )
    # Draw the contours of the ROI area
    cv.drawContours(
        image=image_draw,
        contours=[roi_poly],
        contourIdx=-1,
        color=(0, 0, 255),
        thickness=8,
        lineType=cv.LINE_AA
    )

    # Detect the mean crop rows direction
    direction = rows.detect_direction(
        veg_mask=veg_mask,
        window_shape=(20, 30),
        resolution=res
    )
    
    # Draw an arrow indicating the direction of the crop rows
    pt1 = (int(w/2), int(h/2))
    length = res * 20
    pt2 = (
        pt1[0] + min(max(0, int(math.cos(direction) * length)), w - 1),
        pt1[1] + min(max(0, int(math.sin(direction) * length)), h - 1)
    )
    cv.arrowedLine(image_draw, pt1, pt2, color=(0, 255, 0), thickness=8)

    # Normalize the crop field image and the vegetation mask (trim then to ROI
    # area and rotate them such that crop rows are horizontal)
    image_rows, roi_poly_norm, _ = veget.norm_image(
        image=image,
        roi_poly=roi_poly,
        rows_direction=direction
    )
    veg_mask, roi_poly_norm, _ = veget.norm_image(
        image=veg_mask,
        roi_poly=roi_poly,
        rows_direction=direction,
        is_mask=True
    )
    # Build a mask image from the ROI polyline
    roi_mask = utils.poly_mask(roi_poly_norm, veg_mask.shape[:2])

    # Create a row-oriented vegetation density map from the vegetation mask
    density_map = veget.mask_density(
        veg_mask,
        roi_mask,
        resolution=res,
        cell_size=(8, 8)
    )

    # Convert the row-oriented vegetation density map to a color image
    density_image = utils.array_image(density_map, colormap=cv.COLORMAP_JET)

    # Detect the crop rows (ridges and furrows)
    row_ridges, row_furrows = rows.detect_rows(
        veg_mask,
        roi_mask,
        resolution=res,
        row_sep=row_sep,
        fusion_thr=0.4
    )
    # Draw the crop rows lines
    image_rows = utils.draw_rows(image_rows, row_ridges)

    cv.imshow('Crop field image', image_draw)
    cv.imshow('Vegetation mask', veg_mask)
    cv.imshow('Vegetation density map', density_image)
    cv.imshow('Detected crop rows', image_rows)

    cv.waitKey(0)
    cv.destroyAllWindows()

Workflow to contribute
======================

To contribute to mcrops, first create an account on `github
<http://github.com/>`_. Once this is done, fork the `mcrops repository
<http://github.com/raikel/mcrops>`_ to have your own repository,
clone it using 'git clone' on the computers where you want to work. Make
your changes in your clone, push them to your github account, test them
on several computers, and when you are happy with them, send a pull
request to the main repository.