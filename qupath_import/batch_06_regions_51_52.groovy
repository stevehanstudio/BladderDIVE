// Aperio XML import batch 6 for QuPath 0.6.0
// Imports regions 51 to 52

import qupath.lib.objects.PathAnnotationObject
import qupath.lib.objects.classes.PathClass
import qupath.lib.roi.PolygonROI
import qupath.lib.regions.ImagePlane
import qupath.lib.geom.Point2

print("Starting batch 6 import (regions 51-52)...")

try {
    def annotations = []
    def pathClass = PathClass.fromString("Region")

    // Region 52 (area: 475200.0 pixels)
    def points0 = [
        new Point2(17920.0, 87760.0),
        new Point2(17920.0, 87740.0),
        new Point2(18060.0, 87580.0),
        new Point2(18200.0, 87440.0),
        new Point2(18200.0, 87420.0),
        new Point2(18240.0, 87400.0),
        new Point2(18240.0, 87380.0),
        new Point2(18340.0, 87360.0),
        new Point2(18420.0, 87340.0),
        new Point2(18460.0, 87340.0),
        new Point2(18480.0, 87340.0),
        new Point2(18500.0, 87340.0),
        new Point2(18560.0, 87340.0),
        new Point2(18620.0, 87340.0),
        new Point2(18740.0, 87380.0),
        new Point2(18940.0, 87460.0),
        new Point2(19080.0, 87520.0),
        new Point2(19180.0, 87560.0),
        new Point2(19280.0, 87620.0),
        new Point2(19380.0, 87660.0),
        new Point2(19540.0, 87720.0),
        new Point2(19820.0, 87760.0),
        new Point2(19900.0, 87780.0),
        new Point2(20100.0, 87800.0),
        new Point2(20140.0, 87800.0),
        new Point2(20200.0, 87800.0)
    ]
    def roi0 = new PolygonROI(points0, ImagePlane.getDefaultPlane())
    def annotation0 = new PathAnnotationObject(roi0, pathClass)
    annotation0.setName("Region_52")
    annotations.add(annotation0)
    print("Created region 52")

    // Region 53 (area: 137600.0 pixels)
    def points1 = [
        new Point2(23760.0, 87720.0),
        new Point2(23780.0, 87720.0),
        new Point2(23800.0, 87660.0),
        new Point2(23840.0, 87580.0),
        new Point2(23880.0, 87560.0),
        new Point2(23940.0, 87500.0),
        new Point2(23960.0, 87480.0),
        new Point2(24020.0, 87380.0),
        new Point2(24040.0, 87360.0),
        new Point2(24060.0, 87360.0),
        new Point2(24080.0, 87360.0),
        new Point2(24120.0, 87360.0),
        new Point2(24140.0, 87400.0),
        new Point2(24240.0, 87500.0),
        new Point2(24300.0, 87620.0),
        new Point2(24320.0, 87700.0),
        new Point2(24340.0, 87720.0),
        new Point2(24360.0, 87720.0),
        new Point2(24400.0, 87720.0),
        new Point2(24440.0, 87740.0),
        new Point2(24460.0, 87760.0)
    ]
    def roi1 = new PolygonROI(points1, ImagePlane.getDefaultPlane())
    def annotation1 = new PathAnnotationObject(roi1, pathClass)
    annotation1.setName("Region_53")
    annotations.add(annotation1)
    print("Created region 53")

    print("Adding " + annotations.size() + " annotations to image...")
    addObjects(annotations)
    fireHierarchyUpdate()
    print("SUCCESS: Batch 6 completed - imported " + annotations.size() + " regions!")

} catch (Exception e) {
    print("Error in batch {batch_num + 1}: " + e.getMessage())
    e.printStackTrace()
}