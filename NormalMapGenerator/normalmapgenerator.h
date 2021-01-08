

#ifndef NORMALMAP_GEN_H
#define NORMALMAP_GEN_H

#include <QImage>
#include "intensitymap.h"

class NormalmapGenerator
{
public:

    NormalmapGenerator();
    QImage calculateNormalmap(const QImage& input, double strength = 2.0, 
                              bool tileable = true);
    const IntensityMap& getIntensityMap() const;

private:
    IntensityMap intensity;
    bool tileable;
    double redMultiplier, greenMultiplier, blueMultiplier, alphaMultiplier;

    int handleEdges(int iterator, int maxValue) const;
    int mapComponent(double value) const;
    QVector3D sobel(const double convolution_kernel[3][3], double strengthInv) const;
};

#endif // SOBEL_H
