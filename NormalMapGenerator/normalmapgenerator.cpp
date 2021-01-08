#include "normalmapgenerator.h"
#include <QVector3D>
#include <QColor>

NormalmapGenerator::NormalmapGenerator()
{
}

const IntensityMap& NormalmapGenerator::getIntensityMap() const {
    return this->intensity;
}


QImage NormalmapGenerator::calculateNormalmap(const QImage& input, double strength,bool tileable) {
    this->tileable = tileable;

    this->intensity.InitIntensityMap(input);

    QImage result(input.width(), input.height(), QImage::Format_ARGB32);
    
    // optimization
    double strengthInv = 1.0 / strength;

    for(int y = 0; y < input.height(); y++) {
        QRgb *scanline = (QRgb*) result.scanLine(y);

        for(int x = 0; x < input.width(); x++) {

            const double topLeft      = intensity.at(handleEdges(x - 1, input.width()), handleEdges(y - 1, input.height()));
            const double top          = intensity.at(handleEdges(x - 1, input.width()), handleEdges(y,     input.height()));
            const double topRight     = intensity.at(handleEdges(x - 1, input.width()), handleEdges(y + 1, input.height()));
            const double right        = intensity.at(handleEdges(x,     input.width()), handleEdges(y + 1, input.height()));
            const double bottomRight  = intensity.at(handleEdges(x + 1, input.width()), handleEdges(y + 1, input.height()));
            const double bottom       = intensity.at(handleEdges(x + 1, input.width()), handleEdges(y,     input.height()));
            const double bottomLeft   = intensity.at(handleEdges(x + 1, input.width()), handleEdges(y - 1, input.height()));
            const double left         = intensity.at(handleEdges(x,     input.width()), handleEdges(y - 1, input.height()));

            const double convolution_kernel[3][3] = {{topLeft, top, topRight},
                                               {left, 0.0, right},
                                               {bottomLeft, bottom, bottomRight}};

            QVector3D normal(0, 0, 0);

            normal = sobel(convolution_kernel, strengthInv);

            scanline[x] = qRgb(mapComponent(normal.x()), mapComponent(normal.y()), mapComponent(normal.z()));
        }
    }
    return result;
}

QVector3D NormalmapGenerator::sobel(const double convolution_kernel[3][3], double strengthInv) const {
    const double top_side    = convolution_kernel[0][0] + 2.0 * convolution_kernel[0][1] + convolution_kernel[0][2];
    const double bottom_side = convolution_kernel[2][0] + 2.0 * convolution_kernel[2][1] + convolution_kernel[2][2];
    const double right_side  = convolution_kernel[0][2] + 2.0 * convolution_kernel[1][2] + convolution_kernel[2][2];
    const double left_side   = convolution_kernel[0][0] + 2.0 * convolution_kernel[1][0] + convolution_kernel[2][0];

    const double dY = right_side - left_side;
    const double dX = bottom_side - top_side;
    const double dZ = strengthInv;

    return QVector3D(dX, dY, dZ).normalized();
}

int NormalmapGenerator::handleEdges(int iterator, int maxValue) const {
    if(iterator >= maxValue) {
        //move iterator from end to beginning + overhead
        if(tileable)
            return maxValue - iterator;
        else
            return maxValue - 1;
    }
    else if(iterator < 0) {
        //move iterator from beginning to end - overhead
        if(tileable)
            return maxValue + iterator;
        else
            return 0;
    }
    else {
        return iterator;
    }
}

//convert - to 1 to 0 to 255 for rgba
int NormalmapGenerator::mapComponent(double value) const {
    return (value + 1.0) * (255.0 / 2.0);
}