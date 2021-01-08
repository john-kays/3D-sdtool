
#include "intensitymap.h"
#include <QColor>
#include <iostream>

IntensityMap::IntensityMap() {


}

IntensityMap::IntensityMap(int width, int height) {
    map = std::vector< std::vector<double> >(height, std::vector<double>(width, 0.0));
}

void IntensityMap::InitIntensityMap(const QImage& rgbImage)
{

    map = std::vector< std::vector<double> >(rgbImage.height(), std::vector<double>(rgbImage.width(), 0.0));


    //for every row of the image
    for (int y = 0; y < rgbImage.height(); y++) {
        //for every column of the image
        for (int x = 0; x < rgbImage.width(); x++) {
            double intensity = 0.0;

            const double r = QColor(rgbImage.pixel(x, y)).redF();
            const double g = QColor(rgbImage.pixel(x, y)).greenF();
            const double b = QColor(rgbImage.pixel(x, y)).blueF();
            const double  a = QColor(rgbImage.pixel(x, y)).alphaF();
            //average of all the channels
            int channelCount = 4; //rgba
            intensity += r;
            intensity += g;
            intensity += b;
            intensity += a;
            intensity /= channelCount;
            this->map.at(y).at(x) = intensity;
        }
    }
}

double IntensityMap::at(int x, int y) const {
    return this->map.at(y).at(x);
}

double IntensityMap::at(int pos) const {
    const int x = pos % this->getWidth();
    const int y = pos / this->getWidth();

    return this->at(x, y);
}

void IntensityMap::setValue(int x, int y, double value) {
    this->map.at(y).at(x) = value;
}

void IntensityMap::setValue(int pos, double value) {
    const int x = pos % this->getWidth();
    const int y = pos / this->getWidth();

    this->map.at(y).at(x) = value;
}

size_t IntensityMap::getWidth() const {
    return this->map.at(0).size();
}

size_t IntensityMap::getHeight() const {
    return this->map.size();
}

QImage IntensityMap::convertToQImage() const {
    QImage result(this->getWidth(), this->getHeight(), QImage::Format_ARGB32);

    for(size_t y = 0; y < this->getHeight(); y++) {
        QRgb *scanline = (QRgb*) result.scanLine(y);

        for(size_t x = 0; x < this->getWidth(); x++) {
            const int c = 255 * map.at(y).at(x);
            scanline[x] = qRgba(c, c, c, 255);
        }
    }

    return result;
}
