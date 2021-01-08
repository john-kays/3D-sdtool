#ifndef INTENSITYMAP_H
#define INTENSITYMAP_H

#include <QImage>

class IntensityMap
{
public:

    IntensityMap();
    IntensityMap(int width, int height);
    void        InitIntensityMap(const QImage& rgbImage);
    double at(int x, int y) const;
    double at(int pos) const;
    void setValue(int x, int y, double value);
    void setValue(int pos, double value);
    size_t getWidth() const;
    size_t getHeight() const;
    QImage convertToQImage() const;

private:
    std::vector< std::vector<double> > map;
};

#endif // INTENSITYMAP_H
