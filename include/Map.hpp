#pragma once

#include "MapPoint.hpp"
#include "KeyFrame.hpp"
#include <set>

#include <mutex>

namespace ORB_SLAM_CUSTOM
{

class MapPoint;
class KeyFrame;

class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Map();

    void AddKeyFrame(std::shared_ptr<KeyFrame> pKF);
    void AddMapPoint(std::shared_ptr<MapPoint> pMP);
    void EraseMapPoint(std::shared_ptr<MapPoint> pMP);
    void EraseKeyFrame(std::shared_ptr<KeyFrame> pKF);
    void SetReferenceMapPoints(const std::vector<std::shared_ptr<MapPoint>> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    std::vector<std::shared_ptr<KeyFrame>> GetAllKeyFrames();
    std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();
    std::vector<std::shared_ptr<MapPoint>> GetReferenceMapPoints();

    long unsigned int MapPointsInMap();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    vector<std::shared_ptr<KeyFrame>> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

protected:
    std::set<std::shared_ptr<MapPoint>> mspMapPoints;
    std::set<std::shared_ptr<KeyFrame>> mspKeyFrames;

    std::vector<std::shared_ptr<MapPoint>> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM
