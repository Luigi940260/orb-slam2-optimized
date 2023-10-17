#pragma once

#include <map>
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM_CUSTOM
{

class KeyFrame;

typedef std::pair<std::set<std::shared_ptr<KeyFrame>>,int> ConsistentGroup;    
typedef std::map<std::shared_ptr<KeyFrame>,g2o::Sim3,std::less<std::shared_ptr<KeyFrame>>,
    	Eigen::aligned_allocator<std::pair<std::shared_ptr<KeyFrame> const, g2o::Sim3> > > KeyFrameAndPose;

}