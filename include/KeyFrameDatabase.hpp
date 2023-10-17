#pragma once

#include <vector>
#include <list>
#include <set>

#include "KeyFrame.hpp"
#include "Frame.hpp"
#include "ORBVocabulary.hpp"

#include<mutex>
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"


namespace ORB_SLAM_CUSTOM
{

class KeyFrame;
class Frame;


class KeyFrameDatabase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KeyFrameDatabase(const std::shared_ptr<ORBVocabulary> voc);

  void add(std::shared_ptr<KeyFrame> pKF);

  void erase(std::shared_ptr<KeyFrame> pKF);

  void clear();

  // Loop Detection
  std::vector<std::shared_ptr<KeyFrame>> DetectLoopCandidates(std::shared_ptr<KeyFrame> pKF, float minScore);

  // Relocalization
  std::vector<std::shared_ptr<KeyFrame>> DetectRelocalizationCandidates(Frame* F);

protected:

  // Associated vocabulary
  const std::shared_ptr<ORBVocabulary> mpVoc;

  // Inverted file
  std::vector<list<std::shared_ptr<KeyFrame>> > mvInvertedFile;

  // Mutex
  std::mutex mMutex;
};

} //namespace ORB_SLAM
