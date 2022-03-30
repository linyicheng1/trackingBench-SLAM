//#include "Viewer.h"
//#include <iostream>
//using namespace TRACK_BENCH;
//
//int main()
//{
//    std::cout<<"test viewer"<<std::endl;
//    Viewer viewer;
//    viewer.SetCameraPos(position::Identity());
//
//    std::vector<position> kf;
////    for(int i = 0;i < 10; i ++)
//    {
//        position p = position::Identity();
////        p(12) = (float)i;
//        kf.emplace_back(p);
//    }
//
//    viewer.SetKeyFrames(kf);
//
//    std::vector<point3d> map;
//    std::vector<point3d> ref;
//    for(int i = 0;i < 100; i++)
//    {
//        map.emplace_back(point3d(i, 100 - i, 0));
//        ref.emplace_back(point3d(100 -i, i , 0));
//    }
//    viewer.SetMapPoints(map, ref);
//
//    viewer.Run();
//    std::cout<<"test end!!"<<std::endl;
//}
//
