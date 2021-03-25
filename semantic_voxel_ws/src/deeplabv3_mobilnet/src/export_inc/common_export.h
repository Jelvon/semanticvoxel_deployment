#ifndef COMMON_EXPORT_H
#define COMMON_EXPORT_H

#include "maxvision_type.h"
#include "algorithm_sdk_error_code.h"
#include <vector>
#include <opencv2/core/core.hpp>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if(defined WIN32||defined_WIN32|| defined WINCE)
#define COMMON_DLL_EXPORT __declspec(dllexport)
#else
#define COMMON_DLL_EXPORT
#endif

namespace ATHENA_algorithm {
 struct Obj_Point
{
    int x;
    int y;
};

 struct Obj_Point_Info
{
    Obj_Point obj_point;
    float   obj_prob;
    int     obj_class;
};

#define MAX_DETECT_POINTS_NUM 50
 struct Obj_Points_Info
{
    Obj_Point_Info obj_points[MAX_DETECT_POINTS_NUM];
    int obj_points_num;
};

 struct Obj_Box
{
    int x;
    int y;
    int width;
    int height;
};

 struct Obj_Info
{
    Obj_Box obj_box;
    float   obj_prob;
    int     obj_class;
};

struct Obj3D_Info
{
    float l; // 长
    float w; // 宽
    float h; // 高
    float location_x; // 物体中心点世界坐标系 x
    float location_y; // 物体中心点世界坐标系 y
    float location_z; // 物体中心点世界坐标系 z
    float rotation_y; // 物体中心点世界坐标系 旋转量
    cv::Mat box_3d; // 物体角点世界坐标系 坐标
    cv::Mat box_2d; // 物体角点世界坐标系投影到图片中的 坐标
};

struct Mono3D_Obj_Info
{
    Obj3D_Info obj_3d;
    float   obj_prob;
    int     obj_class;
};

#define MAX_DETECTOBJ_NUM 100

}
#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif


