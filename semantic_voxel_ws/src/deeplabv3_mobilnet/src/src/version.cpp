#include "algorithm_sdk.h"


#define  ALG_VERSION  "V0.0.0.1 "
static char   g_chALGVersion[128];
#ifdef TK1_DEVICE
static const char * LIB_BUILD_TIME = __DATE__", " __TIME__;
static const char * LIB_PLATFORM = "tk1";
#elif ARM64_DEVICE
static const char * LIB_BUILD_TIME = __DATE__", " __TIME__;
static const char * LIB_PLATFORM = "aarch64";
#else
static const char * LIB_BUILD_TIME = __DATE__", " __TIME__;
static const char * LIB_PLATFORM = "x86_64";
#endif
namespace ATHENA_algorithm{


    const char * GetLibraryVersion(){
        memset(g_chALGVersion,0,sizeof(g_chALGVersion));
        #ifdef WIN32
        sprintf_s(g_chALGVersion, sizeof(g_chALGVersion), "%s build on %s", ALG_VERSION,  LIB_BUILD_TIME);
        #else
        ::snprintf(g_chALGVersion, sizeof(g_chALGVersion), "%s build on %s %s", ALG_VERSION, LIB_PLATFORM ,LIB_BUILD_TIME);
        #endif
        return  g_chALGVersion;
    }

}