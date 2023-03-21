//#include "cuda_runtime_api.h"
//#include "opencv2/opencv.hpp"
//#include <fstream>
#include <iostream>
//#include <chrono>
//#include "nvjpeg.h"
//#include "kernels.h"
#include "camera_collect_worker.h"
#include "gnss_collect_worker.h"
#include <map>
#include <stdio.h>
#include <string.h>
#include "cfg.h"

void syncv4l2EventCallBack(int nEvent, void *pHint1, void *pHint2, void *pUserData)
{
    switch (nEvent)
    {
    case SYNCV4L2_TIMESTAMP_ERROR:
    case SYNCV4L2_FRAME_ERROR:
        /*为了多通道同步，这里需要软重启*/
        SYNCV4L2_Stop();
        SYNCV4L2_Start();
        break;
    case SYNCV4L2_FRAME_LOSE:
    {
        SYNCV4L2_TEventFrameHint *pFrameEvent = (SYNCV4L2_TEventFrameHint *)pHint1;
        printf("frame lose,chan[%d],timestame[%ld.%09ld]\n", pFrameEvent->nChan, pFrameEvent->stTime.tv_sec, pFrameEvent->stTime.tv_nsec);
    }
    break;
    default:
        break;
    }
}

int main(int argc, char** argv) {
    if (argc != 4)
    {
        printf("sudo %s  <cfgname> <module_name> <saving dir>\n", argv[0]);
        return -1;
    }

    //init
	std::string strCfg = argv[1];
    CFG_init(strCfg.c_str());

    SYNCV4L2_TSyncPara stuSyncPara;
    memset(&stuSyncPara, 0, sizeof(stuSyncPara));
    stuSyncPara.nMode = CFG_get_section_value_int(strCfg.c_str(), "common", "gps_mode", 0);
    stuSyncPara.nInitTriggerSignal = CFG_get_section_value_int(strCfg.c_str(), "common", "init_trigger_signal", 0);
    stuSyncPara.nSlave = CFG_get_section_value_int(strCfg.c_str(), "common", "slave", 0);
    stuSyncPara.nVersion = CFG_get_section_value_int(strCfg.c_str(), "common", "version", 1);
    CFG_get_section_value(strCfg.c_str(), "common", "trigger_dev_name", stuSyncPara.szDevname, sizeof(stuSyncPara.szDevname));
    stuSyncPara.nReset = CFG_get_section_value_int(strCfg.c_str(), "common", "reset", 0);
    SYNCV4L2_Init(&stuSyncPara);
    SYNCV4L2_SetEventCallback(syncv4l2EventCallBack, nullptr);

    //init camera chan
    std::map<int, std::unique_ptr<CameraCollectWorker>> mapJpeg;
	const int MAX_CAMER_NUM = 8;
    std::shared_ptr<PBWriter> writer(new PBWriter(argv[2], argv[3], (uint64_t)2000 * 1024 * 1024));
    writer->Open();
    for (int i = 0; i < MAX_CAMER_NUM; i++)
    {
        int chan = i;
		std::string tag = "video" + std::to_string(chan);

        if (CFG_get_section_value_int(strCfg.c_str(), tag.c_str(), "enable", 0) == 0)
        {
            continue;
        }

        mapJpeg[chan].reset(new (std::nothrow) CameraCollectWorker(argv[2], chan, strCfg, writer));
        if (mapJpeg.at(chan) == nullptr)
        {
            printf("new (std::nothrow)CameraCollectWorker failed,chan=%d\n", chan);
            continue;
        }

        mapJpeg.at(chan)->Init();
        mapJpeg.at(chan)->Open();
    }
    std::unique_ptr<GNSSCollectWorker> gnss(new (std::nothrow)GNSSCollectWorker(42, 115200, writer));
    gnss->Init();
    //gnss->Open();

    //start
    SYNCV4L2_Start();
	
	getchar();

    //stop
    SYNCV4L2_Stop();
    for (auto& it : mapJpeg)
    {
        it.second->Release();
        it.second.reset();
    }
    SYNCV4L2_Release();
    //gnss->Release();
    writer->Close();
    CFG_free(strCfg.c_str());
 
    return 0;
}
